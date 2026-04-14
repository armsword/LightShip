//! Graph Executor for running inference
//!
//! This module provides end-to-end model execution using the backend.

use std::fmt::Debug;
use std::collections::HashMap;
use std::sync::Arc;
use crate::backend::{Backend, CpuBackend};
use crate::backend::CompiledOperator;
use crate::common::{LightShipError, Result};
use crate::ir::{Graph, NodeId, OperatorDef, OperatorType, Tensor};

/// Graph executor for running inference
pub struct GraphExecutor {
    backend: Arc<dyn Backend + Send + Sync>,
    compiled_ops: HashMap<String, CompiledOperator>,
}

impl Debug for GraphExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphExecutor")
            .field("num_compiled_ops", &self.compiled_ops.len())
            .finish()
    }
}

impl GraphExecutor {
    /// Create a new graph executor with Arc<dyn Backend>
    pub fn new(backend: Arc<dyn Backend + Send + Sync>) -> Self {
        Self {
            backend,
            compiled_ops: HashMap::new(),
        }
    }

    /// Prepare a graph for execution by compiling all operators
    pub fn prepare(&mut self, graph: &Graph) -> Result<()> {
        // Compile all operators in the graph
        for node in &graph.nodes {
            let mut def = OperatorDef::new(
                node.name.clone(),
                node.operator_type,
            );

            // Build input/output tensor specs
            for input in &node.inputs {
                def.inputs.push(crate::ir::NodeIO {
                    tensor_name: input.tensor_name.clone(),
                    data_type: input.data_type,
                });
            }
            for output in &node.outputs {
                def.outputs.push(crate::ir::NodeIO {
                    tensor_name: output.tensor_name.clone(),
                    data_type: output.data_type,
                });
            }

            // Compile using backend
            let compiled = self.backend.as_ref().compile_operator(&def, node.fusion.as_ref(), &[], &[])?;
            self.compiled_ops.insert(node.name.clone(), compiled);
        }

        tracing::info!("Prepared graph with {} operators", graph.nodes.len());
        Ok(())
    }

    /// Execute the graph with given inputs
    pub fn execute(
        &self,
        graph: &Graph,
        inputs: &[(&str, Tensor)],
        outputs: &mut [(&str, Tensor)],
    ) -> Result<()> {
        // Create tensor storage for intermediate results
        // Use Arc to share tensor data without deep cloning
        let mut tensor_storage: HashMap<String, Arc<Tensor>> = HashMap::new();

        // Build a map from output name to index for quick lookup (use String to avoid borrow issues)
        let output_names: Vec<String> = outputs.iter().map(|(name, _)| name.to_string()).collect();
        let output_indices: HashMap<&str, usize> = output_names
            .iter()
            .enumerate()
            .map(|(i, name)| (name.as_str(), i))
            .collect();

        // Register input tensors - Arc clone is cheap (just reference count increment)
        for (name, tensor) in inputs {
            tensor_storage.insert(name.to_string(), Arc::new(tensor.clone()));
        }

        // Get topological order
        let order = graph.topological_sort();

        // Execute nodes in order
        for &node_id in &order {
            // Find node by traversing (nodes are stored by insertion order, not by id)
            let node = graph.nodes.iter()
                .find(|n| n.id == node_id)
                .ok_or_else(|| LightShipError::InvalidParam(format!("Node {} not found", node_id)))?;

            // Get input tensors - use Arc clone to share data
            let mut input_tensors: Vec<Arc<Tensor>> = Vec::new();
            for input in &node.inputs {
                if let Some(tensor) = tensor_storage.get(&input.tensor_name) {
                    input_tensors.push(Arc::clone(tensor));
                } else if let Some(var_tensor) = graph.variables.get(&input.tensor_name) {
                    // Use variable (initializer/weight) from graph
                    tracing::debug!("Using initializer {} for node {}", input.tensor_name, node.name);
                    tracing::debug!("  shape: {:?}, data_len: {}", var_tensor.shape, var_tensor.data_as_bytes().len());
                    input_tensors.push(Arc::clone(var_tensor));
                } else {
                    // Create a placeholder tensor for missing inputs
                    tracing::warn!("Input tensor {} not found for node {}, using placeholder",
                        input.tensor_name, node.name);
                    let placeholder = Tensor::new(
                        input.tensor_name.clone(),
                        vec![1],
                        input.data_type,
                    );
                    let placeholder_arc = Arc::new(placeholder);
                    tensor_storage.insert(input.tensor_name.clone(), Arc::clone(&placeholder_arc));
                    input_tensors.push(placeholder_arc);
                }
            }

            // Convert Arc<Tensor> to &Tensor for backend execute
            let input_tensor_refs: Vec<&Tensor> = input_tensors.iter().map(|t| t.as_ref()).collect();

            // Get output tensors - create new or get from storage
            // For each output, we need a mutable Tensor to pass to execute
            let mut output_arcs: Vec<Arc<Tensor>> = Vec::new();
            let mut output_refs: Vec<&mut Tensor> = Vec::new();

            for output in &node.outputs {
                // Check if this is a model output (pre-allocated by caller)
                let is_model_output = output_indices.contains_key(output.tensor_name.as_str());

                let tensor_arc = if is_model_output {
                    // For model outputs, we need to use the actual output tensor from caller
                    // Get the mutable reference from outputs array
                    let idx = *output_indices.get(output.tensor_name.as_str()).unwrap();
                    // Clone the tensor and we'll track that this should be written back
                    Arc::new(outputs[idx].1.clone())
                } else if let Some(existing) = tensor_storage.get(&output.tensor_name) {
                    Arc::clone(existing)
                } else {
                    Arc::new(Tensor::new(output.tensor_name.clone(), vec![1], output.data_type))
                };

                output_arcs.push(Arc::clone(&tensor_arc));

                // Get mutable reference for execute
                // Safety: We ensure single-threaded execution and exclusive access pattern
                let tensor_ptr = Arc::as_ptr(&tensor_arc) as *mut Tensor;
                output_refs.push(unsafe { &mut *tensor_ptr });
            }

            // Get compiled operator
            let compiled = self.compiled_ops.get(&node.name)
                .ok_or_else(|| LightShipError::InvalidParam(
                    format!("Compiled operator {} not found", node.name)
                ))?;

            // Execute
            self.backend.as_ref().execute(
                compiled,
                &input_tensor_refs,
                &mut output_refs,
            )?;

            // Store outputs back to tensor_storage and update model outputs
            for (i, output) in node.outputs.iter().enumerate() {
                let tensor_arc = &output_arcs[i];
                let is_model_output = output_indices.contains_key(output.tensor_name.as_str());

                if is_model_output {
                    // Update the actual output tensor in the outputs array
                    let idx = *output_indices.get(output.tensor_name.as_str()).unwrap();
                    outputs[idx].1.data = tensor_arc.data.clone();
                } else {
                    // Store intermediate result in tensor_storage
                    tensor_storage.insert(output.tensor_name.clone(), Arc::clone(tensor_arc));
                }
            }
        }

        tracing::debug!("Graph execution completed");
        Ok(())
    }

    /// Compute topological levels for parallel execution.
    /// Returns nodes grouped by level where all nodes in a level are independent.
    fn compute_levels(&self, graph: &Graph) -> Vec<Vec<NodeId>> {
        let n = graph.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        // in_degree[node_id] = number of unresolved input dependencies
        let mut in_degree = vec![0u32; n];
        // adjacency[producer] = list of nodes that depend on producer
        let mut adjacency: std::collections::HashMap<NodeId, Vec<NodeId>> =
            std::collections::HashMap::new();

        for node in &graph.nodes {
            for input in &node.inputs {
                if let Some(producer_id) = graph.find_tensor_producer(&input.tensor_name) {
                    adjacency.entry(producer_id).or_default().push(node.id);
                    in_degree[node.id as usize] += 1;
                }
            }
        }

        let mut levels: Vec<Vec<NodeId>> = Vec::new();
        let mut current: Vec<NodeId> = (0..n as NodeId)
            .filter(|&id| in_degree[id as usize] == 0)
            .collect();

        while !current.is_empty() {
            levels.push(current.clone());
            let mut next = Vec::new();
            for &node_id in &current {
                if let Some(deps) = adjacency.get(&node_id) {
                    for &dep in deps {
                        in_degree[dep as usize] -= 1;
                        if in_degree[dep as usize] == 0 {
                            next.push(dep);
                        }
                    }
                }
            }
            current = next;
        }

        levels
    }

    /// Execute the graph with parallel execution of independent nodes (level-by-level).
    pub fn execute_parallel(
        &self,
        graph: &Graph,
        inputs: &[(&str, Tensor)],
        outputs: &mut [(&str, Tensor)],
    ) -> Result<()> {
        let mut tensor_storage: std::collections::HashMap<String, Arc<Tensor>> =
            std::collections::HashMap::new();

        let output_names: Vec<String> = outputs.iter().map(|(n, _)| n.to_string()).collect();
        let output_indices: std::collections::HashMap<&str, usize> = output_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();

        for (name, tensor) in inputs {
            tensor_storage.insert(name.to_string(), Arc::new(tensor.clone()));
        }

        // Get topological levels (nodes at same level are independent)
        let levels = self.compute_levels(graph);

        for level in &levels {
            // Collect inputs for all nodes in this level upfront
            let level_inputs: Vec<_> = level
                .iter()
                .map(|&node_id| {
                    let node = graph.nodes.get(node_id as usize).unwrap();
                    let node_inputs: Vec<Arc<Tensor>> = node
                        .inputs
                        .iter()
                        .map(|input| {
                            if let Some(t) = tensor_storage.get(&input.tensor_name) {
                                Arc::clone(t)
                            } else if let Some(v) = graph.variables.get(&input.tensor_name) {
                                Arc::clone(v)
                            } else {
                                Arc::new(Tensor::new(
                                    input.tensor_name.clone(),
                                    vec![1],
                                    input.data_type,
                                ))
                            }
                        })
                        .collect();
                    (node_id, node_inputs)
                })
                .collect();

            // Execute all nodes in this level in parallel using scoped threads
            type NodeResult = Result<Vec<(String, Arc<Tensor>)>>;
            let mut results: Vec<NodeResult> = Vec::with_capacity(level.len());

            std::thread::scope(|s| {
                let mut handles: Vec<_> = level_inputs
                    .iter()
                    .map(|&(node_id, ref node_inputs)| {
                        let node = graph.nodes.get(node_id as usize).unwrap();
                        let compiled = self
                            .compiled_ops
                            .get(&node.name)
                            .unwrap_or_else(|| {
                                panic!("Compiled operator {} not found", node.name)
                            });

                        s.spawn(move || {
                            execute_node_single(
                                node,
                                compiled,
                                node_inputs.iter().map(|t| t.as_ref()).collect::<Vec<_>>(),
                            )
                        })
                    })
                    .collect();

                // Wait for all threads and collect results
                for handle in handles.drain(..) {
                    results.push(handle.join().unwrap());
                }
            });

            // Merge results into tensor_storage
            for node_results in results {
                let node_outputs = node_results?;
                for (tensor_name, tensor_arc) in node_outputs {
                    if output_indices.contains_key(tensor_name.as_str()) {
                        let idx = *output_indices.get(tensor_name.as_str()).unwrap();
                        outputs[idx].1.data = tensor_arc.data.clone();
                    }
                    tensor_storage.insert(tensor_name, tensor_arc);
                }
            }
        }

        tracing::debug!("Parallel graph execution completed");
        Ok(())
    }
}

/// Execute a single node (used by parallel executor).
/// Returns output tensor name -> tensor pairs.
fn execute_node_single(
    node: &crate::ir::Node,
    compiled: &crate::backend::CompiledOperator,
    inputs: Vec<&crate::ir::Tensor>,
) -> Result<Vec<(String, Arc<Tensor>)>> {
    let backend: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend::new());

    let mut output_tensors: Vec<Tensor> = node
        .outputs
        .iter()
        .map(|o| {
            let mut t = Tensor::new(o.tensor_name.clone(), vec![1], o.data_type);
            t.layout = crate::common::StorageLayout::Default;
            t
        })
        .collect();

    let mut output_refs: Vec<&mut Tensor> = output_tensors.iter_mut().collect();

    backend.execute(compiled, &inputs, &mut output_refs)?;

    Ok(output_tensors
        .into_iter()
        .map(|t| (t.name.clone(), Arc::new(t)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::ir::{Graph, Node, NodeIO, OperatorType};
    use crate::common::DataType;
    use std::sync::Arc;

    #[test]
    fn test_graph_executor_creation() {
        let backend = Arc::new(CpuBackend::new());
        let executor = GraphExecutor::new(backend);
        assert!(executor.compiled_ops.is_empty());
    }

    #[test]
    fn test_graph_executor_single_relu() {
        let backend: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend::new());
        let mut executor = GraphExecutor::new(Arc::clone(&backend));

        // Create a simple graph: input -> ReLU -> output
        let mut graph = Graph::new("test_relu".to_string());
        let mut node = Node::new(0, "relu".to_string(), OperatorType::ReLU);
        node.inputs.push(NodeIO {
            tensor_name: "input".to_string(),
            data_type: DataType::F32,
        });
        node.outputs.push(NodeIO {
            tensor_name: "output".to_string(),
            data_type: DataType::F32,
        });
        graph.add_node(node);

        // Prepare the graph
        executor.prepare(&graph).unwrap();
        assert_eq!(executor.compiled_ops.len(), 1);

        // Create input tensor with values: [-1.0, 0.0, 1.0, 2.0, -5.0, 3.0]
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, -5.0, 3.0];
        let input = Tensor::from_data("input".to_string(), vec![6], DataType::F32, input_data);

        // Create output tensor that will be populated by execute
        let mut output = Tensor::new("output".to_string(), vec![6], DataType::F32);

        // Execute - need to pass &mut output directly in a tuple
        // We construct the outputs slice carefully to avoid moving output
        let mut outputs: &mut [(&str, Tensor)] = &mut [("output", Tensor::new("placeholder".into(), vec![6], DataType::F32))];

        // Copy our output into the slice
        outputs[0].1 = output;

        // Now call execute
        executor.execute(&graph, &[("input", input)], outputs).unwrap();

        // After execute, the output in the slice should have data
        let output_bytes = outputs[0].1.data_as_bytes();
        let output_data: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(output_data.len(), 6);
        assert_eq!(output_data[0], 0.0);  // max(-1, 0)
        assert_eq!(output_data[1], 0.0);  // max(0, 0)
        assert_eq!(output_data[2], 1.0);  // max(1, 0)
        assert_eq!(output_data[3], 2.0);  // max(2, 0)
        assert_eq!(output_data[4], 0.0);  // max(-5, 0)
        assert_eq!(output_data[5], 3.0);  // max(3, 0)
    }

    #[test]
    fn test_graph_executor_parallel_level() {
        // Graph: input1 → ReLU1, input2 → ReLU2 (both in same level, no dependency)
        let backend: Arc<dyn Backend + Send + Sync> = Arc::new(CpuBackend::new());
        let mut executor = GraphExecutor::new(Arc::clone(&backend));

        let mut graph = Graph::new("parallel_relu".to_string());

        // Node 0: ReLU1 (takes input1)
        let mut node0 = Node::new(0, "relu1".to_string(), OperatorType::ReLU);
        node0.inputs.push(NodeIO {
            tensor_name: "input1".to_string(),
            data_type: DataType::F32,
        });
        node0.outputs.push(NodeIO {
            tensor_name: "output1".to_string(),
            data_type: DataType::F32,
        });
        graph.add_node(node0);

        // Node 1: ReLU2 (takes input2, independent from node0)
        let mut node1 = Node::new(1, "relu2".to_string(), OperatorType::ReLU);
        node1.inputs.push(NodeIO {
            tensor_name: "input2".to_string(),
            data_type: DataType::F32,
        });
        node1.outputs.push(NodeIO {
            tensor_name: "output2".to_string(),
            data_type: DataType::F32,
        });
        graph.add_node(node1);

        executor.prepare(&graph).unwrap();

        let input1 = Tensor::from_data("input1".to_string(), vec![4], DataType::F32, vec![-1.0, 2.0, -3.0, 4.0]);
        let input2 = Tensor::from_data("input2".to_string(), vec![4], DataType::F32, vec![5.0, -6.0, 7.0, -8.0]);

        let mut outputs_arr = [
            ("output1", Tensor::new("output1".to_string(), vec![4], DataType::F32)),
            ("output2", Tensor::new("output2".to_string(), vec![4], DataType::F32)),
        ];

        executor.execute_parallel(
            &graph,
            &[("input1", input1), ("input2", input2)],
            &mut outputs_arr,
        ).unwrap();

        let o1_bytes = outputs_arr[0].1.data_as_bytes();
        let o1: Vec<f32> = o1_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let o2_bytes = outputs_arr[1].1.data_as_bytes();
        let o2: Vec<f32> = o2_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

        assert_eq!(o1, vec![0.0, 2.0, 0.0, 4.0]);
        assert_eq!(o2, vec![5.0, 0.0, 7.0, 0.0]);
    }

    #[test]
    fn test_graph_executor_empty_graph() {
        let backend = Arc::new(CpuBackend::new());
        let mut executor = GraphExecutor::new(backend);

        let graph = Graph::new("empty".to_string());
        executor.prepare(&graph).unwrap();
        assert!(executor.compiled_ops.is_empty());
    }
}
