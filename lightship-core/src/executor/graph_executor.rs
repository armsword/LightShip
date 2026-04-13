//! Graph Executor for running inference
//!
//! This module provides end-to-end model execution using the backend.

use std::fmt::Debug;
use std::sync::Arc;
use crate::backend::Backend;
use crate::backend::CompiledOperator;
use crate::common::{LightShipError, Result};
use crate::ir::{Graph, OperatorDef, OperatorType, Tensor};
use std::collections::HashMap;

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
            let compiled = self.backend.as_ref().compile_operator(&def, &[], &[])?;
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
        let mut tensor_storage: HashMap<String, Tensor> = HashMap::new();

        // Register input tensors
        for (name, tensor) in inputs {
            tensor_storage.insert(name.to_string(), tensor.clone());
        }

        // Register output tensors (pre-allocated by caller)
        for (name, tensor) in outputs.iter_mut() {
            tensor_storage.insert(name.to_string(), tensor.clone());
        }

        // Get topological order
        let order = graph.topological_sort();

        // Execute nodes in order
        for &node_id in &order {
            // Find node by traversing (nodes are stored by insertion order, not by id)
            let node = graph.nodes.iter()
                .find(|n| n.id == node_id)
                .ok_or_else(|| LightShipError::InvalidParam(format!("Node {} not found", node_id)))?;

            // Get or create input tensors
            // Note: For now, we create placeholder tensors for missing inputs (e.g., weights from initializers)
            // This is a temporary solution until we properly parse initializers
            let mut input_tensors: Vec<Tensor> = Vec::new();
            for input in &node.inputs {
                if let Some(tensor) = tensor_storage.get(&input.tensor_name) {
                    input_tensors.push(tensor.clone());
                } else {
                    // Create a placeholder tensor for missing inputs (e.g., weights)
                    // This is temporary - proper initializers parsing is needed
                    tracing::warn!("Input tensor {} not found for node {}, using placeholder",
                        input.tensor_name, node.name);
                    let placeholder = Tensor::new(
                        input.tensor_name.clone(),
                        vec![1],
                        input.data_type,
                    );
                    tensor_storage.insert(input.tensor_name.clone(), placeholder.clone());
                    input_tensors.push(placeholder);
                }
            }
            let input_tensor_refs: Vec<&Tensor> = input_tensors.iter().collect();

            // Get output tensors from storage (pre-allocated by caller)
            let mut output_tensors: Vec<Tensor> = node.outputs.iter()
                .map(|output| {
                    tensor_storage.get(&output.tensor_name)
                        .cloned()
                        .unwrap_or_else(|| {
                            // Create placeholder tensor - should not reach here if caller provided outputs
                            Tensor::new(
                                output.tensor_name.clone(),
                                vec![1],
                                output.data_type,
                            )
                        })
                })
                .collect();

            // Get compiled operator
            let compiled = self.compiled_ops.get(&node.name)
                .ok_or_else(|| LightShipError::InvalidParam(
                    format!("Compiled operator {} not found", node.name)
                ))?;

            // Convert output refs
            let mut output_refs: Vec<&mut Tensor> = output_tensors.iter_mut()
                .map(|t| t as &mut Tensor)
                .collect();

            // Execute
            self.backend.as_ref().execute(
                compiled,
                &input_tensor_refs,
                &mut output_refs,
            )?;

            // Store outputs back to tensor_storage
            for (i, output) in node.outputs.iter().enumerate() {
                tensor_storage.insert(output.tensor_name.clone(), output_tensors[i].clone());
            }
        }

        // Copy output tensors back to caller
        for (name, output_tensor) in outputs.iter_mut() {
            if let Some(tensor) = tensor_storage.get(*name) {
                output_tensor.data = tensor.data.clone();
            }
        }

        tracing::debug!("Graph execution completed");
        Ok(())
    }
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
    fn test_graph_executor_empty_graph() {
        let backend = Arc::new(CpuBackend::new());
        let mut executor = GraphExecutor::new(backend);

        let graph = Graph::new("empty".to_string());
        executor.prepare(&graph).unwrap();
        assert!(executor.compiled_ops.is_empty());
    }
}
