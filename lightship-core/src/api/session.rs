//! Session API for LightShip

use crate::common::{BackendType, InferenceMode, LightShipError, Result};
use crate::executor::GraphExecutor;
use crate::ir::Graph;
use crate::backend::CpuBackend;
use std::fmt::Debug;
use std::sync::Arc;

/// Session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Preferred backend type
    pub preferred_backend: BackendType,
    /// Number of threads (0 = auto)
    pub num_threads: usize,
    /// Enable low memory mode
    pub low_memory_mode: bool,
    /// Maximum memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Inference mode
    pub inference_mode: InferenceMode,
    /// Enable operator fusion
    pub enable_fusion: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            preferred_backend: BackendType::CPU,
            num_threads: 0,
            low_memory_mode: false,
            memory_limit: None,
            inference_mode: InferenceMode::Synchronous,
            enable_fusion: true,
        }
    }
}

/// Session handle for inference
pub struct SessionHandle {
    /// Backend for execution (owned via Arc)
    backend: Arc<CpuBackend>,
    /// Graph executor
    executor: GraphExecutor,
    /// Prepared graph
    graph: Option<Graph>,
    /// Input tensor names
    input_names: Vec<String>,
    /// Output tensor names
    output_names: Vec<String>,
}

impl Debug for SessionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionHandle")
            .field("input_names", &self.input_names)
            .field("output_names", &self.output_names)
            .field("is_prepared", &self.graph.is_some())
            .finish()
    }
}

impl SessionHandle {
    /// Create a new session handle with CPU backend
    pub fn new() -> Result<Self> {
        let backend: Arc<CpuBackend> = Arc::new(CpuBackend::new());
        let executor = GraphExecutor::new(backend.clone());
        Ok(Self {
            backend,
            executor,
            graph: None,
            input_names: Vec::new(),
            output_names: Vec::new(),
        })
    }

    /// Prepare a graph for execution
    pub fn prepare_graph(&mut self, graph: Graph) -> Result<()> {
        // Extract input/output tensor names
        self.input_names = graph.inputs.iter()
            .map(|g| g.name.clone())
            .collect();
        self.output_names = graph.outputs.iter()
            .map(|g| g.name.clone())
            .collect();

        // Compile all operators
        self.executor.prepare(&graph)?;
        self.graph = Some(graph);
        Ok(())
    }

    /// Execute forward pass with given inputs
    pub fn forward(
        &self,
        inputs: &[(&str, crate::ir::Tensor)],
        outputs: &mut [(&str, crate::ir::Tensor)],
    ) -> Result<()> {
        let graph = self.graph.as_ref()
            .ok_or_else(|| LightShipError::InvalidParam("Graph not prepared".into()))?;
        self.executor.execute(graph, inputs, outputs)
    }

    /// Get input tensor names
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output tensor names
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Check if session is prepared
    pub fn is_prepared(&self) -> bool {
        self.graph.is_some()
    }
}

impl Default for SessionHandle {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Graph, Node, NodeIO, OperatorType};
    use crate::common::DataType;

    #[test]
    fn test_session_creation() {
        let session = SessionHandle::new().unwrap();
        assert!(!session.is_prepared());
    }

    #[test]
    fn test_session_with_graph() {
        let mut session = SessionHandle::new().unwrap();

        // Create a simple graph
        let mut graph = Graph::new("test".to_string());
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

        // Set up graph inputs and outputs
        graph.inputs.push(crate::ir::GraphIO {
            name: "input".to_string(),
            io: NodeIO {
                tensor_name: "input".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: true,
            is_model_output: false,
        });
        graph.outputs.push(crate::ir::GraphIO {
            name: "output".to_string(),
            io: NodeIO {
                tensor_name: "output".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: false,
            is_model_output: true,
        });

        session.prepare_graph(graph).unwrap();
        assert!(session.is_prepared());
        assert_eq!(session.input_names(), &["input"]);
        assert_eq!(session.output_names(), &["output"]);
    }

    #[test]
    fn test_session_forward() {
        let mut session = SessionHandle::new().unwrap();

        // Create a simple graph: input -> ReLU -> output
        let mut graph = Graph::new("test".to_string());
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

        // Set up graph inputs and outputs
        graph.inputs.push(crate::ir::GraphIO {
            name: "input".to_string(),
            io: NodeIO {
                tensor_name: "input".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: true,
            is_model_output: false,
        });
        graph.outputs.push(crate::ir::GraphIO {
            name: "output".to_string(),
            io: NodeIO {
                tensor_name: "output".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: false,
            is_model_output: true,
        });

        session.prepare_graph(graph).unwrap();

        // Create input tensor
        let input_data: Vec<f32> = vec![-1.0, 0.0, 1.0, 2.0, -5.0, 3.0];
        let input = crate::ir::Tensor::from_data("input".to_string(), vec![6], DataType::F32, input_data);

        // Create output tensor - need to put it in the outputs slice directly
        let mut outputs_arr: &mut [(&str, crate::ir::Tensor)] = &mut [("output", crate::ir::Tensor::new("output".to_string(), vec![6], DataType::F32))];
        session.forward(&[("input", input)], outputs_arr).unwrap();

        // Verify - outputs_arr[0].1 is the output tensor after forward
        let output_bytes = outputs_arr[0].1.data_as_bytes();
        let output_data: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(output_data, vec![0.0, 0.0, 1.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_session_multi_node_graph() {
        // Test a graph with two ReLU nodes: input -> relu1 -> relu2 -> output
        // After first ReLU: [-1, 0, 2, -3] -> [0, 0, 2, 0]
        // After second ReLU: [0, 0, 2, 0] -> [0, 0, 2, 0] (no change, all >= 0)
        let mut session = SessionHandle::new().unwrap();

        let mut graph = Graph::new("test_multi".to_string());

        // First ReLU node
        let mut relu1 = Node::new(0, "relu1".to_string(), OperatorType::ReLU);
        relu1.inputs.push(NodeIO {
            tensor_name: "input".to_string(),
            data_type: DataType::F32,
        });
        relu1.outputs.push(NodeIO {
            tensor_name: "intermediate".to_string(),
            data_type: DataType::F32,
        });
        graph.add_node(relu1);

        // Second ReLU node
        let mut relu2 = Node::new(1, "relu2".to_string(), OperatorType::ReLU);
        relu2.inputs.push(NodeIO {
            tensor_name: "intermediate".to_string(),
            data_type: DataType::F32,
        });
        relu2.outputs.push(NodeIO {
            tensor_name: "output".to_string(),
            data_type: DataType::F32,
        });
        graph.add_node(relu2);

        // Set up graph inputs and outputs
        graph.inputs.push(crate::ir::GraphIO {
            name: "input".to_string(),
            io: NodeIO {
                tensor_name: "input".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: true,
            is_model_output: false,
        });
        graph.outputs.push(crate::ir::GraphIO {
            name: "output".to_string(),
            io: NodeIO {
                tensor_name: "output".to_string(),
                data_type: DataType::F32,
            },
            is_model_input: false,
            is_model_output: true,
        });

        session.prepare_graph(graph).unwrap();
        assert!(session.is_prepared());

        // Input: [-1.0, 0.0, 2.0, -3.0]
        // After first ReLU: [0.0, 0.0, 2.0, 0.0]
        // After second ReLU: [0.0, 0.0, 2.0, 0.0]
        let input_data: Vec<f32> = vec![-1.0, 0.0, 2.0, -3.0];
        let input = crate::ir::Tensor::from_data("input".to_string(), vec![4], DataType::F32, input_data);

        let mut outputs_arr: &mut [(&str, crate::ir::Tensor)] = &mut [
            ("output", crate::ir::Tensor::new("output".to_string(), vec![4], DataType::F32))
        ];
        session.forward(&[("input", input)], outputs_arr).unwrap();

        let output_bytes = outputs_arr[0].1.data_as_bytes();
        let output_data: Vec<f32> = output_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Both ReLUs should produce [0, 0, 2, 0]
        assert_eq!(output_data, vec![0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_session_not_prepared_error() {
        let session = SessionHandle::new().unwrap();

        let input = crate::ir::Tensor::new("input".to_string(), vec![4], DataType::F32);
        let mut outputs_arr: &mut [(&str, crate::ir::Tensor)] = &mut [
            ("output", crate::ir::Tensor::new("output".to_string(), vec![4], DataType::F32))
        ];

        let result = session.forward(&[("input", input)], outputs_arr);
        assert!(result.is_err());
    }
}
