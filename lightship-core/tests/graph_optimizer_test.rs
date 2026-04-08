//! Unit tests for graph optimization

use lightship_core::ir::{
    FusionInfo, FusionType, Graph, Node, NodeIO,
    OperatorType,
};
use lightship_core::ir::optimizer::GraphOptimizer;
use lightship_core::common::DataType;

fn create_test_graph() -> Graph {
    let mut graph = Graph::new("test".to_string());

    // Add input node
    let input = Node::new(0, "input".to_string(), OperatorType::Reshape);
    graph.add_node(input);

    // Add conv node
    let mut conv = Node::new(1, "conv".to_string(), OperatorType::Conv2d);
    conv.inputs.push(NodeIO {
        tensor_name: "input_tensor".to_string(),
        data_type: DataType::F32,
    });
    conv.outputs.push(NodeIO {
        tensor_name: "conv_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(conv);

    // Add relu node
    let mut relu = Node::new(2, "relu".to_string(), OperatorType::ReLU);
    relu.inputs.push(NodeIO {
        tensor_name: "conv_out".to_string(),
        data_type: DataType::F32,
    });
    relu.outputs.push(NodeIO {
        tensor_name: "relu_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(relu);

    graph
}

#[test]
fn test_graph_topological_sort() {
    let graph = create_test_graph();
    let order = graph.topological_sort();

    // Should have 3 nodes
    assert_eq!(order.len(), 3);

    // Reshape (id=0) should come before Conv (id=1)
    // Conv (id=1) should come before ReLU (id=2)
    let reshape_pos = order.iter().position(|&id| id == 0).unwrap();
    let conv_pos = order.iter().position(|&id| id == 1).unwrap();
    let relu_pos = order.iter().position(|&id| id == 2).unwrap();

    assert!(reshape_pos < conv_pos);
    assert!(conv_pos < relu_pos);
}

#[test]
fn test_graph_node_lookup() {
    let mut graph = create_test_graph();

    assert!(graph.node("conv").is_some());
    assert!(graph.node("nonexistent").is_none());
    assert!(graph.node_by_id(1).is_some());
    assert!(graph.node_by_id(99).is_none());
}

#[test]
fn test_constant_folding_identity() {
    let mut graph = Graph::new("test".to_string());

    // Create nodes with identity constants that can be folded
    let mut add = Node::new(0, "add".to_string(), OperatorType::Add);
    add.inputs.push(NodeIO {
        tensor_name: "x".to_string(),
        data_type: DataType::F32,
    });
    add.outputs.push(NodeIO {
        tensor_name: "add_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(add);

    // Apply constant folding optimization
    use lightship_core::ir::optimizer::ConstantFolding;
    let optimizer = ConstantFolding::new();
    optimizer.optimize(&mut graph);

    // After folding identity operations, single-input Add nodes are removed
    // So graph should have 0 nodes
    assert_eq!(graph.nodes.len(), 0);
}

#[test]
fn test_dead_code_elimination() {
    let mut graph = Graph::new("test".to_string());

    // Add unused nodes
    let mut unused = Node::new(0, "unused".to_string(), OperatorType::Print);
    unused.outputs.push(NodeIO {
        tensor_name: "unused_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(unused);

    // Add used node
    let mut used = Node::new(1, "used".to_string(), OperatorType::ReLU);
    used.inputs.push(NodeIO {
        tensor_name: "input".to_string(),
        data_type: DataType::F32,
    });
    used.outputs.push(NodeIO {
        tensor_name: "output".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(used);

    // Apply DCE
    use lightship_core::ir::optimizer::DeadCodeElimination;
    let mut optimizer = DeadCodeElimination::new();
    optimizer.set_inputs(vec!["input".to_string()]);
    optimizer.set_outputs(vec!["output".to_string()]);
    optimizer.optimize(&mut graph);

    // Only the used node should remain
    assert!(graph.node("used").is_some());
}

#[test]
fn test_fusion_type_display() {
    assert_eq!(format!("{}", FusionType::ConvReLU), "Conv+ReLU");
    assert_eq!(format!("{}", FusionType::ConvBatchNorm), "Conv+BatchNorm");
    assert_eq!(format!("{}", FusionType::BatchNormReLU), "BatchNorm+ReLU");
}

#[test]
fn test_fusion_info_creation() {
    let fusion = FusionInfo::conv_relu();
    assert_eq!(fusion.fusion_type, FusionType::ConvReLU);
    assert!(!fusion.eliminate_batch_norm);

    let fusion_bn = FusionInfo::conv_batch_norm();
    assert_eq!(fusion_bn.fusion_type, FusionType::ConvBatchNorm);
    assert!(fusion_bn.eliminate_batch_norm);
}

#[test]
fn test_node_replacement() {
    let mut graph = Graph::new("test".to_string());

    // Create two nodes where one can replace another
    let mut original = Node::new(0, "original".to_string(), OperatorType::ReLU);
    original.outputs.push(NodeIO {
        tensor_name: "out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(original);

    // Create replacement node
    let mut replacement = Node::new(1, "replacement".to_string(), OperatorType::ReLU6);
    replacement.outputs.push(NodeIO {
        tensor_name: "out".to_string(),
        data_type: DataType::F32,
    });

    // Apply node replacement
    use lightship_core::ir::optimizer::NodeReplacement;
    let mut optimizer = NodeReplacement::new();
    optimizer.replace_node(&mut graph, "original", replacement);

    // Original node should be replaced
    assert!(graph.node("original").is_none());
    assert!(graph.node("replacement").is_some());
}
