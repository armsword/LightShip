//! Unit tests for Graph

use lightship_core::common::DataType;
use lightship_core::ir::{
    Attribute, AttributeMap, AttributeValue, Graph, Node, NodeIO, NodeId, OperatorDef,
    OperatorType,
};

#[test]
fn test_graph_creation() {
    let graph = Graph::new("test_graph".into());
    assert_eq!(graph.name, "test_graph");
    assert!(graph.nodes.is_empty());
    assert!(graph.inputs.is_empty());
    assert!(graph.outputs.is_empty());
}

#[test]
fn test_graph_add_node() {
    let mut graph = Graph::new("test".into());

    let mut node = Node::new(0, "conv1".into(), OperatorType::Conv2d);
    node.inputs.push(NodeIO {
        tensor_name: "input".into(),
        data_type: DataType::F32,
    });
    node.outputs.push(NodeIO {
        tensor_name: "output".into(),
        data_type: DataType::F32,
    });

    let id = graph.add_node(node);
    assert_eq!(id, 0);
    assert_eq!(graph.nodes.len(), 1);

    let retrieved = graph.node("conv1").unwrap();
    assert_eq!(retrieved.name, "conv1");
    assert_eq!(retrieved.operator_type, OperatorType::Conv2d);
}

#[test]
fn test_graph_topological_sort_empty() {
    let graph = Graph::new("empty".into());
    let sorted = graph.topological_sort();
    assert!(sorted.is_empty());
}

#[test]
fn test_graph_topological_sort_linear() {
    // Linear graph: input -> conv -> relu -> output
    let mut graph = Graph::new("linear".into());

    let mut conv = Node::new(0, "conv".into(), OperatorType::Conv2d);
    conv.inputs.push(NodeIO { tensor_name: "input".into(), data_type: DataType::F32 });
    conv.outputs.push(NodeIO { tensor_name: "conv_out".into(), data_type: DataType::F32 });
    graph.add_node(conv);

    let mut relu = Node::new(1, "relu".into(), OperatorType::ReLU);
    relu.inputs.push(NodeIO { tensor_name: "conv_out".into(), data_type: DataType::F32 });
    relu.outputs.push(NodeIO { tensor_name: "relu_out".into(), data_type: DataType::F32 });
    graph.add_node(relu);

    let sorted = graph.topological_sort();
    // conv should come before relu
    let conv_pos = sorted.iter().position(|&id| id == 0).unwrap();
    let relu_pos = sorted.iter().position(|&id| id == 1).unwrap();
    assert!(conv_pos < relu_pos);
}

#[test]
fn test_attribute_map() {
    let mut attrs = AttributeMap::new();

    attrs.insert(Attribute::new_int("kernel_shape", 3));
    attrs.insert(Attribute::new_float("pad", 1.0));
    attrs.insert(Attribute::new_int_list("strides", vec![1, 1]));

    assert_eq!(attrs.len(), 3);
    assert!(attrs.contains("kernel_shape"));
    assert_eq!(attrs.get_int("kernel_shape"), Some(3));
    assert_eq!(attrs.get_float("pad"), Some(1.0));
    assert_eq!(attrs.get_int_list("strides"), Some(&[1, 1][..]));
}

#[test]
fn test_operator_def() {
    let mut op = OperatorDef::new("conv1".into(), OperatorType::Conv2d);
    op.inputs.push(NodeIO { tensor_name: "input".into(), data_type: DataType::F32 });
    op.outputs.push(NodeIO { tensor_name: "output".into(), data_type: DataType::F32 });
    op.attributes.insert(Attribute::new_int("kernel_shape", 3));

    assert_eq!(op.name, "conv1");
    assert_eq!(op.operator_type, OperatorType::Conv2d);
    assert_eq!(op.inputs.len(), 1);
    assert_eq!(op.outputs.len(), 1);
    assert_eq!(op.attributes.get_int("kernel_shape"), Some(3));
}

#[test]
fn test_operator_type_categories() {
    assert_eq!(OperatorType::Conv2d.category(), "convolution");
    assert_eq!(OperatorType::ReLU.category(), "activation");
    assert_eq!(OperatorType::MaxPool2d.category(), "pooling");
    assert_eq!(OperatorType::MatMul.category(), "broadcast");
    assert_eq!(OperatorType::SelfAttention.category(), "attention");
    assert_eq!(OperatorType::LayerNorm.category(), "normalization");
}

#[test]
fn test_operator_type_display() {
    assert_eq!(format!("{}", OperatorType::Conv2d), "Conv2d");
    assert_eq!(format!("{}", OperatorType::ReLU), "ReLU");
    assert_eq!(format!("{}", OperatorType::SelfAttention), "SelfAttention");
}
