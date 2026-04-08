//! Unit tests for executor module

use lightship_core::common::BackendType;
use lightship_core::executor::{
    ExecutionPlan, MemoryAllocation, MemoryPlan, ParallelGroup, ScheduledNode, Scheduler,
};
use lightship_core::ir::{FusionInfo, FusionType, Graph, Node, NodeIO, OperatorType};
use lightship_core::common::DataType;

#[test]
fn test_scheduled_node_creation() {
    let node = ScheduledNode::new(0, OperatorType::ReLU);

    assert_eq!(node.node_id, 0);
    assert_eq!(node.operator_type, OperatorType::ReLU);
    assert!(!node.is_fused());
    assert!(!node.parallelizable);
    assert_eq!(node.parallel_group, None);
}

#[test]
fn test_scheduled_node_fusion() {
    let mut node = ScheduledNode::new(0, OperatorType::Conv2d);
    node.fusion = Some(FusionInfo::conv_relu());

    assert!(node.is_fused());
    assert!(node.fusion_type().is_some());
}

#[test]
fn test_memory_allocation() {
    let alloc = MemoryAllocation {
        offset: 0,
        size: 1024,
        alignment: 64,
        reusable: true,
    };

    assert_eq!(alloc.size, 1024);
    assert!(alloc.reusable);
}

#[test]
fn test_memory_plan() {
    let mut plan = MemoryPlan::new();

    plan.input_sizes
        .insert("input".to_string(), 4096);
    plan.output_sizes
        .insert("output".to_string(), 4096);

    assert_eq!(plan.input_sizes.len(), 1);
    assert_eq!(plan.output_sizes.len(), 1);

    let total = plan.calculate_total();
    assert_eq!(total, 8192);
}

#[test]
fn test_memory_plan_add_allocation() {
    let mut plan = MemoryPlan::new();

    let alloc = MemoryAllocation {
        offset: 0,
        size: 1024,
        alignment: 64,
        reusable: true,
    };

    plan.add_allocation(0, alloc);

    assert_eq!(plan.node_allocations.len(), 1);
}

#[test]
fn test_parallel_group() {
    let mut group = ParallelGroup::new(0);
    group.add_node(1);
    group.add_node(2);
    group.add_node(3);

    assert_eq!(group.id, 0);
    assert_eq!(group.node_ids.len(), 3);
}

#[test]
fn test_execution_plan() {
    let mut plan = ExecutionPlan::new();

    let node = ScheduledNode::new(0, OperatorType::ReLU);
    plan.add_node(node);

    let mut group = ParallelGroup::new(0);
    group.add_node(0);
    plan.add_parallel_group(group);

    assert_eq!(plan.num_nodes(), 1);
    assert_eq!(plan.num_parallel_groups(), 1);
    assert!(!plan.supports_async());
}

#[test]
fn test_execution_plan_parallel_candidates() {
    let mut plan = ExecutionPlan::new();

    let mut node1 = ScheduledNode::new(0, OperatorType::ReLU);
    node1.parallelizable = true;
    node1.parallel_group = Some(0);
    node1.order_in_group = 0;

    let mut node2 = ScheduledNode::new(1, OperatorType::ReLU);
    node2.parallelizable = true;
    node2.parallel_group = Some(0);
    node2.order_in_group = 1;

    plan.add_node(node1);
    plan.add_node(node2);

    let candidates = plan.get_parallel_candidates(0);
    // Node at index 0 is not parallelizable in the sense that it starts the group
    // So it should return itself
    assert_eq!(candidates.len(), 1);
}

#[test]
fn test_scheduler_default() {
    let scheduler = Scheduler::new();
    let graph = Graph::new("test".to_string());
    let plan = scheduler.schedule(&graph);

    assert_eq!(plan.num_nodes(), 0);
}

#[test]
fn test_scheduler_single_node() {
    let scheduler = Scheduler::new();

    let mut graph = Graph::new("test".to_string());
    let mut node = Node::new(0, "relu".to_string(), OperatorType::ReLU);
    node.outputs.push(NodeIO {
        tensor_name: "output".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(node);

    let plan = scheduler.schedule(&graph);

    assert_eq!(plan.num_nodes(), 1);
}

#[test]
fn test_scheduler_conv_relu() {
    let scheduler = Scheduler::new();

    let mut graph = Graph::new("test".to_string());

    // Add conv node
    let mut conv = Node::new(0, "conv".to_string(), OperatorType::Conv2d);
    conv.outputs.push(NodeIO {
        tensor_name: "conv_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(conv);

    // Add relu node
    let mut relu = Node::new(1, "relu".to_string(), OperatorType::ReLU);
    relu.inputs.push(NodeIO {
        tensor_name: "conv_out".to_string(),
        data_type: DataType::F32,
    });
    relu.outputs.push(NodeIO {
        tensor_name: "relu_out".to_string(),
        data_type: DataType::F32,
    });
    graph.add_node(relu);

    let plan = scheduler.schedule(&graph);

    assert_eq!(plan.num_nodes(), 2);
}
