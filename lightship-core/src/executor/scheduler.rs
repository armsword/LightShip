//! Execution scheduler
//!
//! This module provides scheduling functionality to create ExecutionPlans from graphs.

use crate::common::BackendType;
use crate::ir::{Graph, Node, NodeId, OperatorType};
use std::collections::{HashMap, HashSet};

use super::plan::{ExecutionPlan, MemoryAllocation, MemoryPlan, ParallelGroup, ScheduledNode};

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Maximum parallel group size
    pub max_parallel_size: usize,
    /// Enable memory optimization
    pub optimize_memory: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            max_parallel_size: 4,
            optimize_memory: true,
        }
    }
}

/// Scheduler that creates execution plans from graphs
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
}

impl Scheduler {
    /// Create a new scheduler with default configuration
    pub fn new() -> Self {
        Self {
            config: SchedulerConfig::default(),
        }
    }

    /// Create a scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self { config }
    }

    /// Schedule a graph into an execution plan
    pub fn schedule(&self, graph: &Graph) -> ExecutionPlan {
        let mut plan = ExecutionPlan::new();

        // Topological sort to get execution order
        let order = graph.topological_sort();

        // Build node lookup map
        let node_map: HashMap<NodeId, &Node> =
            graph.nodes.iter().map(|n| (n.id, n)).collect();

        // Track parallelizable nodes
        let mut parallelizable_ops: HashSet<NodeId> = HashSet::new();
        if self.config.enable_parallel {
            parallelizable_ops = self.find_parallelizable_nodes(graph, &order);
        }

        // Schedule nodes
        for (order_idx, &node_id) in order.iter().enumerate() {
            if let Some(node) = node_map.get(&node_id) {
                let mut scheduled = self.schedule_node(node, order_idx, &parallelizable_ops);
                plan.add_node(scheduled);
            }
        }

        // Create parallel groups
        if self.config.enable_parallel {
            self.create_parallel_groups(&mut plan);
        }

        // Create memory plan
        if self.config.optimize_memory {
            plan.memory_plan = self.create_memory_plan(graph, &plan);
        }

        plan
    }

    /// Schedule a single node
    fn schedule_node(
        &self,
        node: &Node,
        order: usize,
        parallelizable: &HashSet<NodeId>,
    ) -> ScheduledNode {
        let mut scheduled = ScheduledNode::new(node.id, node.operator_type.clone());

        // Copy inputs and outputs
        scheduled.input_ids = node.inputs.iter().map(|i| i.tensor_name.clone()).collect();
        scheduled.output_ids = node.outputs.iter().map(|o| o.tensor_name.clone()).collect();

        // Determine backend
        scheduled.backend = self.select_backend(&node.operator_type);

        // Check if parallelizable
        if parallelizable.contains(&node.id) {
            scheduled.parallelizable = true;
            scheduled.parallel_group = Some(node.id as usize % self.config.max_parallel_size);
        }

        // Estimate memory
        scheduled.memory_estimate = self.estimate_memory(node);

        scheduled
    }

    /// Find nodes that can run in parallel
    fn find_parallelizable_nodes(&self, graph: &Graph, order: &[NodeId]) -> HashSet<NodeId> {
        let mut parallelizable = HashSet::new();

        for &node_id in order {
            if let Some(node) = graph.node_by_id(node_id) {
                if self.is_parallelizable(node) {
                    parallelizable.insert(node_id);
                }
            }
        }

        parallelizable
    }

    /// Check if a node type is parallelizable
    fn is_parallelizable(&self, node: &Node) -> bool {
        matches!(
            node.operator_type,
            OperatorType::Add
                | OperatorType::Mul
                | OperatorType::ReLU
                | OperatorType::Sigmoid
                | OperatorType::Tanh
                | OperatorType::Reshape
                | OperatorType::Transpose
        )
    }

    /// Select appropriate backend for an operator
    fn select_backend(&self, op: &OperatorType) -> BackendType {
        match op {
            OperatorType::Conv2d | OperatorType::ConvTranspose2d => BackendType::CPU,
            OperatorType::SelfAttention | OperatorType::MultiHeadAttention => BackendType::CPU,
            _ => BackendType::CPU,
        }
    }

    /// Estimate memory requirement for a node
    fn estimate_memory(&self, node: &Node) -> usize {
        // Simple estimation based on operator type
        let base_size = match node.operator_type {
            OperatorType::Conv2d => 64 * 1024,   // 64KB for conv buffers
            OperatorType::FullyConnected => 16 * 1024, // 16KB for FC
            OperatorType::SelfAttention => 128 * 1024, // 128KB for attention
            _ => 4 * 1024, // 4KB default
        };

        // Add input/output sizes
        let io_size: usize = node
            .inputs
            .iter()
            .chain(node.outputs.iter())
            .map(|_| 1024) // Assume 1KB per tensor for simplicity
            .sum();

        base_size + io_size
    }

    /// Create parallel execution groups
    fn create_parallel_groups(&self, plan: &mut ExecutionPlan) {
        let mut group_map: HashMap<usize, Vec<NodeId>> = HashMap::new();

        for node in &plan.nodes {
            if node.parallelizable {
                let group_id = node.parallel_group.unwrap_or(0);
                group_map.entry(group_id).or_default().push(node.node_id);
            }
        }

        for (group_id, node_ids) in group_map {
            let mut group = ParallelGroup::new(group_id);
            for node_id in node_ids {
                group.add_node(node_id);
            }
            plan.add_parallel_group(group);
        }
    }

    /// Create memory plan for the execution
    fn create_memory_plan(&self, graph: &Graph, plan: &ExecutionPlan) -> MemoryPlan {
        let mut memory_plan = MemoryPlan::new();

        // Collect input/output sizes from graph
        for input in &graph.inputs {
            memory_plan
                .input_sizes
                .insert(input.io.tensor_name.clone(), 4096); // Default 4KB
        }

        for output in &graph.outputs {
            memory_plan
                .output_sizes
                .insert(output.io.tensor_name.clone(), 4096);
        }

        // Add allocations for scheduled nodes
        for scheduled in &plan.nodes {
            let allocation = MemoryAllocation {
                offset: 0,
                size: scheduled.memory_estimate,
                alignment: 64,
                reusable: scheduled.parallelizable,
            };
            memory_plan.add_allocation(scheduled.node_id, allocation);
        }

        // Calculate totals
        memory_plan.total_memory = memory_plan.calculate_total();
        memory_plan.peak_memory = memory_plan.total_memory;

        memory_plan
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
