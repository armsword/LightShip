//! Execution planning structures
//!
//! This module contains the core execution planning components:
//! - ScheduledNode: A node prepared for execution with runtime info
//! - MemoryPlan: Memory allocation and reuse plan
//! - ExecutionPlan: Complete execution plan for a model

use crate::common::BackendType;
use crate::ir::{FusionInfo, NodeId, OperatorType, Tensor};
use std::collections::HashMap;

/// Represents a node in the execution schedule
#[derive(Debug, Clone)]
pub struct ScheduledNode {
    /// Node ID in the original graph
    pub node_id: NodeId,
    /// Operator type
    pub operator_type: OperatorType,
    /// Input tensor IDs
    pub input_ids: Vec<String>,
    /// Output tensor IDs
    pub output_ids: Vec<String>,
    /// Backend to use for execution
    pub backend: BackendType,
    /// Fusion info if this node is fused
    pub fusion: Option<FusionInfo>,
    /// Whether this node can run in parallel
    pub parallelizable: bool,
    /// Execution order within its parallel group
    pub order_in_group: usize,
    /// Parallel group ID (None if not parallelizable)
    pub parallel_group: Option<usize>,
    /// Estimated memory requirement in bytes
    pub memory_estimate: usize,
}

impl ScheduledNode {
    /// Create a new scheduled node
    pub fn new(node_id: NodeId, operator_type: OperatorType) -> Self {
        Self {
            node_id,
            operator_type,
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            backend: BackendType::CPU,
            fusion: None,
            parallelizable: false,
            order_in_group: 0,
            parallel_group: None,
            memory_estimate: 0,
        }
    }

    /// Check if this node is a fusion node
    pub fn is_fused(&self) -> bool {
        self.fusion.is_some()
    }

    /// Get the fusion type if fused
    pub fn fusion_type(&self) -> Option<&FusionInfo> {
        self.fusion.as_ref()
    }
}

/// Memory plan for execution
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Total estimated memory usage
    pub total_memory: usize,
    /// Peak memory usage during execution
    pub peak_memory: usize,
    /// Memory allocations per node
    pub node_allocations: HashMap<NodeId, MemoryAllocation>,
    /// Memory reuse pairs (nodes that can share memory)
    pub reuse_pairs: Vec<(NodeId, NodeId)>,
    /// Input tensor sizes
    pub input_sizes: HashMap<String, usize>,
    /// Output tensor sizes
    pub output_sizes: HashMap<String, usize>,
    /// Temporary tensor sizes
    pub temp_sizes: HashMap<String, usize>,
}

impl MemoryPlan {
    /// Create a new empty memory plan
    pub fn new() -> Self {
        Self {
            total_memory: 0,
            peak_memory: 0,
            node_allocations: HashMap::new(),
            reuse_pairs: Vec::new(),
            input_sizes: HashMap::new(),
            output_sizes: HashMap::new(),
            temp_sizes: HashMap::new(),
        }
    }

    /// Add a node allocation
    pub fn add_allocation(&mut self, node_id: NodeId, allocation: MemoryAllocation) {
        self.node_allocations.insert(node_id, allocation);
    }

    /// Calculate total memory requirement
    pub fn calculate_total(&self) -> usize {
        let inputs: usize = self.input_sizes.values().sum();
        let outputs: usize = self.output_sizes.values().sum();
        let temps: usize = self.temp_sizes.values().sum();
        inputs + outputs + temps
    }
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory allocation for a node
#[derive(Debug, Clone)]
pub struct MemoryAllocation {
    /// Start offset in the memory buffer
    pub offset: usize,
    /// Size in bytes
    pub size: usize,
    /// Alignment requirement
    pub alignment: usize,
    /// Whether this allocation can be reused
    pub reusable: bool,
}

/// Parallel execution group
#[derive(Debug, Clone)]
pub struct ParallelGroup {
    /// Group ID
    pub id: usize,
    /// Node IDs in this group
    pub node_ids: Vec<NodeId>,
    /// Estimated execution time in cycles
    pub estimated_cycles: u64,
}

impl ParallelGroup {
    /// Create a new parallel group
    pub fn new(id: usize) -> Self {
        Self {
            id,
            node_ids: Vec::new(),
            estimated_cycles: 0,
        }
    }

    /// Add a node to this group
    pub fn add_node(&mut self, node_id: NodeId) {
        self.node_ids.push(node_id);
    }
}

/// Complete execution plan for a model
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Scheduled nodes in execution order
    pub nodes: Vec<ScheduledNode>,
    /// Parallel execution groups
    pub parallel_groups: Vec<ParallelGroup>,
    /// Memory plan
    pub memory_plan: MemoryPlan,
    /// Total estimated execution time in cycles
    pub total_cycles: u64,
    /// Whether async execution is supported
    pub supports_async: bool,
}

impl ExecutionPlan {
    /// Create a new empty execution plan
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            parallel_groups: Vec::new(),
            memory_plan: MemoryPlan::new(),
            total_cycles: 0,
            supports_async: false,
        }
    }

    /// Add a scheduled node
    pub fn add_node(&mut self, node: ScheduledNode) {
        self.nodes.push(node);
    }

    /// Add a parallel group
    pub fn add_parallel_group(&mut self, group: ParallelGroup) {
        self.parallel_groups.push(group);
    }

    /// Get the number of nodes in the plan
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of parallel groups
    pub fn num_parallel_groups(&self) -> usize {
        self.parallel_groups.len()
    }

    /// Check if the plan supports async execution
    pub fn supports_async(&self) -> bool {
        self.supports_async
    }

    /// Get nodes that can run in parallel at a given index
    pub fn get_parallel_candidates(&self, index: usize) -> Vec<&ScheduledNode> {
        if index >= self.nodes.len() {
            return Vec::new();
        }

        let current = &self.nodes[index];
        if !current.parallelizable {
            return vec![current];
        }

        let group_id = current.parallel_group;
        let order = current.order_in_group;
        self.nodes
            .iter()
            .filter(|n| {
                n.parallelizable
                    && n.parallel_group == group_id
                    && n.order_in_group == order
            })
            .collect()
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}
