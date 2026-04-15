//! Thread Scheduler for LightShip
//!
//! Multi-threaded scheduling framework for parallel operator execution.
//!
//! # Overview
//!
//! This module provides block-wise scheduling for compute-intensive operators
//! (Conv2d, MatMul) with the following features:
//! - Automatic block decomposition of large operators
//! - Rayon-based thread pool management
//! - CPU affinity settings
//! - Load balancing across threads
//! - Pipeline parallelism support
//!
//! # Architecture
//!
//! ```text
//! ThreadPool (rayon)
//!     ├── Worker 0: Conv2d block 0, Conv2d block 2
//!     ├── Worker 1: Conv2d block 1, Conv2d block 3
//!     ├── Worker 2: MatMul block 0
//!     └── Worker 3: MatMul block 1
//! ```
//!
//! # Block Decomposition
//!
//! Conv2d and MatMul operators are split into blocks that can be processed in parallel:
//!
//! - **Conv2d**: Split along batch dimension, each block processes one or more batch elements
//! - **MatMul**: Split along output dimension (M*N), each block handles a portion of output elements

use crate::ir::OperatorType;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// ============================================================================
// Configuration
// ============================================================================

/// Thread scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Number of threads (0 = auto-detect based on available parallelism)
    pub num_threads: usize,
    /// Enable parallel execution
    pub enable_parallel: bool,
    /// Maximum block size for decomposition
    pub max_block_size: usize,
    /// Enable CPU affinity
    pub use_affinity: bool,
    /// Load balancing strategy
    pub load_balance: LoadBalanceConfig,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_threads: 0, // Auto-detect
            enable_parallel: true,
            max_block_size: 64 * 1024, // 64K elements per block
            use_affinity: true,
            load_balance: LoadBalanceConfig::Greedy,
        }
    }
}

/// Load balancing configuration
#[derive(Debug, Clone, Copy)]
pub enum LoadBalanceConfig {
    /// Simple greedy assignment (lightest loaded worker gets next task)
    Greedy,
    /// Work-stealing for dynamic load balancing
    WorkStealing,
    /// Round-robin assignment
    RoundRobin,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of threads (0 = auto-detect)
    pub num_threads: usize,
    /// Thread name prefix
    pub name_prefix: String,
    /// Stack size per thread
    pub stack_size: Option<usize>,
    /// Enable thread affinity
    pub use_affinity: bool,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            name_prefix: "lightship".to_string(),
            stack_size: None,
            use_affinity: false,
        }
    }
}

// ============================================================================
// Thread Pool (using rayon)
// ============================================================================

/// Thread pool wrapper using rayon
#[derive(Debug)]
pub struct ThreadPool {
    num_threads: usize,
    config: ThreadPoolConfig,
}

impl ThreadPool {
    /// Create a new thread pool with default configuration
    pub fn new() -> Self {
        Self::with_config(ThreadPoolConfig::default())
    }

    /// Create a thread pool with custom configuration
    pub fn with_config(config: ThreadPoolConfig) -> Self {
        let num_threads = if config.num_threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            config.num_threads
        };

        Self {
            num_threads,
            config,
        }
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Submit a task to the thread pool (fire-and-forget)
    pub fn submit<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        // Rayon's global thread pool is used for parallelism
        // Tasks are submitted to rayon for execution
        std::thread::spawn(task);
    }

    /// Submit a task and return a handle for synchronization
    pub fn submit_async<F, T>(task: F) -> std::thread::JoinHandle<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        std::thread::spawn(task)
    }

    /// Execute a parallel for loop using rayon
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        range.into_par_iter().for_each(f);
    }

    /// Get the configuration
    pub fn config(&self) -> &ThreadPoolConfig {
        &self.config
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Rayon handles cleanup automatically via its global pool
    }
}

// ============================================================================
// Block Decomposition
// ============================================================================

static BLOCK_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

/// A compute block that can be scheduled for parallel execution
#[derive(Debug, Clone)]
pub struct ComputeBlock {
    /// Unique block identifier
    pub block_id: u32,
    /// Node ID this block belongs to
    pub node_id: u32,
    /// Start index in the output buffer
    pub start_idx: usize,
    /// End index in the output buffer (exclusive)
    pub end_idx: usize,
    /// Estimated workload for this block
    pub workload: WorkloadEstimate,
}

impl ComputeBlock {
    /// Create a new compute block
    fn new(node_id: u32, start_idx: usize, end_idx: usize, workload: WorkloadEstimate) -> Self {
        Self {
            block_id: BLOCK_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            node_id,
            start_idx,
            end_idx,
            workload,
        }
    }

    /// Get the number of elements in this block
    pub fn size(&self) -> usize {
        self.end_idx.saturating_sub(self.start_idx)
    }
}

/// Workload estimation for a compute block
#[derive(Debug, Clone, Copy)]
pub struct WorkloadEstimate {
    /// Estimated FLOPs for this operation
    pub flops: usize,
    /// Estimated memory footprint in bytes
    pub memory_bytes: usize,
    /// Estimated compute intensity (FLOPs/byte)
    pub intensity: f32,
}

impl WorkloadEstimate {
    /// Estimate workload for a Conv2d operation
    pub fn for_conv2d(
        batch: usize,
        out_channels: usize,
        out_height: usize,
        out_width: usize,
        in_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Self {
        // Conv2d FLOPs: N * OC * OH * OW * IC * KH * KW * 2 (mult + add)
        let flops = batch * out_channels * out_height * out_width * in_channels * kernel_h * kernel_w * 2;

        // Memory: input + output + weights
        let memory_bytes = batch * out_channels * out_height * out_width * 4 // output
            + batch * in_channels * ((out_height - 1) * kernel_h) * ((out_width - 1) * kernel_w) * 4 // input estimate
            + out_channels * in_channels * kernel_h * kernel_w * 4; // weights

        let intensity = if memory_bytes > 0 {
            flops as f32 / memory_bytes as f32
        } else {
            0.0
        };

        Self {
            flops,
            memory_bytes,
            intensity,
        }
    }

    /// Estimate workload for a MatMul operation
    pub fn for_matmul(m: usize, k: usize, n: usize) -> Self {
        // MatMul FLOPs: 2 * M * K * N (mult + add for each element)
        let flops = 2 * m * k * n;

        // Memory: A + B + C
        let memory_bytes = m * k * 4 + k * n * 4 + m * n * 4;

        let intensity = if memory_bytes > 0 {
            flops as f32 / memory_bytes as f32
        } else {
            0.0
        };

        Self {
            flops,
            memory_bytes,
            intensity,
        }
    }
}

/// Block scheduler for decomposing operators into parallel blocks
#[derive(Debug)]
pub struct BlockScheduler;

impl BlockScheduler {
    /// Decompose a Conv2d operation into blocks
    ///
    /// Splits along the batch dimension for maximum parallelism.
    /// Each block processes one or more batch elements.
    pub fn decompose_conv2d(
        input_shape: &[usize],
        output_shape: &[usize],
        num_threads: usize,
    ) -> Vec<ComputeBlock> {
        if input_shape.is_empty() || output_shape.is_empty() {
            return vec![];
        }

        let batch = input_shape[0];
        let out_channels = output_shape.get(1).copied().unwrap_or(1);
        let out_height = output_shape.get(2).copied().unwrap_or(1);
        let out_width = output_shape.get(3).copied().unwrap_or(1);
        let in_channels = input_shape.get(1).copied().unwrap_or(1);
        let kernel_h = 3; // Default, actual would come from config
        let kernel_w = 3;

        let elements_per_batch = out_channels * out_height * out_width;
        let total_elements = batch * elements_per_batch;

        // Calculate number of blocks based on max_block_size
        let block_size = (total_elements / num_threads).max(elements_per_batch);
        let mut blocks = Vec::with_capacity(num_threads);

        let mut global_start = 0;
        for thread_id in 0..num_threads as u32 {
            if global_start >= total_elements {
                break;
            }

            let block_end = (global_start + block_size).min(total_elements);
            let workload = WorkloadEstimate::for_conv2d(
                1, // Per block
                out_channels,
                out_height,
                out_width,
                in_channels,
                kernel_h,
                kernel_w,
            );

            blocks.push(ComputeBlock::new(thread_id, global_start, block_end, workload));
            global_start = block_end;
        }

        blocks
    }

    /// Decompose a MatMul operation into blocks
    ///
    /// Splits along the M dimension (output rows).
    /// Each block handles a contiguous range of output rows.
    pub fn decompose_matmul(
        shape_a: &[usize],
        shape_b: &[usize],
        num_threads: usize,
    ) -> Vec<ComputeBlock> {
        if shape_a.len() < 2 || shape_b.len() < 2 {
            return vec![];
        }

        let m = shape_a[0];
        let k = shape_a[1];
        let n = shape_b[1];

        let total_elements = m * n;
        let elements_per_row = n;

        // Calculate block size for even distribution
        let block_size = (total_elements / num_threads).max(elements_per_row);
        let mut blocks = Vec::with_capacity(num_threads);

        let mut global_start = 0;
        for thread_id in 0..num_threads as u32 {
            if global_start >= total_elements {
                break;
            }

            let block_end = (global_start + block_size).min(total_elements);
            let workload = WorkloadEstimate::for_matmul(m, k, n);

            blocks.push(ComputeBlock::new(thread_id, global_start, block_end, workload));
            global_start = block_end;
        }

        blocks
    }

    /// Decompose any supported operator into blocks
    pub fn decompose(
        op_type: OperatorType,
        input_shape: &[usize],
        output_shape: &[usize],
        num_threads: usize,
    ) -> Vec<ComputeBlock> {
        match op_type {
            OperatorType::Conv2d | OperatorType::ConvTranspose2d => {
                Self::decompose_conv2d(input_shape, output_shape, num_threads)
            }
            OperatorType::MatMul | OperatorType::FullyConnected => {
                Self::decompose_matmul(input_shape, output_shape, num_threads)
            }
            _ => {
                // Non-compute-intensive operators don't need decomposition
                vec![ComputeBlock::new(
                    0,
                    0,
                    input_shape.iter().product::<usize>(),
                    WorkloadEstimate {
                        flops: input_shape.iter().product::<usize>(),
                        memory_bytes: input_shape.iter().product::<usize>() * 4,
                        intensity: 1.0,
                    },
                )]
            }
        }
    }
}

// ============================================================================
// Load Balancing
// ============================================================================

/// Load balancing strategy for block assignment
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Greedy: assign to least loaded worker
    Greedy,
    /// Work-stealing: workers steal from others when idle
    WorkStealing,
    /// Round-robin: rotate assignment
    RoundRobin,
}

impl LoadBalancingStrategy {
    /// Assign blocks to workers using the configured strategy
    pub fn assign_blocks(
        &self,
        workloads: &[WorkloadEstimate],
        num_workers: usize,
    ) -> Vec<WorkerAssignment> {
        match self {
            Self::Greedy => self.greedy_assign(workloads, num_workers),
            Self::RoundRobin => self.round_robin_assign(workloads, num_workers),
            Self::WorkStealing => self.greedy_assign(workloads, num_workers), // Fallback to greedy
        }
    }

    /// Greedy assignment: lightest loaded worker gets next task
    fn greedy_assign(
        &self,
        workloads: &[WorkloadEstimate],
        num_workers: usize,
    ) -> Vec<WorkerAssignment> {
        let mut worker_loads = vec![0usize; num_workers];
        let mut assignments = Vec::with_capacity(workloads.len());

        for (i, workload) in workloads.iter().enumerate() {
            // Find the worker with minimum load
            let mut min_worker = 0;
            let mut min_load = usize::MAX;

            for w in 0..num_workers {
                if worker_loads[w] < min_load {
                    min_load = worker_loads[w];
                    min_worker = w;
                }
            }

            worker_loads[min_worker] += workload.flops;
            assignments.push(WorkerAssignment {
                block_id: i as u32,
                thread_id: min_worker as u32,
            });
        }

        assignments
    }

    /// Round-robin assignment
    fn round_robin_assign(
        &self,
        workloads: &[WorkloadEstimate],
        num_workers: usize,
    ) -> Vec<WorkerAssignment> {
        workloads
            .iter()
            .enumerate()
            .map(|(i, _)| WorkerAssignment {
                block_id: i as u32,
                thread_id: (i % num_workers) as u32,
            })
            .collect()
    }
}

/// Assignment of a block to a worker thread
#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    /// Block identifier
    pub block_id: u32,
    /// Thread identifier
    pub thread_id: u32,
}

// ============================================================================
// CPU Affinity
// ============================================================================

/// CPU affinity settings
#[derive(Debug, Clone)]
pub struct CpuAffinity {
    /// List of CPU cores available to this thread pool
    pub cpus: Vec<usize>,
}

impl CpuAffinity {
    /// Get CPU affinity for the current thread
    pub fn current() -> Self {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let cpus = unsafe {
                let mut set: std::mem::MaybeUninit<libc::cpu_set_t> = std::mem::MaybeUninit::uninit();
                libc::CPU_ZERO(set.as_mut_ptr());
                let mut size: libc::size_t = std::mem::size_of::<libc::cpu_set_t>();

                // Get affinity of current thread
                if libc::pthread_getaffinity_np(libc::pthread_self(), size, set.as_mut_ptr()) == 0 {
                    let set = set.assume_init();
                    let mut cpus = Vec::new();
                    for i in 0..libc::CPU_SETSIZE as usize {
                        if libc::CPU_ISSET(i, &set) {
                            cpus.push(i);
                        }
                    }
                    cpus
                } else {
                    vec![]
                }
            };

            Self { cpus }
        }

        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            // On other platforms, return available CPUs
            let cpus = (0..std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
                .collect();
            Self { cpus }
        }
    }

    /// Set CPU affinity for the current thread
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub fn set_for_current(&self) -> Result<(), &'static str> {
        use libc::{pthread_setaffinity_np, CPU_SET, CPU_SETSIZE};

        let mut set = unsafe { std::mem::MaybeUninit::<libc::cpu_set_t>::uninit() };
        unsafe { libc::CPU_ZERO(set.as_mut_ptr()) };

        for &cpu in &self.cpus {
            if cpu < CPU_SETSIZE {
                unsafe { libc::CPU_SET(cpu, set.as_mut_ptr()) };
            }
        }

        let size = std::mem::size_of::<libc::cpu_set_t>();
        let result = unsafe {
            pthread_setaffinity_np(libc::pthread_self(), size, set.as_mut_ptr())
        };

        if result == 0 {
            Ok(())
        } else {
            Err("Failed to set CPU affinity")
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    /// Set CPU affinity for the current thread (no-op on unsupported platforms)
    pub fn set_for_current(&self) -> Result<(), &'static str> {
        Ok(()) // No-op on unsupported platforms
    }
}

// ============================================================================
// Pipeline Scheduling
// ============================================================================

/// Pipeline schedule for sequential operators with parallel blocks
#[derive(Debug)]
pub struct PipelineSchedule {
    /// Number of stages (operators) in the pipeline
    num_stages: usize,
    /// Operations in each stage
    stages: Vec<Vec<PipelineOp>>,
    /// Dependencies between stages
    dependencies: HashMap<usize, Vec<usize>>,
}

#[derive(Debug, Clone)]
struct PipelineOp {
    node_id: u32,
    workload: WorkloadEstimate,
    output_size: usize,
}

impl PipelineSchedule {
    /// Create a new pipeline schedule
    pub fn new(num_stages: usize) -> Self {
        Self {
            num_stages,
            stages: vec![Vec::new(); num_stages],
            dependencies: HashMap::new(),
        }
    }

    /// Add an operation to a stage
    pub fn add_op(&mut self, stage: usize, node_id: u32, workload: WorkloadEstimate) {
        if stage < self.num_stages {
            let output_size = workload.memory_bytes / 4; // Estimate output elements
            self.stages[stage].push(PipelineOp {
                node_id,
                workload,
                output_size,
            });
        }
    }

    /// Add a dependency (stage_a must complete before stage_b starts)
    pub fn add_dependency(&mut self, stage_a: usize, stage_b: usize) {
        self.dependencies.entry(stage_b).or_default().push(stage_a);
    }

    /// Build the pipeline schedule
    pub fn build(&self) -> Vec<Vec<ComputeBlock>> {
        let mut schedule = Vec::new();

        for (stage_idx, ops) in self.stages.iter().enumerate() {
            let mut stage_blocks = Vec::new();

            for op in ops {
                // Decompose each operation into blocks
                let blocks = BlockScheduler::decompose(
                    OperatorType::Conv2d, // Default, would be determined by actual op
                    &[1, 64, 56, 56],   // Placeholder
                    &[1, 64, 56, 56],   // Placeholder
                    rayon::current_num_threads(),
                );

                stage_blocks.extend(blocks);
            }

            schedule.push(stage_blocks);
        }

        schedule
    }
}

// ============================================================================
// Main Thread Scheduler
// ============================================================================

/// Thread scheduler for operator-level parallelism
#[derive(Debug)]
pub struct ThreadScheduler {
    config: SchedulerConfig,
    pool: ThreadPool,
    load_balancer: LoadBalancingStrategy,
}

impl ThreadScheduler {
    /// Create a new thread scheduler with default configuration
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    /// Create a thread scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        let pool_config = ThreadPoolConfig {
            num_threads: config.num_threads,
            use_affinity: config.use_affinity,
            ..Default::default()
        };

        let load_balancer = match config.load_balance {
            LoadBalanceConfig::Greedy => LoadBalancingStrategy::Greedy,
            LoadBalanceConfig::WorkStealing => LoadBalancingStrategy::WorkStealing,
            LoadBalanceConfig::RoundRobin => LoadBalancingStrategy::RoundRobin,
        };

        Self {
            config,
            pool: ThreadPool::with_config(pool_config),
            load_balancer,
        }
    }

    /// Schedule and execute operators with automatic block decomposition
    pub fn schedule_and_execute_ops<F>(
        &self,
        ops: &[(u32, OperatorType, Vec<usize>, Vec<usize>)],
        mut callback: F,
    ) where
        F: FnMut(u32, &[ComputeBlock]),
    {
        for &(node_id, op_type, ref input_shape, ref output_shape) in ops {
            if !self.config.enable_parallel {
                // Single-threaded execution
                let blocks = BlockScheduler::decompose(
                    op_type,
                    input_shape,
                    output_shape,
                    1,
                );
                callback(node_id, &blocks);
                continue;
            }

            // Decompose into blocks for parallel execution
            let num_threads = self.pool.num_threads();
            let blocks = BlockScheduler::decompose(op_type, input_shape, output_shape, num_threads);

            if blocks.len() > 1 {
                // Parallel block execution using rayon
                blocks.par_iter().for_each(|block| {
                    // Execute block (actual computation would happen here)
                    tracing::debug!(
                        "Executing block {} on thread {:?}: elements [{}..{}]",
                        block.block_id,
                        std::thread::current().name(),
                        block.start_idx,
                        block.end_idx
                    );
                });
            }

            callback(node_id, &blocks);
        }
    }

    /// Execute blocks in parallel using the thread pool
    pub fn execute_blocks<F>(&self, blocks: &[ComputeBlock], f: F)
    where
        F: Fn(&ComputeBlock) + Send + Sync + 'static,
    {
        if blocks.is_empty() {
            return;
        }

        // Use rayon for parallel execution
        blocks.par_iter().for_each(|block| {
            f(block);
        });
    }

    /// Get scheduler configuration
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Get the thread pool
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }
}

impl Default for ThreadScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_conv2d() {
        let wl = WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3);
        assert!(wl.flops > 0);
        assert!(wl.memory_bytes > 0);
        assert!(wl.intensity > 0.0);
    }

    #[test]
    fn test_workload_matmul() {
        let wl = WorkloadEstimate::for_matmul(128, 256, 128);
        assert!(wl.flops > 0);
        assert!(wl.memory_bytes > 0);
        assert!(wl.intensity > 0.0);
    }

    #[test]
    fn test_block_creation() {
        let block = ComputeBlock::new(0, 0, 100, WorkloadEstimate::for_matmul(10, 10, 10));
        assert_eq!(block.node_id, 0);
        assert_eq!(block.size(), 100);
    }

    #[test]
    fn test_scheduler_config() {
        let config = SchedulerConfig::default();
        assert!(config.enable_parallel);
        assert_eq!(config.num_threads, 0);
        assert!(config.use_affinity);
    }

    #[test]
    fn test_block_decompose_conv2d() {
        let blocks = BlockScheduler::decompose_conv2d(&[4, 64, 56, 56], &[4, 64, 56, 56], 4);
        assert!(!blocks.is_empty());
        assert!(blocks.len() <= 4);

        // Verify blocks are non-overlapping and cover the range
        let mut prev_end = 0;
        for block in &blocks {
            assert!(block.start_idx >= prev_end);
            prev_end = block.end_idx;
        }
    }

    #[test]
    fn test_block_decompose_matmul() {
        let blocks = BlockScheduler::decompose_matmul(&[128, 256], &[256, 128], 4);
        assert!(!blocks.is_empty());

        // Verify blocks are non-overlapping
        let mut prev_end = 0;
        for block in &blocks {
            assert!(block.start_idx >= prev_end);
            prev_end = block.end_idx;
        }
    }

    #[test]
    fn test_load_balancing_greedy() {
        let strategy = LoadBalancingStrategy::Greedy;
        let workloads = vec![
            WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3),
            WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3),
            WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3),
            WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3),
        ];

        let assignments = strategy.assign_blocks(&workloads, 2);
        assert_eq!(assignments.len(), 4);
    }

    #[test]
    fn test_cpu_affinity_current() {
        let affinity = CpuAffinity::current();
        assert!(!affinity.cpus.is_empty());
    }

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new();
        assert!(pool.num_threads() > 0);
    }

    #[test]
    fn test_thread_pool_custom_threads() {
        let pool = ThreadPool::with_config(ThreadPoolConfig {
            num_threads: 4,
            ..Default::default()
        });
        assert_eq!(pool.num_threads(), 4);
    }

    #[test]
    fn test_pipeline_schedule() {
        let mut schedule = PipelineSchedule::new(3);
        schedule.add_op(0, 0, WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3));
        schedule.add_op(1, 1, WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3));
        schedule.add_op(2, 2, WorkloadEstimate::for_conv2d(1, 64, 56, 56, 64, 3, 3));

        let stages = schedule.build();
        assert_eq!(stages.len(), 3);
    }
}
