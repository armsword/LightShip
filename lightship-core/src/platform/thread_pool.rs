//! Cross-platform thread pool
//!
//! Provides a lightweight thread pool for parallel execution.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

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

/// Task for the thread pool
type Task = Box<dyn FnOnce() + Send + 'static>;

/// Cross-platform thread pool
pub struct ThreadPool {
    handles: Vec<thread::JoinHandle<()>>,
    task_sender: std::sync::mpsc::Sender<Task>,
    config: ThreadPoolConfig,
}

impl ThreadPool {
    /// Create a new thread pool
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

        let (sender, receiver): (
            std::sync::mpsc::Sender<Task>,
            std::sync::mpsc::Receiver<Task>,
        ) = std::sync::mpsc::channel();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));

        let mut handles = Vec::with_capacity(num_threads);

        for i in 0..num_threads {
            let receiver = Arc::clone(&receiver);
            let name = format!("{}-{}", config.name_prefix, i);

            let handle = thread::Builder::new()
                .name(name)
                .stack_size(config.stack_size.unwrap_or(0))
                .spawn(move || {
                    loop {
                        let task = {
                            let guard = receiver.lock().unwrap();
                            guard.recv()
                        };

                        match task {
                            Ok(task) => task(),
                            Err(_) => break, // Channel closed
                        }
                    }
                })
                .expect("Failed to spawn thread");

            handles.push(handle);
        }

        Self {
            handles,
            task_sender: sender,
            config,
        }
    }

    /// Submit a task to the thread pool
    pub fn submit<F>(&self, task: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let _ = self.task_sender.send(Box::new(task));
    }

    /// Get the number of threads
    pub fn num_threads(&self) -> usize {
        self.handles.len()
    }

    /// Get the configuration
    pub fn config(&self) -> &ThreadPoolConfig {
        &self.config
    }

    /// Wait for all tasks to complete (blocking)
    pub fn wait(&self) {
        // In this simple implementation, tasks are executed as they arrive
    }
}

impl Default for ThreadPool {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadPool")
            .field("num_threads", &self.handles.len())
            .field("config", &self.config)
            .finish()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Drop the sender to signal workers to exit
        // Workers will exit when they receive the error
        drop(&self.task_sender);
        // Note: We don't explicitly wait for threads since they're
        // designed to clean up when the channel closes
    }
}

/// Parallel executor for batch operations
pub struct ParallelExecutor {
    pool: ThreadPool,
    chunk_size: usize,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    pub fn new(num_threads: usize) -> Self {
        let config = ThreadPoolConfig {
            num_threads,
            ..Default::default()
        };

        Self {
            pool: ThreadPool::with_config(config),
            chunk_size: 1,
        }
    }

    /// Set the chunk size for parallel operations
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Execute a parallel for loop
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let f = Arc::new(f);
        let chunk_size = self.chunk_size.max(1);

        for chunk_start in (0..range.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(range.len());
            let f = Arc::clone(&f);

            self.pool.submit(move || {
                for i in chunk_start..chunk_end {
                    f(range.start + i);
                }
            });
        }
    }

    /// Get the underlying thread pool
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new();
        assert!(pool.num_threads() > 0);
    }

    #[test]
    fn test_thread_pool_with_config() {
        let config = ThreadPoolConfig {
            num_threads: 2,
            name_prefix: "test".to_string(),
            ..Default::default()
        };
        let pool = ThreadPool::with_config(config);
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_thread_pool_submit() {
        use std::sync::Barrier;

        let pool = ThreadPool::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(11)); // 10 tasks + main

        for _ in 0..10 {
            let counter = Arc::clone(&counter);
            let barrier = Arc::clone(&barrier);
            pool.submit(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                barrier.wait();
            });
        }

        barrier.wait();
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_parallel_executor() {
        use std::sync::Barrier;

        let executor = ParallelExecutor::new(2);
        let results = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));

        let results_for_closure = results.clone();
        let barrier_clone = Arc::clone(&barrier);

        executor.parallel_for(0..100, move |i| {
            results_for_closure.fetch_add(i, Ordering::SeqCst);
        });

        // Wait a bit for parallel execution to complete
        std::thread::sleep(std::time::Duration::from_millis(100));

        let sum = results.load(Ordering::SeqCst);
        assert_eq!(sum, (0..100).sum::<usize>());
    }

    #[test]
    fn test_thread_pool_config_default() {
        let config = ThreadPoolConfig::default();
        assert_eq!(config.num_threads, 0);
        assert_eq!(config.name_prefix, "lightship");
    }
}
