//! Timing information for inference
//!
//! Provides timing metrics for tracking inference performance.

use std::fmt;
use std::time::{Duration, Instant};

/// Timing information for a single inference
#[derive(Debug, Clone)]
pub struct TimingInfo {
    /// Total inference time
    pub total_time: Duration,
    /// Time spent in model loading (if applicable)
    pub load_time: Option<Duration>,
    /// Time spent in compilation (if applicable)
    pub compile_time: Option<Duration>,
    /// Time spent in execution
    pub execution_time: Duration,
    /// Per-operator timing breakdown
    pub operator_times: Vec<OperatorTiming>,
    /// Memory allocation time
    pub memory_time: Option<Duration>,
    /// Whether timing is valid
    pub is_valid: bool,
}

impl TimingInfo {
    /// Create a new empty timing info
    pub fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            load_time: None,
            compile_time: None,
            execution_time: Duration::ZERO,
            operator_times: Vec::new(),
            memory_time: None,
            is_valid: false,
        }
    }

    /// Create timing info with execution time only
    pub fn from_execution(execution_time: Duration) -> Self {
        Self {
            total_time: execution_time,
            load_time: None,
            compile_time: None,
            execution_time,
            operator_times: Vec::new(),
            memory_time: None,
            is_valid: true,
        }
    }

    /// Set total time
    pub fn with_total_time(mut self, time: Duration) -> Self {
        self.total_time = time;
        self
    }

    /// Set load time
    pub fn with_load_time(mut self, time: Duration) -> Self {
        self.load_time = Some(time);
        self
    }

    /// Set compile time
    pub fn with_compile_time(mut self, time: Duration) -> Self {
        self.compile_time = Some(time);
        self
    }

    /// Set execution time
    pub fn with_execution_time(mut self, time: Duration) -> Self {
        self.execution_time = time;
        self
    }

    /// Add an operator timing
    pub fn add_operator_time(&mut self, timing: OperatorTiming) {
        self.operator_times.push(timing);
    }

    /// Get total operator time
    pub fn total_operator_time(&self) -> Duration {
        self.operator_times.iter().map(|t| t.duration).sum()
    }

    /// Get time in nanoseconds
    pub fn total_time_ns(&self) -> u64 {
        self.total_time.as_nanos() as u64
    }

    /// Get execution time in nanoseconds
    pub fn execution_time_ns(&self) -> u64 {
        self.execution_time.as_nanos() as u64
    }

    /// Check if timing is valid
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
}

impl Default for TimingInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TimingInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimingInfo(total={:.3}ms", self.total_time.as_secs_f64() * 1000.0)?;
        if let Some(load) = self.load_time {
            write!(f, ", load={:.3}ms", load.as_secs_f64() * 1000.0)?;
        }
        if let Some(compile) = self.compile_time {
            write!(f, ", compile={:.3}ms", compile.as_secs_f64() * 1000.0)?;
        }
        write!(
            f,
            ", exec={:.3}ms, operators={})",
            self.execution_time.as_secs_f64() * 1000.0,
            self.operator_times.len()
        )?;
        Ok(())
    }
}

/// Timing for a single operator
#[derive(Debug, Clone)]
pub struct OperatorTiming {
    /// Operator name
    pub name: String,
    /// Operator type
    pub operator_type: String,
    /// Time spent in this operator
    pub duration: Duration,
    /// Number of times this operator was called
    pub call_count: usize,
}

impl OperatorTiming {
    /// Create a new operator timing
    pub fn new(name: String, operator_type: String, duration: Duration) -> Self {
        Self {
            name,
            operator_type,
            duration,
            call_count: 1,
        }
    }

    /// Add to the duration
    pub fn add_duration(&mut self, duration: Duration) {
        self.duration += duration;
        self.call_count += 1;
    }

    /// Get average time per call
    pub fn average_time(&self) -> Duration {
        if self.call_count > 0 {
            Duration::from_nanos(self.duration.as_nanos() as u64 / self.call_count as u64)
        } else {
            Duration::ZERO
        }
    }
}

/// Timer for measuring elapsed time
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    end: Option<Instant>,
}

impl Timer {
    /// Start the timer
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
            end: None,
        }
    }

    /// Stop the timer and return the elapsed duration
    pub fn stop(&mut self) -> Duration {
        let end = Instant::now();
        self.end = Some(end);
        end.duration_since(self.start)
    }

    /// Get elapsed time without stopping
    pub fn elapsed(&self) -> Duration {
        self.end
            .map(|e| e.duration_since(self.start))
            .unwrap_or_else(|| self.start.elapsed())
    }

    /// Check if timer is stopped
    pub fn is_stopped(&self) -> bool {
        self.end.is_some()
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::start()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_info_creation() {
        let timing = TimingInfo::new();
        assert!(!timing.is_valid());
        assert_eq!(timing.total_time, Duration::ZERO);
    }

    #[test]
    fn test_timing_info_with_execution() {
        let timing = TimingInfo::from_execution(Duration::from_millis(10));
        assert!(timing.is_valid());
        assert_eq!(timing.execution_time, Duration::from_millis(10));
    }

    #[test]
    fn test_timing_info_operators() {
        let mut timing = TimingInfo::new();
        timing.add_operator_time(OperatorTiming::new(
            "conv".to_string(),
            "Conv2d".to_string(),
            Duration::from_millis(5),
        ));
        timing.add_operator_time(OperatorTiming::new(
            "relu".to_string(),
            "ReLU".to_string(),
            Duration::from_millis(1),
        ));

        assert_eq!(timing.operator_times.len(), 2);
        assert_eq!(timing.total_operator_time(), Duration::from_millis(6));
    }

    #[test]
    fn test_timing_info_display() {
        let timing = TimingInfo::from_execution(Duration::from_millis(10));
        let s = format!("{}", timing);
        assert!(s.contains("10.000"));
    }

    #[test]
    fn test_operator_timing() {
        let mut timing = OperatorTiming::new(
            "conv".to_string(),
            "Conv2d".to_string(),
            Duration::from_millis(5),
        );
        timing.add_duration(Duration::from_millis(5));

        assert_eq!(timing.call_count, 2);
        assert_eq!(timing.average_time(), Duration::from_millis(5));
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();

        assert!(elapsed >= Duration::from_millis(10));
        assert!(timer.is_stopped());
    }

    #[test]
    fn test_timer_elapsed() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = timer.elapsed();

        assert!(elapsed >= Duration::from_millis(5));
    }
}
