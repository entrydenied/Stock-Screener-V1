// Execution module for Pine Script runtime
// Defines the execution result structure

use std::collections::HashMap;
use super::value::Value;

/// Represents the result of executing a Pine Script
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Alerts generated during execution
    pub alerts: Vec<HashMap<String, Value>>,
    
    /// Plots generated during execution
    pub plots: Vec<(Value, HashMap<String, Value>)>,
}

impl ExecutionResult {
    /// Create a new empty execution result
    pub fn new() -> Self {
        ExecutionResult {
            alerts: Vec::new(),
            plots: Vec::new(),
        }
    }
    
    /// Check if there are any alerts
    pub fn has_alerts(&self) -> bool {
        !self.alerts.is_empty()
    }
    
    /// Check if there are any plots
    pub fn has_plots(&self) -> bool {
        !self.plots.is_empty()
    }
    
    /// Get the number of alerts
    pub fn alert_count(&self) -> usize {
        self.alerts.len()
    }
    
    /// Get the number of plots
    pub fn plot_count(&self) -> usize {
        self.plots.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_execution_result() {
        let mut result = ExecutionResult::new();
        
        // Initially empty
        assert!(!result.has_alerts());
        assert!(!result.has_plots());
        assert_eq!(result.alert_count(), 0);
        assert_eq!(result.plot_count(), 0);
        
        // Add an alert
        let mut alert = HashMap::new();
        alert.insert("title".to_string(), Value::String("Test Alert".to_string()));
        result.alerts.push(alert);
        
        assert!(result.has_alerts());
        assert_eq!(result.alert_count(), 1);
        
        // Add a plot
        let plot_value = Value::Float(42.0);
        let mut plot_params = HashMap::new();
        plot_params.insert("color".to_string(), Value::Color("#FF0000".to_string()));
        result.plots.push((plot_value, plot_params));
        
        assert!(result.has_plots());
        assert_eq!(result.plot_count(), 1);
    }
}
