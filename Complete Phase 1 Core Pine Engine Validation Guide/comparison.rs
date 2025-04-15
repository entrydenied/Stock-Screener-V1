// Comparison module for validation
// Responsible for comparing Pine Engine results with TradingView results

use crate::runtime::execution::ExecutionResult;
use crate::runtime::value::Value;
use crate::validation::{ValidationError, TradingViewResult, ComparisonResult, Mismatch};
use std::collections::HashMap;

/// Compare Pine Engine results with TradingView results
pub fn compare_results(
    engine_result: &ExecutionResult,
    tv_result: &TradingViewResult,
) -> Result<ComparisonResult, ValidationError> {
    let mut mismatches = Vec::new();
    
    // Compare alert counts
    if engine_result.alerts.len() != tv_result.alerts.len() {
        mismatches.push(Mismatch::AlertCount {
            engine_count: engine_result.alerts.len(),
            tv_count: tv_result.alerts.len(),
        });
    }
    
    // Compare alerts
    let mut alert_matches = 0;
    for (i, tv_alert) in tv_result.alerts.iter().enumerate() {
        if i < engine_result.alerts.len() {
            let engine_alert = &engine_result.alerts[i];
            
            // Extract title and message from engine alert
            let engine_title = engine_alert.get("title")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            
            let engine_message = engine_alert.get("message")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            
            // Compare title and message
            if engine_title != tv_alert.title || engine_message != tv_alert.message {
                mismatches.push(Mismatch::AlertMismatch {
                    index: i,
                    engine_alert: format!("{}: {}", engine_title, engine_message),
                    tv_alert: format!("{}: {}", tv_alert.title, tv_alert.message),
                });
            } else {
                alert_matches += 1;
            }
        }
    }
    
    // Compare plot counts
    if engine_result.plots.len() != tv_result.plots.len() {
        mismatches.push(Mismatch::PlotCount {
            engine_count: engine_result.plots.len(),
            tv_count: tv_result.plots.len(),
        });
    }
    
    // Compare plot values
    let mut plot_value_matches = 0;
    let mut total_plot_values = 0;
    
    for (i, tv_plot) in tv_result.plots.iter().enumerate() {
        if i < engine_result.plots.len() {
            let (engine_plot_value, engine_plot_params) = &engine_result.plots[i];
            
            // Extract plot name from engine plot
            let engine_plot_name = engine_plot_params.get("title")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            
            // Compare plot values
            if let Value::Series(engine_values) = engine_plot_value {
                for (j, tv_value) in tv_plot.values.iter().enumerate() {
                    if j < engine_values.len() {
                        total_plot_values += 1;
                        
                        if let Value::Float(engine_value) = engine_values[j] {
                            // Allow for small floating-point differences
                            let difference = (engine_value - tv_value).abs();
                            if difference > 0.0001 {
                                mismatches.push(Mismatch::PlotValueMismatch {
                                    plot_name: tv_plot.name.clone(),
                                    index: j,
                                    engine_value,
                                    tv_value: *tv_value,
                                    difference,
                                });
                            } else {
                                plot_value_matches += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Calculate match percentages
    let alert_match_percentage = if tv_result.alerts.is_empty() {
        100.0
    } else {
        (alert_matches as f64 / tv_result.alerts.len() as f64) * 100.0
    };
    
    let plot_match_percentage = if total_plot_values == 0 {
        100.0
    } else {
        (plot_value_matches as f64 / total_plot_values as f64) * 100.0
    };
    
    Ok(ComparisonResult {
        alert_match_percentage,
        plot_match_percentage,
        mismatches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::{TradingViewAlert, TradingViewPlot};
    
    #[test]
    fn test_compare_matching_results() {
        // Create engine result
        let mut engine_result = ExecutionResult::new();
        
        let mut alert_params = HashMap::new();
        alert_params.insert("title".to_string(), Value::String("Buy Signal".to_string()));
        alert_params.insert("message".to_string(), Value::String("Hull Crossing above MA_Min, Bullish".to_string()));
        engine_result.alerts.push(alert_params);
        
        let plot_value = Value::Series(vec![Value::Float(100.5), Value::Float(101.2), Value::Float(102.3)]);
        let mut plot_params = HashMap::new();
        plot_params.insert("title".to_string(), Value::String("HMA".to_string()));
        plot_params.insert("color".to_string(), Value::String("#00fa03".to_string()));
        engine_result.plots.push((plot_value, plot_params));
        
        // Create matching TradingView result
        let tv_result = TradingViewResult {
            alerts: vec![
                TradingViewAlert {
                    timestamp: 1617235200000,
                    title: "Buy Signal".to_string(),
                    message: "Hull Crossing above MA_Min, Bullish".to_string(),
                },
            ],
            plots: vec![
                TradingViewPlot {
                    name: "HMA".to_string(),
                    values: vec![100.5, 101.2, 102.3],
                    color: "#00fa03".to_string(),
                },
            ],
        };
        
        // Compare results
        let comparison = compare_results(&engine_result, &tv_result).unwrap();
        
        // Check that there are no mismatches
        assert!(comparison.mismatches.is_empty());
        assert_eq!(comparison.alert_match_percentage, 100.0);
        assert_eq!(comparison.plot_match_percentage, 100.0);
    }
    
    #[test]
    fn test_compare_mismatched_results() {
        // Create engine result
        let mut engine_result = ExecutionResult::new();
        
        let mut alert_params = HashMap::new();
        alert_params.insert("title".to_string(), Value::String("Buy Signal".to_string()));
        alert_params.insert("message".to_string(), Value::String("Hull Crossing above MA_Min, Bullish".to_string()));
        engine_result.alerts.push(alert_params);
        
        let plot_value = Value::Series(vec![Value::Float(100.5), Value::Float(101.2), Value::Float(102.3)]);
        let mut plot_params = HashMap::new();
        plot_params.insert("title".to_string(), Value::String("HMA".to_string()));
        plot_params.insert("color".to_string(), Value::String("#00fa03".to_string()));
        engine_result.plots.push((plot_value, plot_params));
        
        // Create mismatched TradingView result
        let tv_result = TradingViewResult {
            alerts: vec![
                TradingViewAlert {
                    timestamp: 1617235200000,
                    title: "Sell Signal".to_string(), // Different title
                    message: "Hull Crossing above MA_Min, Bullish".to_string(),
                },
            ],
            plots: vec![
                TradingViewPlot {
                    name: "HMA".to_string(),
                    values: vec![100.5, 101.3, 102.3], // Different value at index 1
                    color: "#00fa03".to_string(),
                },
            ],
        };
        
        // Compare results
        let comparison = compare_results(&engine_result, &tv_result).unwrap();
        
        // Check that there are mismatches
        assert_eq!(comparison.mismatches.len(), 2);
        assert!(comparison.alert_match_percentage < 100.0);
        assert!(comparison.plot_match_percentage < 100.0);
    }
}
