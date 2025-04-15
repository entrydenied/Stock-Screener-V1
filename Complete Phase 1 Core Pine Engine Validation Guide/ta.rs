// Technical Analysis functions module
// Implements TA functions for Pine Script

use std::collections::HashMap;
use crate::runtime::environment::Environment;
use crate::runtime::error::RuntimeError;
use crate::runtime::value::Value;

/// Register all technical analysis functions
pub fn register_functions(env: &mut Environment) {
    // Register HMA function
    env.register_function("ta.hma", hma);
    
    // Register crossover function
    env.register_function("ta.crossover", crossover);
    
    // Register crossunder function
    env.register_function("ta.crossunder", crossunder);
    
    // Register valuewhen function
    env.register_function("ta.valuewhen", valuewhen);
}

/// Hull Moving Average (HMA)
/// 
/// HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
fn hma(env: &mut Environment, args: &[Value], named_args: &HashMap<String, Value>) -> Result<Value, RuntimeError> {
    if args.len() < 2 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.hma".to_string(),
            message: "Expected at least 2 arguments: source and length".to_string(),
        });
    }
    
    // Get source series
    let source = match &args[0] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric".to_string(),
                found: args[0].type_name(),
                message: "First argument to ta.hma must be a series or numeric value".to_string(),
            });
        }
    };
    
    // Get length
    let length = match &args[1] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "integer".to_string(),
                found: args[1].type_name(),
                message: "Second argument to ta.hma must be an integer".to_string(),
            });
        }
    };
    
    if length < 1 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.hma".to_string(),
            message: "Length must be at least 1".to_string(),
        });
    }
    
    // Calculate HMA
    let half_length = length / 2;
    let sqrt_length = (length as f64).sqrt() as usize;
    
    if sqrt_length < 1 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.hma".to_string(),
            message: "Square root of length must be at least 1".to_string(),
        });
    }
    
    // Calculate WMA with length
    let wma_full = weighted_moving_average(&source, length);
    
    // Calculate WMA with half length
    let wma_half = weighted_moving_average(&source, half_length);
    
    // Calculate 2 * WMA(half) - WMA(full)
    let mut intermediate = Vec::with_capacity(source.len());
    for i in 0..source.len() {
        if i < wma_half.len() && i < wma_full.len() {
            let wma_half_val = match &wma_half[i] {
                Value::Float(f) => *f,
                _ => 0.0, // Should not happen
            };
            
            let wma_full_val = match &wma_full[i] {
                Value::Float(f) => *f,
                _ => 0.0, // Should not happen
            };
            
            intermediate.push(Value::Float(2.0 * wma_half_val - wma_full_val));
        } else {
            intermediate.push(Value::NA);
        }
    }
    
    // Calculate final WMA with sqrt(length)
    let hma_result = weighted_moving_average(&intermediate, sqrt_length);
    
    Ok(Value::Series(hma_result))
}

/// Weighted Moving Average helper function
fn weighted_moving_average(source: &[Value], length: usize) -> Vec<Value> {
    if length == 0 {
        return vec![Value::NA; source.len()];
    }
    
    let mut result = Vec::with_capacity(source.len());
    let denominator = (length * (length + 1)) / 2;
    
    for i in 0..source.len() {
        if i < length - 1 {
            // Not enough data yet
            result.push(Value::NA);
            continue;
        }
        
        let mut sum = 0.0;
        let mut weight_sum = 0;
        let mut has_na = false;
        
        for j in 0..length {
            let idx = i - j;
            let weight = length - j;
            
            match &source[idx] {
                Value::Float(f) => {
                    sum += f * weight as f64;
                    weight_sum += weight;
                },
                Value::Integer(n) => {
                    sum += *n as f64 * weight as f64;
                    weight_sum += weight;
                },
                Value::NA => {
                    has_na = true;
                    break;
                },
                _ => {
                    // Skip non-numeric values
                    has_na = true;
                    break;
                }
            }
        }
        
        if has_na || weight_sum == 0 {
            result.push(Value::NA);
        } else {
            result.push(Value::Float(sum / weight_sum as f64));
        }
    }
    
    result
}

/// Crossover function
/// Returns true when series1 crosses above series2
fn crossover(env: &mut Environment, args: &[Value], named_args: &HashMap<String, Value>) -> Result<Value, RuntimeError> {
    if args.len() < 2 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.crossover".to_string(),
            message: "Expected 2 arguments: series1 and series2".to_string(),
        });
    }
    
    // Get series1
    let series1 = match &args[0] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric".to_string(),
                found: args[0].type_name(),
                message: "First argument to ta.crossover must be a series or numeric value".to_string(),
            });
        }
    };
    
    // Get series2
    let series2 = match &args[1] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric".to_string(),
                found: args[1].type_name(),
                message: "Second argument to ta.crossover must be a series or numeric value".to_string(),
            });
        }
    };
    
    // Calculate crossover
    let mut result = Vec::with_capacity(series1.len().max(series2.len()));
    
    // Fill with NA for the first value since we need at least 2 values to detect a crossover
    if !series1.is_empty() && !series2.is_empty() {
        result.push(Value::Boolean(false));
    }
    
    for i in 1..series1.len().max(series2.len()) {
        if i >= series1.len() || i >= series2.len() || i - 1 >= series1.len() || i - 1 >= series2.len() {
            result.push(Value::Boolean(false));
            continue;
        }
        
        let s1_curr = match &series1[i] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s1_prev = match &series1[i - 1] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s2_curr = match &series2[i] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s2_prev = match &series2[i - 1] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        // Crossover occurs when previous value of series1 is <= previous value of series2
        // AND current value of series1 is > current value of series2
        let crossover = s1_prev <= s2_prev && s1_curr > s2_curr;
        result.push(Value::Boolean(crossover));
    }
    
    Ok(Value::Series(result))
}

/// Crossunder function
/// Returns true when series1 crosses below series2
fn crossunder(env: &mut Environment, args: &[Value], named_args: &HashMap<String, Value>) -> Result<Value, RuntimeError> {
    if args.len() < 2 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.crossunder".to_string(),
            message: "Expected 2 arguments: series1 and series2".to_string(),
        });
    }
    
    // Get series1
    let series1 = match &args[0] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric".to_string(),
                found: args[0].type_name(),
                message: "First argument to ta.crossunder must be a series or numeric value".to_string(),
            });
        }
    };
    
    // Get series2
    let series2 = match &args[1] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric".to_string(),
                found: args[1].type_name(),
                message: "Second argument to ta.crossunder must be a series or numeric value".to_string(),
            });
        }
    };
    
    // Calculate crossunder
    let mut result = Vec::with_capacity(series1.len().max(series2.len()));
    
    // Fill with NA for the first value since we need at least 2 values to detect a crossunder
    if !series1.is_empty() && !series2.is_empty() {
        result.push(Value::Boolean(false));
    }
    
    for i in 1..series1.len().max(series2.len()) {
        if i >= series1.len() || i >= series2.len() || i - 1 >= series1.len() || i - 1 >= series2.len() {
            result.push(Value::Boolean(false));
            continue;
        }
        
        let s1_curr = match &series1[i] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s1_prev = match &series1[i - 1] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s2_curr = match &series2[i] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        let s2_prev = match &series2[i - 1] {
            Value::Float(f) => *f,
            Value::Integer(n) => *n as f64,
            _ => {
                result.push(Value::Boolean(false));
                continue;
            }
        };
        
        // Crossunder occurs when previous value of series1 is >= previous value of series2
        // AND current value of series1 is < current value of series2
        let crossunder = s1_prev >= s2_prev && s1_curr < s2_curr;
        result.push(Value::Boolean(crossunder));
    }
    
    Ok(Value::Series(result))
}

/// Valuewhen function
/// Returns the value of source when condition was true lookback bars ago
fn valuewhen(env: &mut Environment, args: &[Value], named_args: &HashMap<String, Value>) -> Result<Value, RuntimeError> {
    if args.len() < 3 {
        return Err(RuntimeError::InvalidArguments {
            function: "ta.valuewhen".to_string(),
            message: "Expected 3 arguments: condition, source, and occurrence".to_string(),
        });
    }
    
    // Get condition series
    let condition = match &args[0] {
        Value::Series(series) => series.clone(),
        Value::Boolean(b) => vec![Value::Boolean(*b)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or boolean".to_string(),
                found: args[0].type_name(),
                message: "First argument to ta.valuewhen must be a series or boolean value".to_string(),
            });
        }
    };
    
    // Get source series
    let source = match &args[1] {
        Value::Series(series) => series.clone(),
        Value::Float(f) => vec![Value::Float(*f)],
        Value::Integer(i) => vec![Value::Float(*i as f64)],
        Value::Boolean(b) => vec![Value::Boolean(*b)],
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "series or numeric or boolean".to_string(),
                found: args[1].type_name(),
                message: "Second argument to ta.valuewhen must be a series or numeric or boolean value".to_string(),
            });
        }
    };
    
    // Get occurrence (lookback)
    let occurrence = match &args[2] {
        Value::Integer(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => {
            return Err(RuntimeError::TypeError {
                expected: "integer".to_string(),
                found: args[2].type_name(),
                message: "Third argument to ta.valuewhen must be an integer".to_string(),
            });
        }
    };
    
    // Calculate valuewhen
    let mut result = Vec::with_capacity(condition.len());
    let mut true_indices = Vec::new();
    
    for i in 0..condition.len() {
        let is_true = match &condition[i] {
            Value::Boolean(b) => *b,
            _ => false,
        };
        
        if is_true {
            true_indices.push(i);
        }
        
        if true_indices.len() > occurrence && i < source.len() {
            let idx = true_indices[true_indices.len() - 1 - occurrence];
            if idx < source.len() {
                result.push(source[idx].clone());
            } else {
                result.push(Value::NA);
            }
        } else {
            result.push(Value::NA);
        }
    }
    
    Ok(Value::Series(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hma() {
        let mut env = Environment::new();
        register_functions(&mut env);
        
        // Create a simple series
        let series = vec![
            Value::Float(1.0),
            Value::Float(2.0),
            Value::Float(3.0),
            Value::Float(4.0),
            Value::Float(5.0),
            Value::Float(6.0),
            Value::Float(7.0),
            Value::Float(8.0),
            Value::Float(9.0),
            Value::Float(10.0),
        ];
        
        // Test HMA with length 4
        let args = vec![Value::Series(series.clone()), Value::Integer(4)];
        let result = hma(&mut env, &args, &HashMap::new()).unwrap();
        
        if let Value::Series(hma_result) = result {
            // First 3 values should be NA (not enough data)
            assert!(matches!(hma_result[0], Value::NA));
            assert!(matches!(hma_result[1], Value::NA));
            assert!(matches!(hma_result[2], Value::NA));
            
            // Check that we have values for the rest
            for i in 3..hma_result.len() {
                assert!(!matches!(hma_result[i], Value::NA));
            }
        } else {
            panic!("Expected Series result");
        }
    }
    
    #[test]
    fn test_crossover() {
        let mut env = Environme
(Content truncated due to size limit. Use line ranges to read in chunks)