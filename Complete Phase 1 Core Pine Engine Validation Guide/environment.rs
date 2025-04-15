// Environment module for Pine Script runtime
// Defines the execution environment for Pine Script

use std::collections::HashMap;
use super::error::RuntimeError;
use super::value::Value;
use super::data::OHLCVData;
use std::rc::Rc;
use std::cell::RefCell;

/// Type for built-in function implementations
pub type BuiltInFunction = fn(&mut Environment, &[Value], &HashMap<String, Value>) -> Result<Value, RuntimeError>;

/// Represents a scope in the environment
#[derive(Debug)]
struct Scope {
    variables: HashMap<String, Value>,
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
        }
    }
}

/// Represents the execution environment for Pine Script
pub struct Environment {
    // Scopes for variables (stack-based)
    scopes: Vec<Scope>,
    
    // Built-in functions
    functions: HashMap<String, BuiltInFunction>,
    
    // Input data
    data: Option<Rc<OHLCVData>>,
    
    // Current bar index
    current_bar: usize,
    
    // Indicator information
    indicator_name: String,
    indicator_parameters: HashMap<String, Value>,
    
    // Results
    alerts: Vec<HashMap<String, Value>>,
    plots: Vec<(Value, HashMap<String, Value>)>,
    plot_shapes: Vec<(Value, HashMap<String, Value>)>,
    
    // Series values for functions like valuewhen
    series_values: HashMap<String, Vec<Value>>,
}

impl Environment {
    /// Create a new environment
    pub fn new() -> Self {
        let mut env = Environment {
            scopes: vec![Scope::new()], // Global scope
            functions: HashMap::new(),
            data: None,
            current_bar: 0,
            indicator_name: String::new(),
            indicator_parameters: HashMap::new(),
            alerts: Vec::new(),
            plots: Vec::new(),
            plot_shapes: Vec::new(),
            series_values: HashMap::new(),
        };
        
        // Register built-in functions
        env.register_built_in_functions();
        
        env
    }
    
    /// Register built-in functions
    fn register_built_in_functions(&mut self) {
        // This will be populated by the functions module
    }
    
    /// Set the input data
    pub fn set_data(&mut self, data: &OHLCVData) {
        self.data = Some(Rc::new(data.clone()));
        self.current_bar = 0;
    }
    
    /// Get the current data
    pub fn get_data(&self) -> Result<Rc<OHLCVData>, RuntimeError> {
        self.data.clone().ok_or_else(|| RuntimeError::DataError {
            message: "No data available".to_string(),
        })
    }
    
    /// Set the current bar index
    pub fn set_current_bar(&mut self, bar: usize) {
        self.current_bar = bar;
    }
    
    /// Get the current bar index
    pub fn get_current_bar(&self) -> usize {
        self.current_bar
    }
    
    /// Set the indicator name
    pub fn set_indicator_name(&mut self, name: String) {
        self.indicator_name = name;
    }
    
    /// Get the indicator name
    pub fn get_indicator_name(&self) -> &str {
        &self.indicator_name
    }
    
    /// Set an indicator parameter
    pub fn set_indicator_parameter(&mut self, name: String, value: Value) {
        self.indicator_parameters.insert(name, value);
    }
    
    /// Get an indicator parameter
    pub fn get_indicator_parameter(&self, name: &str) -> Option<&Value> {
        self.indicator_parameters.get(name)
    }
    
    /// Push a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }
    
    /// Pop the current scope
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }
    
    /// Set a variable in the current scope
    pub fn set_variable(&mut self, name: String, value: Value) {
        let scope = self.scopes.last_mut().unwrap();
        scope.variables.insert(name.clone(), value.clone());
        
        // If it's a series value, store it for functions like valuewhen
        if let Value::Series(_) = &value {
            self.series_values.insert(name, vec![value]);
        }
    }
    
    /// Get a variable from the environment
    pub fn get_variable(&self, name: &str) -> Result<Value, RuntimeError> {
        // Search from innermost to outermost scope
        for scope in self.scopes.iter().rev() {
            if let Some(value) = scope.variables.get(name) {
                return Ok(value.clone());
            }
        }
        
        // Check built-in variables
        match name {
            "open" => {
                if let Some(data) = &self.data {
                    let series = data.open();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'open'".to_string(),
                    })
                }
            },
            "high" => {
                if let Some(data) = &self.data {
                    let series = data.high();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'high'".to_string(),
                    })
                }
            },
            "low" => {
                if let Some(data) = &self.data {
                    let series = data.low();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'low'".to_string(),
                    })
                }
            },
            "close" => {
                if let Some(data) = &self.data {
                    let series = data.close();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'close'".to_string(),
                    })
                }
            },
            "volume" => {
                if let Some(data) = &self.data {
                    let series = data.volume();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'volume'".to_string(),
                    })
                }
            },
            "hl2" => {
                if let Some(data) = &self.data {
                    let series = data.hl2();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'hl2'".to_string(),
                    })
                }
            },
            "hlc3" => {
                if let Some(data) = &self.data {
                    let series = data.hlc3();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'hlc3'".to_string(),
                    })
                }
            },
            "ohlc4" => {
                if let Some(data) = &self.data {
                    let series = data.ohlc4();
                    Ok(Value::Series(series))
                } else {
                    Err(RuntimeError::DataError {
                        message: "No data available for 'ohlc4'".to_string(),
                    })
                }
            },
            _ => Err(RuntimeError::VariableNotFound {
                name: name.to_string(),
            }),
        }
    }
    
    /// Register a built-in function
    pub fn register_function(&mut self, name: &str, function: BuiltInFunction) {
        self.functions.insert(name.to_string(), function);
    }
    
    /// Call a function
    pub fn call_function(&mut self, name: &str, args: &[Value], named_args: &HashMap<String, Value>) -> Result<Value, RuntimeError> {
        if let Some(function) = self.functions.get(name) {
            function(self, args, named_args)
        } else {
            Err(RuntimeError::FunctionNotFound {
                name: name.to_string(),
            })
        }
    }
    
    /// Add an alert
    pub fn add_alert(&mut self, parameters: HashMap<String, Value>) {
        self.alerts.push(parameters);
    }
    
    /// Get all alerts
    pub fn get_alerts(&self) -> Vec<HashMap<String, Value>> {
        self.alerts.clone()
    }
    
    /// Add a plot
    pub fn add_plot(&mut self, value: Value, parameters: HashMap<String, Value>) {
        self.plots.push((value, parameters));
    }
    
    /// Get all plots
    pub fn get_plots(&self) -> Vec<(Value, HashMap<String, Value>)> {
        self.plots.clone()
    }
    
    /// Add a plot shape
    pub fn add_plot_shape(&mut self, value: Value, parameters: HashMap<String, Value>) {
        self.plot_shapes.push((value, parameters));
    }
    
    /// Get all plot shapes
    pub fn get_plot_shapes(&self) -> Vec<(Value, HashMap<String, Value>)> {
        self.plot_shapes.clone()
    }
    
    /// Store a series value for a variable
    pub fn store_series_value(&mut self, name: &str, value: Value) {
        if let Some(series) = self.series_values.get_mut(name) {
            series.push(value);
        } else {
            self.series_values.insert(name.to_string(), vec![value]);
        }
    }
    
    /// Get a series value for a variable
    pub fn get_series_value(&self, name: &str, index: usize) -> Option<Value> {
        if let Some(series) = self.series_values.get(name) {
            if index < series.len() {
                Some(series[index].clone())
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_environment_variables() {
        let mut env = Environment::new();
        
        // Set and get a variable
        env.set_variable("x".to_string(), Value::Integer(42));
        let value = env.get_variable("x").unwrap();
        assert!(matches!(value, Value::Integer(42)));
        
        // Test scopes
        env.push_scope();
        env.set_variable("y".to_string(), Value::Float(3.14));
        assert!(matches!(env.get_variable("y").unwrap(), Value::Float(3.14)));
        assert!(matches!(env.get_variable("x").unwrap(), Value::Integer(42))); // Can see outer scope
        
        env.pop_scope();
        assert!(env.get_variable("y").is_err()); // y is no longer in scope
        assert!(matches!(env.get_variable("x").unwrap(), Value::Integer(42))); // x is still in scope
    }
}
