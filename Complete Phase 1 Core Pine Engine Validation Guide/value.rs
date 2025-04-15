// Value module for Pine Script runtime
// Defines the value types used in the runtime

use std::collections::HashMap;
use super::error::RuntimeError;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// Represents a value in Pine Script
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Color(String),
    Array(Vec<Value>),
    Series(Vec<Value>),
    NA,
}

impl Value {
    /// Convert value to boolean
    pub fn as_bool(&self) -> Result<bool, RuntimeError> {
        match self {
            Value::Boolean(b) => Ok(*b),
            Value::Integer(i) => Ok(*i != 0),
            Value::Float(f) => Ok(*f != 0.0),
            Value::NA => Ok(false),
            _ => Err(RuntimeError::TypeError {
                expected: "boolean".to_string(),
                found: self.type_name(),
                message: "Cannot convert to boolean".to_string(),
            }),
        }
    }
    
    /// Convert value to integer
    pub fn as_integer(&self) -> Result<i64, RuntimeError> {
        match self {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            Value::Boolean(b) => Ok(if *b { 1 } else { 0 }),
            Value::NA => Err(RuntimeError::TypeError {
                expected: "integer".to_string(),
                found: "na".to_string(),
                message: "Cannot convert NA to integer".to_string(),
            }),
            _ => Err(RuntimeError::TypeError {
                expected: "integer".to_string(),
                found: self.type_name(),
                message: "Cannot convert to integer".to_string(),
            }),
        }
    }
    
    /// Convert value to float
    pub fn as_float(&self) -> Result<f64, RuntimeError> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Integer(i) => Ok(*i as f64),
            Value::Boolean(b) => Ok(if *b { 1.0 } else { 0.0 }),
            Value::NA => Err(RuntimeError::TypeError {
                expected: "float".to_string(),
                found: "na".to_string(),
                message: "Cannot convert NA to float".to_string(),
            }),
            _ => Err(RuntimeError::TypeError {
                expected: "float".to_string(),
                found: self.type_name(),
                message: "Cannot convert to float".to_string(),
            }),
        }
    }
    
    /// Convert value to string
    pub fn as_string(&self) -> Result<String, RuntimeError> {
        match self {
            Value::String(s) => Ok(s.clone()),
            Value::Integer(i) => Ok(i.to_string()),
            Value::Float(f) => Ok(f.to_string()),
            Value::Boolean(b) => Ok(b.to_string()),
            Value::Color(c) => Ok(c.clone()),
            Value::NA => Ok("na".to_string()),
            Value::Array(_) => Err(RuntimeError::TypeError {
                expected: "string".to_string(),
                found: "array".to_string(),
                message: "Cannot convert array to string".to_string(),
            }),
            Value::Series(_) => Err(RuntimeError::TypeError {
                expected: "string".to_string(),
                found: "series".to_string(),
                message: "Cannot convert series to string".to_string(),
            }),
        }
    }
    
    /// Get the type name of the value
    pub fn type_name(&self) -> String {
        match self {
            Value::Integer(_) => "integer".to_string(),
            Value::Float(_) => "float".to_string(),
            Value::Boolean(_) => "boolean".to_string(),
            Value::String(_) => "string".to_string(),
            Value::Color(_) => "color".to_string(),
            Value::Array(_) => "array".to_string(),
            Value::Series(_) => "series".to_string(),
            Value::NA => "na".to_string(),
        }
    }
    
    /// Check if the value is NA
    pub fn is_na(&self) -> bool {
        matches!(self, Value::NA)
    }
    
    /// Add operation
    pub fn add(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a + *b as f64)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(a.clone() + b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "numeric or string".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot add these types".to_string(),
            }),
        }
    }
    
    /// Subtract operation
    pub fn subtract(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a - *b as f64)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "numeric".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot subtract these types".to_string(),
            }),
        }
    }
    
    /// Multiply operation
    pub fn multiply(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(a * *b as f64)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "numeric".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot multiply these types".to_string(),
            }),
        }
    }
    
    /// Divide operation
    pub fn divide(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(*a as f64 / *b as f64))
                }
            },
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(a / b))
                }
            },
            (Value::Integer(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(*a as f64 / b))
                }
            },
            (Value::Float(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(a / *b as f64))
                }
            },
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "numeric".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot divide these types".to_string(),
            }),
        }
    }
    
    /// Modulo operation
    pub fn modulo(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Integer(a % b))
                }
            },
            (Value::Float(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(a % b))
                }
            },
            (Value::Integer(a), Value::Float(b)) => {
                if *b == 0.0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(*a as f64 % b))
                }
            },
            (Value::Float(a), Value::Integer(b)) => {
                if *b == 0 {
                    Err(RuntimeError::DivisionByZero)
                } else {
                    Ok(Value::Float(a % *b as f64))
                }
            },
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "numeric".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot perform modulo on these types".to_string(),
            }),
        }
    }
    
    /// Equal operation
    pub fn equal(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a == b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a == b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean(*a as f64 == *b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a == *b as f64)),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(a == b)),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a == b)),
            (Value::Color(a), Value::Color(b)) => Ok(Value::Boolean(a == b)),
            (Value::NA, Value::NA) => Ok(Value::Boolean(true)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::Boolean(false)),
            _ => Err(RuntimeError::TypeError {
                expected: "comparable types".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot compare these types for equality".to_string(),
            }),
        }
    }
    
    /// Not equal operation
    pub fn not_equal(&self, other: &Value) -> Result<Value, RuntimeError> {
        self.equal(other).map(|v| {
            if let Value::Boolean(b) = v {
                Value::Boolean(!b)
            } else {
                v
            }
        })
    }
    
    /// Less than operation
    pub fn less_than(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a < b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a < b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) < *b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a < (*b as f64))),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a < b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "comparable types".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot compare these types with less than".to_string(),
            }),
        }
    }
    
    /// Less than or equal operation
    pub fn less_than_or_equal(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a <= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a <= b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) <= *b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a <= (*b as f64))),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a <= b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "comparable types".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot compare these types with less than or equal".to_string(),
            }),
        }
    }
    
    /// Greater than operation
    pub fn greater_than(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a > b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a > b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) > *b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a > (*b as f64))),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a > b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "comparable types".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot compare these types with greater than".to_string(),
            }),
        }
    }
    
    /// Greater than or equal operation
    pub fn greater_than_or_equal(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Boolean(a >= b)),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Boolean(a >= b)),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Boolean((*a as f64) >= *b)),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Boolean(*a >= (*b as f64))),
            (Value::String(a), Value::String(b)) => Ok(Value::Boolean(a >= b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "comparable types".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot compare these types with greater than or equal".to_string(),
            }),
        }
    }
    
    /// And operation
    pub fn and(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a && *b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "boolean".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot perform logical AND on these types".to_string(),
            }),
        }
    }
    
    /// Or operation
    pub fn or(&self, other: &Value) -> Result<Value, RuntimeError> {
        match (self, other) {
            (Value::Boolean(a), Value::Boolean(b)) => Ok(Value::Boolean(*a || *b)),
            (Value::NA, _) | (_, Value::NA) => Ok(Value::NA),
            _ => Err(RuntimeError::TypeError {
                expected: "boolean".to_string(),
                found: format!("{} and {}", self.type_name(), other.type_name()),
                message: "Cannot perform logical OR on these types".to_string(),
            }),
        }
    }
    
    /// Negate operation
    pub fn negate(&self) -> Result<Value, RuntimeError> {
        match self {
            Value::Integer(i) => Ok(Value::Integer(-i)),
            Value::Float(f) => Ok(Value::Float(-f)),
            Value::NA => Ok(Value::NA),
            _ =>
(Content truncated due to size limit. Use line ranges to read in chunks)