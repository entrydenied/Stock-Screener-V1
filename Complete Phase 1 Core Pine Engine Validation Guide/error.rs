// Error module for the parser
// Defines error types that can occur during parsing

use std::fmt;

/// Represents errors that can occur during parsing
#[derive(Debug)]
pub enum ParseError {
    /// Lexical error (tokenization failed)
    LexicalError {
        line: usize,
        column: usize,
        message: String,
    },
    /// Syntax error (parsing failed)
    SyntaxError {
        line: usize,
        column: usize,
        message: String,
    },
    /// Semantic error (type checking, etc.)
    SemanticError {
        line: usize,
        column: usize,
        message: String,
    },
    /// Unsupported feature
    UnsupportedFeature {
        feature: String,
        message: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::LexicalError { line, column, message } => {
                write!(f, "Lexical error at line {}, column {}: {}", line, column, message)
            }
            ParseError::SyntaxError { line, column, message } => {
                write!(f, "Syntax error at line {}, column {}: {}", line, column, message)
            }
            ParseError::SemanticError { line, column, message } => {
                write!(f, "Semantic error at line {}, column {}: {}", line, column, message)
            }
            ParseError::UnsupportedFeature { feature, message } => {
                write!(f, "Unsupported feature '{}': {}", feature, message)
            }
        }
    }
}

impl std::error::Error for ParseError {}
