// Parser module for Pine Script
// Responsible for lexical analysis and parsing Pine Script into an AST

pub mod ast;
pub mod error;
pub mod grammar;
pub mod lexer;

use error::ParseError;
use ast::Script;

/// Parse a Pine Script string into an Abstract Syntax Tree (AST)
pub fn parse(input: &str) -> Result<Script, ParseError> {
    // First tokenize the input using the lexer
    let tokens = lexer::tokenize(input)?;
    
    // Then parse the tokens into an AST
    let ast = grammar::parse_tokens(tokens)?;
    
    Ok(ast)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_parsing() {
        let script = r#"//@version=5
        indicator('Test', overlay=true)
        a = 1 + 2
        "#;
        
        let result = parse(script);
        assert!(result.is_ok());
    }
}
