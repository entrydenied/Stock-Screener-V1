// Grammar parser module for Pine Script
// Responsible for parsing tokens into an AST

use super::ast::{Script, Declaration, Statement, Expression, Literal, BinaryOperator, UnaryOperator};
use super::error::ParseError;
use super::lexer::{Token, TokenKind};
use std::collections::HashMap;

/// Parse tokens into an AST
pub fn parse_tokens(tokens: Vec<Token>) -> Result<Script, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse()
}

struct Parser {
    tokens: Vec<Token>,
    current: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser {
            tokens,
            current: 0,
        }
    }
    
    /// Parse the tokens into a Script AST
    fn parse(&mut self) -> Result<Script, ParseError> {
        let mut script = Script::new(String::from("5")); // Default to version 5
        
        // Parse version comment if present
        if let Some(token) = self.peek() {
            if token.kind == TokenKind::Comment && token.value.contains("@version=") {
                let version = token.value
                    .split('@')
                    .nth(1)
                    .and_then(|s| s.split('=').nth(1))
                    .unwrap_or("5")
                    .trim();
                script.version = version.to_string();
                self.advance();
            }
        }
        
        // Parse declarations and statements
        while !self.is_at_end() {
            if self.match_token(TokenKind::Indicator) {
                if let Ok(declaration) = self.parse_indicator_declaration() {
                    script.add_declaration(declaration);
                }
            } else {
                if let Ok(statement) = self.parse_statement() {
                    script.add_statement(statement);
                }
            }
            
            // Skip any errors and continue parsing
            while !self.is_at_end() && !self.check(TokenKind::Newline) && !self.check(TokenKind::EOF) {
                self.advance();
            }
            
            // Skip newlines
            while self.match_token(TokenKind::Newline) {}
        }
        
        Ok(script)
    }
    
    /// Parse an indicator declaration
    fn parse_indicator_declaration(&mut self) -> Result<Declaration, ParseError> {
        // Expect left parenthesis
        self.consume(TokenKind::LeftParen, "Expected '(' after 'indicator'")?;
        
        // Parse indicator name (string literal)
        let name_token = self.consume(TokenKind::String, "Expected string literal for indicator name")?;
        let name = name_token.value.clone();
        
        // Parse parameters
        let mut parameters = HashMap::new();
        
        if self.match_token(TokenKind::Comma) {
            loop {
                if self.check(TokenKind::RightParen) {
                    break;
                }
                
                // Parse parameter name
                let param_name_token = self.consume(TokenKind::Identifier, "Expected parameter name")?;
                let param_name = param_name_token.value.clone();
                
                // Expect equals sign
                self.consume(TokenKind::Assignment, "Expected '=' after parameter name")?;
                
                // Parse parameter value
                let param_value = self.parse_expression(0)?;
                
                // Add parameter to map
                parameters.insert(param_name, param_value);
                
                // Check for comma
                if !self.match_token(TokenKind::Comma) {
                    break;
                }
            }
        }
        
        // Expect right parenthesis
        self.consume(TokenKind::RightParen, "Expected ')' after indicator parameters")?;
        
        Ok(Declaration::Indicator { name, parameters })
    }
    
    /// Parse a statement
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        if self.match_token(TokenKind::If) {
            self.parse_if_statement()
        } else if self.match_token(TokenKind::LeftBrace) {
            self.parse_block_statement()
        } else if self.check(TokenKind::Identifier) && self.peek_next().map_or(false, |t| t.kind == TokenKind::Assignment) {
            self.parse_assignment_statement()
        } else {
            // Try to parse as an expression statement (function call, etc.)
            let expr = self.parse_expression(0)?;
            
            // Check if it's an alertcondition call
            if let Expression::FunctionCall { name, .. } = &expr {
                if name == "alertcondition" {
                    // Convert to AlertCondition statement
                    if let Expression::FunctionCall { name: _, arguments, named_arguments } = expr {
                        if arguments.is_empty() {
                            return Err(ParseError::SyntaxError {
                                line: self.previous().line,
                                column: self.previous().column,
                                message: "alertcondition requires at least one argument".to_string(),
                            });
                        }
                        
                        let condition = arguments[0].clone();
                        let mut params = HashMap::new();
                        
                        // Add remaining positional arguments as parameters
                        for (i, arg) in arguments.iter().skip(1).enumerate() {
                            params.insert(format!("arg{}", i), arg.clone());
                        }
                        
                        // Add named arguments
                        for (name, value) in named_arguments {
                            params.insert(name, value);
                        }
                        
                        return Ok(Statement::AlertCondition {
                            condition,
                            parameters: params,
                        });
                    }
                } else if name == "plot" {
                    // Convert to Plot statement
                    if let Expression::FunctionCall { name: _, arguments, named_arguments } = expr {
                        if arguments.is_empty() {
                            return Err(ParseError::SyntaxError {
                                line: self.previous().line,
                                column: self.previous().column,
                                message: "plot requires at least one argument".to_string(),
                            });
                        }
                        
                        let expression = arguments[0].clone();
                        let mut params = HashMap::new();
                        
                        // Add remaining positional arguments as parameters
                        for (i, arg) in arguments.iter().skip(1).enumerate() {
                            params.insert(format!("arg{}", i), arg.clone());
                        }
                        
                        // Add named arguments
                        for (name, value) in named_arguments {
                            params.insert(name, value);
                        }
                        
                        return Ok(Statement::Plot {
                            expression,
                            parameters: params,
                        });
                    }
                } else if name == "plotshape" {
                    // Convert to PlotShape statement
                    if let Expression::FunctionCall { name: _, arguments, named_arguments } = expr {
                        if arguments.is_empty() {
                            return Err(ParseError::SyntaxError {
                                line: self.previous().line,
                                column: self.previous().column,
                                message: "plotshape requires at least one argument".to_string(),
                            });
                        }
                        
                        let expression = arguments[0].clone();
                        let mut params = HashMap::new();
                        
                        // Add remaining positional arguments as parameters
                        for (i, arg) in arguments.iter().skip(1).enumerate() {
                            params.insert(format!("arg{}", i), arg.clone());
                        }
                        
                        // Add named arguments
                        for (name, value) in named_arguments {
                            params.insert(name, value);
                        }
                        
                        return Ok(Statement::PlotShape {
                            expression,
                            parameters: params,
                        });
                    }
                }
            }
            
            // Otherwise, it's a function call statement
            if let Expression::FunctionCall { name, arguments, named_arguments } = expr {
                Ok(Statement::FunctionCall {
                    name,
                    arguments,
                    named_arguments,
                })
            } else {
                Err(ParseError::SyntaxError {
                    line: self.previous().line,
                    column: self.previous().column,
                    message: "Expected statement".to_string(),
                })
            }
        }
    }
    
    /// Parse an if statement
    fn parse_if_statement(&mut self) -> Result<Statement, ParseError> {
        // Parse condition
        self.consume(TokenKind::LeftParen, "Expected '(' after 'if'")?;
        let condition = self.parse_expression(0)?;
        self.consume(TokenKind::RightParen, "Expected ')' after if condition")?;
        
        // Parse then branch
        let then_branch = Box::new(self.parse_statement()?);
        
        // Parse else branch if present
        let else_branch = if self.match_token(TokenKind::Else) {
            Some(Box::new(self.parse_statement()?))
        } else {
            None
        };
        
        Ok(Statement::Conditional {
            condition,
            then_branch,
            else_branch,
        })
    }
    
    /// Parse a block statement
    fn parse_block_statement(&mut self) -> Result<Statement, ParseError> {
        let mut statements = Vec::new();
        
        // Skip newlines
        while self.match_token(TokenKind::Newline) {}
        
        while !self.check(TokenKind::RightBrace) && !self.is_at_end() {
            statements.push(self.parse_statement()?);
            
            // Skip newlines
            while self.match_token(TokenKind::Newline) {}
        }
        
        self.consume(TokenKind::RightBrace, "Expected '}' after block")?;
        
        Ok(Statement::Block { statements })
    }
    
    /// Parse an assignment statement
    fn parse_assignment_statement(&mut self) -> Result<Statement, ParseError> {
        let variable_token = self.consume(TokenKind::Identifier, "Expected variable name")?;
        let variable = variable_token.value.clone();
        
        self.consume(TokenKind::Assignment, "Expected '=' after variable name")?;
        
        let value = self.parse_expression(0)?;
        
        Ok(Statement::Assignment { variable, value })
    }
    
    /// Parse an expression with precedence climbing
    fn parse_expression(&mut self, precedence: u8) -> Result<Expression, ParseError> {
        let mut left = self.parse_primary()?;
        
        while !self.is_at_end() {
            let op_precedence = self.get_operator_precedence();
            
            if op_precedence <= precedence {
                break;
            }
            
            let operator = self.parse_binary_operator()?;
            let right = self.parse_expression(op_precedence)?;
            
            left = Expression::BinaryOperation {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }
    
    /// Parse a primary expression (literal, variable, function call, etc.)
    fn parse_primary(&mut self) -> Result<Expression, ParseError> {
        if self.match_token(TokenKind::Integer) {
            let value = self.previous().value.parse::<i64>().map_err(|_| {
                ParseError::SyntaxError {
                    line: self.previous().line,
                    column: self.previous().column,
                    message: "Invalid integer literal".to_string(),
                }
            })?;
            
            Ok(Expression::Literal(Literal::Integer(value)))
        } else if self.match_token(TokenKind::Float) {
            let value = self.previous().value.parse::<f64>().map_err(|_| {
                ParseError::SyntaxError {
                    line: self.previous().line,
                    column: self.previous().column,
                    message: "Invalid float literal".to_string(),
                }
            })?;
            
            Ok(Expression::Literal(Literal::Float(value)))
        } else if self.match_token(TokenKind::String) {
            let value = self.previous().value.clone();
            Ok(Expression::Literal(Literal::String(value)))
        } else if self.match_token(TokenKind::Boolean) {
            let value = self.previous().value == "true";
            Ok(Expression::Literal(Literal::Boolean(value)))
        } else if self.match_token(TokenKind::Identifier) {
            let name = self.previous().value.clone();
            
            // Check if it's a function call
            if self.match_token(TokenKind::LeftParen) {
                self.parse_function_call(name)
            } else if self.match_token(TokenKind::LeftBracket) {
                // Array access
                let index = self.parse_expression(0)?;
                self.consume(TokenKind::RightBracket, "Expected ']' after array index")?;
                
                Ok(Expression::ArrayAccess {
                    array: Box::new(Expression::Variable(name)),
                    index: Box::new(index),
                })
            } else {
                // Simple variable
                Ok(Expression::Variable(name))
            }
        } else if self.match_token(TokenKind::LeftParen) {
            // Grouping
            let expr = self.parse_expression(0)?;
            self.consume(TokenKind::RightParen, "Expected ')' after expression")?;
            Ok(expr)
        } else if self.match_token(TokenKind::Minus) || self.match_token(TokenKind::Not) {
            // Unary operation
            let operator = if self.previous().kind == TokenKind::Minus {
                UnaryOperator::Negate
            } else {
                UnaryOperator::Not
            };
            
            let operand = self.parse_expression(7)?; // Precedence of unary operators
            
            Ok(Expression::UnaryOperation {
                operator,
                operand: Box::new(operand),
            })
        } else {
            Err(ParseError::SyntaxError {
                line: self.peek().map_or(0, |t| t.line),
                column: self.peek().map_or(0, |t| t.column),
                message: "Expected expression".to_string(),
            })
        }
    }
    
    /// Parse a function call
    fn parse_function_call(&mut self, name: String) -> Result<Expression, ParseError> {
        let mut arguments = Vec::new();
        let mut named_arguments = HashMap::new();
        
        // Parse arguments
        if !self.check(TokenKind::RightParen) {
            loop {
                // Check if it's a named argument
                if self.check(TokenKind::Identifier) && self.peek_next().map_or(false, |t| t.kind == TokenKind::Assignme
(Content truncated due to size limit. Use line ranges to read in chunks)