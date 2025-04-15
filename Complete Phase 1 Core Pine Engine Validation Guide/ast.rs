// AST (Abstract Syntax Tree) module for Pine Script
// Defines the structures that represent the parsed Pine Script

use std::collections::HashMap;

/// Represents a complete Pine Script
#[derive(Debug, Clone)]
pub struct Script {
    pub version: String,
    pub declarations: Vec<Declaration>,
    pub statements: Vec<Statement>,
}

/// Represents a declaration in Pine Script (e.g., indicator, strategy)
#[derive(Debug, Clone)]
pub enum Declaration {
    Indicator {
        name: String,
        parameters: HashMap<String, Expression>,
    },
    Strategy {
        name: String,
        parameters: HashMap<String, Expression>,
    },
    Study {
        name: String,
        parameters: HashMap<String, Expression>,
    },
}

/// Represents a statement in Pine Script
#[derive(Debug, Clone)]
pub enum Statement {
    Assignment {
        variable: String,
        value: Expression,
    },
    Conditional {
        condition: Expression,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
    },
    Block {
        statements: Vec<Statement>,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
        named_arguments: HashMap<String, Expression>,
    },
    Plot {
        expression: Expression,
        parameters: HashMap<String, Expression>,
    },
    PlotShape {
        expression: Expression,
        parameters: HashMap<String, Expression>,
    },
    AlertCondition {
        condition: Expression,
        parameters: HashMap<String, Expression>,
    },
}

/// Represents an expression in Pine Script
#[derive(Debug, Clone)]
pub enum Expression {
    Literal(Literal),
    Variable(String),
    BinaryOperation {
        left: Box<Expression>,
        operator: BinaryOperator,
        right: Box<Expression>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<Expression>,
    },
    FunctionCall {
        name: String,
        arguments: Vec<Expression>,
        named_arguments: HashMap<String, Expression>,
    },
    ArrayAccess {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    TernaryOperation {
        condition: Box<Expression>,
        then_branch: Box<Expression>,
        else_branch: Box<Expression>,
    },
}

/// Represents a literal value in Pine Script
#[derive(Debug, Clone)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Color(String),
    Array(Vec<Expression>),
}

/// Represents a binary operator in Pine Script
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
}

/// Represents a unary operator in Pine Script
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
}

impl Script {
    /// Create a new empty script
    pub fn new(version: String) -> Self {
        Script {
            version,
            declarations: Vec::new(),
            statements: Vec::new(),
        }
    }
    
    /// Add a declaration to the script
    pub fn add_declaration(&mut self, declaration: Declaration) {
        self.declarations.push(declaration);
    }
    
    /// Add a statement to the script
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
    }
}
