// Lexer module for Pine Script
// Responsible for tokenizing Pine Script into tokens

use super::error::ParseError;

/// Represents a token in Pine Script
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub value: String,
    pub line: usize,
    pub column: usize,
}

/// Represents the kind of token
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Version,
    Indicator,
    Strategy,
    Study,
    Input,
    If,
    Else,
    For,
    To,
    By,
    While,
    Var,
    
    // Literals
    Integer,
    Float,
    String,
    Boolean,
    Color,
    
    // Identifiers
    Identifier,
    
    // Operators
    Plus,
    Minus,
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
    Not,
    Assignment,
    
    // Delimiters
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    LeftBrace,
    RightBrace,
    Comma,
    Dot,
    Colon,
    QuestionMark,
    
    // Special
    Comment,
    Newline,
    Whitespace,
    EOF,
}

/// Tokenize a Pine Script string into a vector of tokens
pub fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();
    let mut line = 1;
    let mut column = 1;
    
    while let Some(&c) = chars.peek() {
        match c {
            // Skip whitespace but track it for position
            ' ' | '\t' => {
                chars.next();
                column += 1;
            },
            
            // Track newlines
            '\n' | '\r' => {
                chars.next();
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                line += 1;
                column = 1;
            },
            
            // Comments
            '/' => {
                chars.next();
                column += 1;
                
                if let Some(&next_char) = chars.peek() {
                    if next_char == '/' {
                        // Line comment
                        chars.next();
                        column += 1;
                        let mut comment = String::from("//");
                        
                        while let Some(&ch) = chars.peek() {
                            if ch == '\n' || ch == '\r' {
                                break;
                            }
                            comment.push(ch);
                            chars.next();
                            column += 1;
                        }
                        
                        tokens.push(Token {
                            kind: TokenKind::Comment,
                            value: comment,
                            line,
                            column: column - comment.len(),
                        });
                    } else {
                        // Division operator
                        tokens.push(Token {
                            kind: TokenKind::Divide,
                            value: String::from("/"),
                            line,
                            column: column - 1,
                        });
                    }
                }
            },
            
            // String literals
            '\'' | '"' => {
                let quote = c;
                chars.next();
                column += 1;
                
                let mut string_value = String::new();
                let start_column = column;
                
                while let Some(&ch) = chars.peek() {
                    if ch == quote {
                        chars.next();
                        column += 1;
                        break;
                    } else if ch == '\\' {
                        // Handle escape sequences
                        chars.next();
                        column += 1;
                        
                        if let Some(&escaped_char) = chars.peek() {
                            match escaped_char {
                                'n' => string_value.push('\n'),
                                't' => string_value.push('\t'),
                                'r' => string_value.push('\r'),
                                '\\' => string_value.push('\\'),
                                '\'' => string_value.push('\''),
                                '"' => string_value.push('"'),
                                _ => {
                                    return Err(ParseError::LexicalError {
                                        line,
                                        column,
                                        message: format!("Invalid escape sequence: \\{}", escaped_char),
                                    });
                                }
                            }
                            chars.next();
                            column += 1;
                        }
                    } else if ch == '\n' || ch == '\r' {
                        return Err(ParseError::LexicalError {
                            line,
                            column,
                            message: "Unterminated string literal".to_string(),
                        });
                    } else {
                        string_value.push(ch);
                        chars.next();
                        column += 1;
                    }
                }
                
                tokens.push(Token {
                    kind: TokenKind::String,
                    value: string_value,
                    line,
                    column: start_column,
                });
            },
            
            // Numbers (integer and float)
            '0'..='9' => {
                let mut number = String::new();
                let start_column = column;
                let mut is_float = false;
                
                while let Some(&ch) = chars.peek() {
                    if ch.is_digit(10) {
                        number.push(ch);
                        chars.next();
                        column += 1;
                    } else if ch == '.' {
                        if is_float {
                            return Err(ParseError::LexicalError {
                                line,
                                column,
                                message: "Invalid number format: multiple decimal points".to_string(),
                            });
                        }
                        is_float = true;
                        number.push(ch);
                        chars.next();
                        column += 1;
                    } else {
                        break;
                    }
                }
                
                tokens.push(Token {
                    kind: if is_float { TokenKind::Float } else { TokenKind::Integer },
                    value: number,
                    line,
                    column: start_column,
                });
            },
            
            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut identifier = String::new();
                let start_column = column;
                
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' || ch == '.' {
                        identifier.push(ch);
                        chars.next();
                        column += 1;
                    } else {
                        break;
                    }
                }
                
                // Check if it's a keyword
                let kind = match identifier.as_str() {
                    "version" => TokenKind::Version,
                    "indicator" => TokenKind::Indicator,
                    "strategy" => TokenKind::Strategy,
                    "study" => TokenKind::Study,
                    "input" => TokenKind::Input,
                    "if" => TokenKind::If,
                    "else" => TokenKind::Else,
                    "for" => TokenKind::For,
                    "to" => TokenKind::To,
                    "by" => TokenKind::By,
                    "while" => TokenKind::While,
                    "var" => TokenKind::Var,
                    "true" | "false" => TokenKind::Boolean,
                    _ => TokenKind::Identifier,
                };
                
                tokens.push(Token {
                    kind,
                    value: identifier,
                    line,
                    column: start_column,
                });
            },
            
            // Operators and delimiters
            '+' => {
                tokens.push(Token {
                    kind: TokenKind::Plus,
                    value: String::from("+"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '-' => {
                tokens.push(Token {
                    kind: TokenKind::Minus,
                    value: String::from("-"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '*' => {
                tokens.push(Token {
                    kind: TokenKind::Multiply,
                    value: String::from("*"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '%' => {
                tokens.push(Token {
                    kind: TokenKind::Modulo,
                    value: String::from("%"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '=' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::Equal,
                        value: String::from("=="),
                        line,
                        column: column - 2,
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Assignment,
                        value: String::from("="),
                        line,
                        column: column - 1,
                    });
                }
            },
            '!' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::NotEqual,
                        value: String::from("!="),
                        line,
                        column: column - 2,
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Not,
                        value: String::from("!"),
                        line,
                        column: column - 1,
                    });
                }
            },
            '<' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::LessThanOrEqual,
                        value: String::from("<="),
                        line,
                        column: column - 2,
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::LessThan,
                        value: String::from("<"),
                        line,
                        column: column - 1,
                    });
                }
            },
            '>' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'=') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::GreaterThanOrEqual,
                        value: String::from(">="),
                        line,
                        column: column - 2,
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::GreaterThan,
                        value: String::from(">"),
                        line,
                        column: column - 1,
                    });
                }
            },
            '&' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'&') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::And,
                        value: String::from("&&"),
                        line,
                        column: column - 2,
                    });
                } else {
                    return Err(ParseError::LexicalError {
                        line,
                        column: column - 1,
                        message: "Expected '&' after '&'".to_string(),
                    });
                }
            },
            '|' => {
                chars.next();
                column += 1;
                
                if chars.peek() == Some(&'|') {
                    chars.next();
                    column += 1;
                    tokens.push(Token {
                        kind: TokenKind::Or,
                        value: String::from("||"),
                        line,
                        column: column - 2,
                    });
                } else {
                    return Err(ParseError::LexicalError {
                        line,
                        column: column - 1,
                        message: "Expected '|' after '|'".to_string(),
                    });
                }
            },
            '(' => {
                tokens.push(Token {
                    kind: TokenKind::LeftParen,
                    value: String::from("("),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            ')' => {
                tokens.push(Token {
                    kind: TokenKind::RightParen,
                    value: String::from(")"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '[' => {
                tokens.push(Token {
                    kind: TokenKind::LeftBracket,
                    value: String::from("["),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            ']' => {
                tokens.push(Token {
                    kind: TokenKind::RightBracket,
                    value: String::from("]"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '{' => {
                tokens.push(Token {
                    kind: TokenKind::LeftBrace,
                    value: String::from("{"),
                    line,
                    column,
                });
                chars.next();
                column += 1;
            },
            '}' => {
                tokens.push(Tok
(Content truncated due to size limit. Use line ranges to read in chunks)