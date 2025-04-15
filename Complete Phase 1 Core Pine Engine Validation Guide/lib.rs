// Pine Engine - A Pine Script execution engine for trading strategy validation
// Main library entry point

pub mod parser;
pub mod runtime;
pub mod functions;
pub mod validation;
pub mod api;

/// Pine Engine version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Main entry point for the Pine Engine
pub struct PineEngine {
    // Configuration and state will be added here
}

impl PineEngine {
    /// Create a new Pine Engine instance
    pub fn new() -> Self {
        PineEngine {}
    }
    
    /// Parse a Pine Script and return an AST
    pub fn parse(&self, script: &str) -> Result<parser::ast::Script, parser::error::ParseError> {
        parser::parse(script)
    }
    
    /// Execute a parsed script with the given input data
    pub fn execute(
        &self, 
        script: &parser::ast::Script, 
        data: &runtime::data::OHLCVData
    ) -> Result<runtime::execution::ExecutionResult, runtime::error::RuntimeError> {
        let mut runtime = runtime::Runtime::new();
        runtime.execute(script, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_creation() {
        let engine = PineEngine::new();
        assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
    }
}
