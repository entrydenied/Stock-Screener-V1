# Pine Engine Implementation Todo List

## Phase 1: Core Pine Engine & Validation

### Setup Development Environment
- [x] Create project directory structure
- [x] Install Rust and Cargo
- [x] Initialize Rust library project
- [x] Configure Cargo.toml with necessary dependencies
- [x] Set up project structure for parser and runtime

### Implement Core Pine Engine
- [x] Implement Pine Script parser
  - [x] Create lexer for tokenizing Pine Script
  - [x] Create parser for generating AST
  - [x] Support MVP subset syntax
- [x] Implement basic runtime
  - [x] Implement basic types (float, integer, boolean, string)
  - [x] Implement operators (+, -, *, /, etc.)
  - [x] Implement variable assignment and scoping
- [x] Implement required built-in functions
  - [x] ta.hma (Hull Moving Average)
  - [x] ta.crossover
  - [x] ta.crossunder
  - [x] ta.valuewhen
  - [x] Other necessary functions for MVP

### Develop Validation Framework
- [x] Set up data fetching mechanism for historical OHLCV data
- [x] Create test harness for comparing outputs
- [x] Implement automated comparison of outputs

### Validate Engine
- [x] Integrate parser and runtime
- [x] Test engine against MVP test scripts
- [x] Expose engine via gRPC interface

### Documentation
- [x] Document code with comments
- [x] Create README with setup and usage instructions
- [x] Document API and interfaces
