# Pine Engine

A Pine Script execution engine for trading strategy validation that replicates TradingView's Pine Script functionality.

## Overview

The Pine Engine is a Rust implementation of a Pine Script interpreter that aims to match TradingView's Pine Script behavior with high precision. This engine is designed for the MVP subset of Pine Script functionality, focusing on core features needed for basic technical analysis and alerting.

## Features

- Pine Script parser that converts Pine Script code to an Abstract Syntax Tree (AST)
- Runtime interpreter that executes the AST with market data
- Implementation of key technical analysis functions:
  - Hull Moving Average (ta.hma)
  - Crossover detection (ta.crossover)
  - Crossunder detection (ta.crossunder)
  - Value tracking (ta.valuewhen)
- Validation framework to compare results with TradingView
- Support for alertcondition functionality

## Project Structure

```
pine_engine/
├── src/
│   ├── parser/           # Pine Script parser
│   │   ├── ast.rs        # Abstract Syntax Tree definitions
│   │   ├── error.rs      # Parser error types
│   │   ├── grammar.rs    # Grammar parser
│   │   ├── lexer.rs      # Lexical analyzer
│   │   └── mod.rs        # Parser module entry point
│   ├── runtime/          # Pine Script runtime
│   │   ├── data.rs       # OHLCV data structures
│   │   ├── environment.rs # Execution environment
│   │   ├── error.rs      # Runtime error types
│   │   ├── execution.rs  # Execution result handling
│   │   ├── functions/    # Built-in functions
│   │   │   ├── ta.rs     # Technical analysis functions
│   │   │   ├── math.rs   # Math functions
│   │   │   ├── utility.rs # Utility functions
│   │   │   └── mod.rs    # Functions module entry point
│   │   ├── value.rs      # Value types and operations
│   │   └── mod.rs        # Runtime module entry point
│   ├── validation/       # Validation framework
│   │   ├── comparison.rs # Result comparison
│   │   ├── data_fetcher.rs # Market data fetching
│   │   ├── test_harness.rs # Test harness for TradingView comparison
│   │   └── mod.rs        # Validation module entry point
│   └── lib.rs            # Library entry point
├── examples/             # Example scripts and tests
│   ├── mashume_hull_tv.pine # MVP subset script
│   ├── test_engine.rs    # Test program
│   └── integration_test.rs # Integration test
├── tests/                # Test suite
├── Cargo.toml            # Project configuration
└── README.md             # This file
```

## Getting Started

### Prerequisites

- Rust 1.70.0 or later
- Cargo package manager

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-org/pine-engine.git
   cd pine-engine
   ```

2. Build the project:
   ```
   cargo build --release
   ```

### Usage

#### Basic Usage

```rust
use pine_engine::PineEngine;
use pine_engine::runtime::data::OHLCVData;

// Create a Pine Engine instance
let engine = PineEngine::new();

// Parse a Pine Script
let script = r#"
//@version=5
indicator('Simple MA', overlay=true)
src = close
len = input(14)
ma = ta.sma(src, len)
plot(ma)
"#;

let ast = engine.parse(script)?;

// Create or load OHLCV data
let data = OHLCVData::with_symbol_and_timeframe("AAPL", "1d");
// ... add candles to data ...

// Execute the script
let result = engine.execute(&ast, &data)?;

// Process the results
for plot in result.plots {
    // ... handle plot data ...
}
```

#### Validation Against TradingView

```rust
use pine_engine::validation;

// Validate a script against TradingView
let validation_result = validation::validate(script, "AAPL", "1d").await?;

// Check the results
println!("Alert match: {:.2}%", validation_result.comparison_result.alert_match_percentage);
println!("Plot match: {:.2}%", validation_result.comparison_result.plot_match_percentage);
```

## Running Tests

```
cargo test
```

## Running Examples

```
cargo run --example integration_test
cargo run --example test_engine
```

## Limitations

The current implementation focuses on the MVP subset of Pine Script functionality:
- Only supports a limited set of technical analysis functions
- Limited support for complex Pine Script features
- Focused on alertcondition functionality for screening

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TradingView for creating the Pine Script language
- The Rust community for providing excellent libraries and tools
