// Test program for the Pine Engine
// This program tests the Pine Engine with the MVP subset script

use pine_engine::{PineEngine, validation};
use std::env;
use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pine Engine Test");
    println!("===============");
    
    // Load the test script
    let script_path = "examples/mashume_hull_tv.pine";
    let script = fs::read_to_string(script_path)?;
    
    println!("Loaded script from {}", script_path);
    
    // Create a Pine Engine instance
    let engine = PineEngine::new();
    println!("Created Pine Engine instance");
    
    // Parse the script
    let ast = engine.parse(&script)?;
    println!("Successfully parsed the script");
    
    // Test with a sample symbol
    let symbol = "AAPL";
    let timeframe = "1d";
    
    println!("Testing with symbol {} and timeframe {}", symbol, timeframe);
    
    // Validate the script against TradingView
    match validation::validate(&script, symbol, timeframe).await {
        Ok(result) => {
            println!("\nValidation Results:");
            println!("-------------------");
            println!("Alert match: {:.2}%", result.comparison_result.alert_match_percentage);
            println!("Plot match: {:.2}%", result.comparison_result.plot_match_percentage);
            
            if result.comparison_result.mismatches.is_empty() {
                println!("No mismatches found! The Pine Engine implementation matches TradingView.");
            } else {
                println!("\nMismatches found:");
                for mismatch in &result.comparison_result.mismatches {
                    println!("- {:?}", mismatch);
                }
            }
        },
        Err(e) => {
            if let validation::ValidationError::TradingViewApiError(msg) = &e {
                if msg.contains("Manual testing required") {
                    println!("\nManual testing required:");
                    println!("------------------------");
                    println!("{}", msg);
                    println!("\nPlease follow the instructions in the generated file to complete the validation.");
                } else {
                    println!("Error validating against TradingView: {}", e);
                }
            } else {
                println!("Error validating against TradingView: {}", e);
            }
        }
    }
    
    Ok(())
}
