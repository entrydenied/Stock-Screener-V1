// Integration test for the Pine Engine
// This file integrates the parser and runtime to test the full engine functionality

use pine_engine::parser;
use pine_engine::runtime::{Runtime, data::OHLCVData, data::Candle};
use chrono::{TimeZone, Utc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Pine Engine Integration Test");
    println!("===========================");
    
    // Create test data
    let mut data = create_test_data();
    println!("Created test data with {} candles", data.len());
    
    // Load the test script
    let script_path = "examples/mashume_hull_tv.pine";
    let script = std::fs::read_to_string(script_path)?;
    println!("Loaded script from {}", script_path);
    
    // Parse the script
    let ast = parser::parse(&script)?;
    println!("Successfully parsed the script");
    
    // Create runtime and execute
    let mut runtime = Runtime::new();
    let result = runtime.execute(&ast, &data)?;
    
    // Print results
    println!("\nExecution Results:");
    println!("-----------------");
    println!("Alerts: {}", result.alert_count());
    println!("Plots: {}", result.plot_count());
    
    if result.has_alerts() {
        println!("\nAlerts:");
        for (i, alert) in result.alerts.iter().enumerate() {
            let title = alert.get("title").map_or("", |v| match v {
                pine_engine::runtime::value::Value::String(s) => s,
                _ => "",
            });
            
            let message = alert.get("message").map_or("", |v| match v {
                pine_engine::runtime::value::Value::String(s) => s,
                _ => "",
            });
            
            println!("  {}. {}: {}", i + 1, title, message);
        }
    }
    
    println!("\nIntegration test completed successfully!");
    Ok(())
}

// Create test data for the integration test
fn create_test_data() -> OHLCVData {
    let mut data = OHLCVData::with_symbol_and_timeframe("TEST", "1d");
    
    // Add 100 candles with a simple pattern
    for i in 0..100 {
        let base_price = 100.0 + (i as f64 / 10.0);
        let volatility = 2.0;
        
        let open = base_price;
        let high = base_price + volatility * (i % 3) as f64;
        let low = base_price - volatility * (i % 2) as f64;
        let close = base_price + volatility * ((i % 5) as f64 - 2.0);
        let volume = 1000.0 + (i % 10) as f64 * 100.0;
        
        let timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap() 
            + chrono::Duration::days(i as i64);
        
        data.add_candle(Candle {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        });
    }
    
    data
}
