// Server binary for Pine Engine
// This program starts a gRPC server for the Pine Engine

use pine_engine::api;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the address from command line arguments or use default
    let addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "0.0.0.0:50051".to_string());
    
    println!("Pine Engine Server");
    println!("=================");
    println!("Starting gRPC server on {}", addr);
    
    // Start the server
    api::start_server(&addr).await?;
    
    Ok(())
}
