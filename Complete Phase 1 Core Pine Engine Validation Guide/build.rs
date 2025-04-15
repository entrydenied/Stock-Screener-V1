// Build script for Pine Engine
// This script generates the gRPC code from the proto file

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get the output directory from Cargo
    let out_dir = env::var("OUT_DIR").unwrap();
    let proto_path = Path::new(&out_dir).join("pine_engine.proto");
    
    // Write the proto file
    let mut file = File::create(&proto_path)?;
    file.write_all(include_str!("src/api/pine_engine.proto").as_bytes())?;
    
    // Configure tonic-build to generate code from the proto file
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&out_dir)
        .compile(&[proto_path], &[out_dir.clone()])?;
    
    println!("cargo:rerun-if-changed=src/api/pine_engine.proto");
    
    Ok(())
}
