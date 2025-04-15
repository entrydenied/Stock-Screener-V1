// Data module for Pine Script runtime
// Defines the data structures for OHLCV data

use chrono::{DateTime, Utc};
use super::error::RuntimeError;
use super::value::Value;

/// Represents a single OHLCV candle
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Represents OHLCV data for a symbol
#[derive(Debug, Clone)]
pub struct OHLCVData {
    pub symbol: String,
    pub timeframe: String,
    pub candles: Vec<Candle>,
}

impl OHLCVData {
    /// Create a new empty OHLCV data
    pub fn new() -> Self {
        OHLCVData {
            symbol: String::new(),
            timeframe: String::new(),
            candles: Vec::new(),
        }
    }
    
    /// Create a new OHLCV data with the given symbol and timeframe
    pub fn with_symbol_and_timeframe(symbol: &str, timeframe: &str) -> Self {
        OHLCVData {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            candles: Vec::new(),
        }
    }
    
    /// Add a candle to the data
    pub fn add_candle(&mut self, candle: Candle) {
        self.candles.push(candle);
    }
    
    /// Get the number of candles
    pub fn len(&self) -> usize {
        self.candles.len()
    }
    
    /// Check if the data is empty
    pub fn is_empty(&self) -> bool {
        self.candles.is_empty()
    }
    
    /// Get the open prices as a series
    pub fn open(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float(c.open)).collect()
    }
    
    /// Get the high prices as a series
    pub fn high(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float(c.high)).collect()
    }
    
    /// Get the low prices as a series
    pub fn low(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float(c.low)).collect()
    }
    
    /// Get the close prices as a series
    pub fn close(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float(c.close)).collect()
    }
    
    /// Get the volume as a series
    pub fn volume(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float(c.volume)).collect()
    }
    
    /// Get the HL2 (high+low)/2 as a series
    pub fn hl2(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float((c.high + c.low) / 2.0)).collect()
    }
    
    /// Get the HLC3 (high+low+close)/3 as a series
    pub fn hlc3(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float((c.high + c.low + c.close) / 3.0)).collect()
    }
    
    /// Get the OHLC4 (open+high+low+close)/4 as a series
    pub fn ohlc4(&self) -> Vec<Value> {
        self.candles.iter().map(|c| Value::Float((c.open + c.high + c.low + c.close) / 4.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    
    #[test]
    fn test_ohlcv_data() {
        let mut data = OHLCVData::with_symbol_and_timeframe("BTCUSD", "1h");
        
        // Add some candles
        data.add_candle(Candle {
            timestamp: Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
        });
        
        data.add_candle(Candle {
            timestamp: Utc.with_ymd_and_hms(2023, 1, 1, 1, 0, 0).unwrap(),
            open: 105.0,
            high: 115.0,
            low: 95.0,
            close: 110.0,
            volume: 1200.0,
        });
        
        // Test data properties
        assert_eq!(data.symbol, "BTCUSD");
        assert_eq!(data.timeframe, "1h");
        assert_eq!(data.len(), 2);
        assert!(!data.is_empty());
        
        // Test series
        let open = data.open();
        assert_eq!(open.len(), 2);
        if let Value::Float(o) = &open[0] {
            assert_eq!(*o, 100.0);
        } else {
            panic!("Expected Float");
        }
        
        let hl2 = data.hl2();
        assert_eq!(hl2.len(), 2);
        if let Value::Float(h) = &hl2[0] {
            assert_eq!(*h, 100.0); // (110 + 90) / 2
        } else {
            panic!("Expected Float");
        }
    }
}
