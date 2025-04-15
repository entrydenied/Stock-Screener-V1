// Data fetcher module for validation
// Responsible for fetching historical OHLCV data

use crate::runtime::data::{OHLCVData, Candle};
use crate::validation::ValidationError;
use chrono::{DateTime, Utc, TimeZone, Duration};
use reqwest;
use serde::{Deserialize, Serialize};

/// Fetch historical OHLCV data for a symbol and timeframe
pub async fn fetch_data(symbol: &str, timeframe: &str) -> Result<OHLCVData, ValidationError> {
    // For the MVP, we'll use a free API like Alpha Vantage or Yahoo Finance
    // This is a simplified implementation that will need to be expanded
    
    match timeframe {
        "1d" => fetch_daily_data(symbol).await,
        "1h" => fetch_hourly_data(symbol).await,
        _ => Err(ValidationError::DataFetchingError(
            format!("Unsupported timeframe: {}", timeframe)
        )),
    }
}

/// Fetch daily OHLCV data
async fn fetch_daily_data(symbol: &str) -> Result<OHLCVData, ValidationError> {
    // Use Alpha Vantage API for daily data
    let api_key = "demo"; // Use demo key for now, would be replaced with a real key
    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={}&apikey={}&outputsize=full",
        symbol, api_key
    );
    
    let response = reqwest::get(&url)
        .await
        .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
    
    let data: AlphaVantageResponse = response
        .json()
        .await
        .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
    
    // Convert Alpha Vantage response to OHLCVData
    let mut ohlcv_data = OHLCVData::with_symbol_and_timeframe(symbol, "1d");
    
    if let Some(time_series) = data.time_series_daily {
        for (date_str, daily_data) in time_series {
            // Parse date
            let date = chrono::NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
            
            // Create UTC datetime at midnight
            let timestamp = Utc.from_utc_datetime(&date.and_hms_opt(0, 0, 0).unwrap());
            
            // Create candle
            let candle = Candle {
                timestamp,
                open: daily_data.open.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                high: daily_data.high.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                low: daily_data.low.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                close: daily_data.close.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                volume: daily_data.volume.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
            };
            
            ohlcv_data.add_candle(candle);
        }
    } else {
        return Err(ValidationError::DataFetchingError(
            "No time series data found in response".to_string()
        ));
    }
    
    // Sort candles by timestamp (oldest first)
    ohlcv_data.candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    
    Ok(ohlcv_data)
}

/// Fetch hourly OHLCV data
async fn fetch_hourly_data(symbol: &str) -> Result<OHLCVData, ValidationError> {
    // Use Alpha Vantage API for hourly data
    let api_key = "demo"; // Use demo key for now, would be replaced with a real key
    let url = format!(
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={}&interval=60min&apikey={}&outputsize=full",
        symbol, api_key
    );
    
    let response = reqwest::get(&url)
        .await
        .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
    
    let data: AlphaVantageIntradayResponse = response
        .json()
        .await
        .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
    
    // Convert Alpha Vantage response to OHLCVData
    let mut ohlcv_data = OHLCVData::with_symbol_and_timeframe(symbol, "1h");
    
    if let Some(time_series) = data.time_series_60min {
        for (datetime_str, hourly_data) in time_series {
            // Parse datetime
            let datetime = DateTime::parse_from_str(&datetime_str, "%Y-%m-%d %H:%M:%S")
                .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?;
            
            // Convert to UTC
            let timestamp = datetime.with_timezone(&Utc);
            
            // Create candle
            let candle = Candle {
                timestamp,
                open: hourly_data.open.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                high: hourly_data.high.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                low: hourly_data.low.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                close: hourly_data.close.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
                volume: hourly_data.volume.parse::<f64>()
                    .map_err(|e| ValidationError::DataFetchingError(e.to_string()))?,
            };
            
            ohlcv_data.add_candle(candle);
        }
    } else {
        return Err(ValidationError::DataFetchingError(
            "No time series data found in response".to_string()
        ));
    }
    
    // Sort candles by timestamp (oldest first)
    ohlcv_data.candles.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    
    Ok(ohlcv_data)
}

/// Alpha Vantage API response for daily data
#[derive(Debug, Deserialize)]
struct AlphaVantageResponse {
    #[serde(rename = "Meta Data")]
    meta_data: Option<AlphaVantageMetaData>,
    
    #[serde(rename = "Time Series (Daily)")]
    time_series_daily: Option<std::collections::HashMap<String, AlphaVantageDaily>>,
}

/// Alpha Vantage API response for intraday data
#[derive(Debug, Deserialize)]
struct AlphaVantageIntradayResponse {
    #[serde(rename = "Meta Data")]
    meta_data: Option<AlphaVantageMetaData>,
    
    #[serde(rename = "Time Series (60min)")]
    time_series_60min: Option<std::collections::HashMap<String, AlphaVantageIntraday>>,
}

/// Alpha Vantage API meta data
#[derive(Debug, Deserialize)]
struct AlphaVantageMetaData {
    #[serde(rename = "1. Information")]
    information: String,
    
    #[serde(rename = "2. Symbol")]
    symbol: String,
    
    #[serde(rename = "3. Last Refreshed")]
    last_refreshed: String,
    
    #[serde(rename = "4. Output Size")]
    output_size: Option<String>,
    
    #[serde(rename = "5. Time Zone")]
    time_zone: String,
}

/// Alpha Vantage API daily data
#[derive(Debug, Deserialize)]
struct AlphaVantageDaily {
    #[serde(rename = "1. open")]
    open: String,
    
    #[serde(rename = "2. high")]
    high: String,
    
    #[serde(rename = "3. low")]
    low: String,
    
    #[serde(rename = "4. close")]
    close: String,
    
    #[serde(rename = "5. volume")]
    volume: String,
}

/// Alpha Vantage API intraday data
#[derive(Debug, Deserialize)]
struct AlphaVantageIntraday {
    #[serde(rename = "1. open")]
    open: String,
    
    #[serde(rename = "2. high")]
    high: String,
    
    #[serde(rename = "3. low")]
    low: String,
    
    #[serde(rename = "4. close")]
    close: String,
    
    #[serde(rename = "5. volume")]
    volume: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_fetch_daily_data() {
        // This test requires internet connection and may fail if API limits are reached
        let result = fetch_daily_data("IBM").await;
        
        // Just check that we get a result, don't validate the data
        assert!(result.is_ok());
        
        let data = result.unwrap();
        assert_eq!(data.symbol, "IBM");
        assert_eq!(data.timeframe, "1d");
        assert!(!data.candles.is_empty());
    }
}
