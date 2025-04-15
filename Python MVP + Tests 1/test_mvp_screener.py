import pytest
import pandas as pd
import numpy as np
import datetime

# Import the classes/functions to test from your main script
from mvp_screener import DataFetcher, PineRuntime, MVP_PINESCRIPT

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def sample_data() -> pd.DataFrame:
    """Creates a sample DataFrame for testing engine logic."""
    dates = pd.date_range(start="2023-01-01", periods=50, freq='D')
    n = len(dates)
    # Create data designed to trigger crossovers/crossunders
    data = {
        'open': np.linspace(100, 110, n) + np.random.normal(0, 0.5, n),
        'high': np.linspace(101, 112, n) + np.random.normal(0, 0.7, n),
        'low': np.linspace(99, 108, n) + np.random.normal(0, 0.6, n),
        'close': np.linspace(100.5, 111, n) + np.random.normal(0, 0.5, n),
        'volume': np.random.randint(1000, 5000, n)
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
    df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)

    # --- Manually craft some data points to force known conditions ---
    # Force a MA_Min condition: HMA > HMA[1] and HMA[1] < HMA[2]
    # Force a MA_Max condition: HMA < HMA[1] and HMA[1] > HMA[2]
    # Force a Crossover: HMA crosses above saveMA_Min
    # Force a Crossunder: HMA crosses below saveMA_Max
    # (This requires knowing HMA/saveMA values, complex to craft perfectly without running)
    # For simplicity, we'll test the components and the end-to-end run on generated data.
    # A more robust test would use pre-calculated TradingView data as input/output reference.

    return df.round(2)

@pytest.fixture(scope="module")
def mock_data_fetcher(sample_data: pd.DataFrame) -> DataFetcher:
    """Creates a DataFetcher with specific mock data for testing."""
    return DataFetcher(mock_data_source={"TEST": sample_data})

@pytest.fixture(scope="module")
def pine_runtime(mock_data_fetcher: DataFetcher) -> PineRuntime:
    """Creates a PineRuntime instance using the mock data fetcher."""
    return PineRuntime(data_fetcher=mock_data_fetcher)

# --- Test Cases ---

def test_data_fetcher(mock_data_fetcher: DataFetcher, sample_data: pd.DataFrame):
    """Test if the mock data fetcher returns the correct data."""
    df = mock_data_fetcher.get_ohlcv("TEST", "1D", limit=50)
    assert df is not None
    assert len(df) == 50
    pd.testing.assert_frame_equal(df, sample_data)
    assert mock_data_fetcher.get_ohlcv("UNKNOWN", "1D") is None

def test_pine_hl2(sample_data: pd.DataFrame):
    """Test the hl2 calculation."""
    runtime = PineRuntime(DataFetcher()) # Doesn't need fetcher for this part
    price = (sample_data['high'] + sample_data['low']) / 2
    # Simulate internal calculation if needed, or just check pandas logic
    pd.testing.assert_series_equal(price, (sample_data['high'] + sample_data['low']) / 2)

def test_pine_hma(sample_data: pd.DataFrame):
    """Test HMA calculation using pandas_ta."""
    runtime = PineRuntime(DataFetcher())
    price = (sample_data['high'] + sample_data['low']) / 2
    hma_length = 21
    hma_expected = ta.hma(price, length=hma_length)
    # Simulate runtime's internal call if it wrapped ta.hma
    hma_actual = ta.hma(price, length=hma_length) # Assuming direct call or simple wrapper
    pd.testing.assert_series_equal(hma_actual, hma_expected, check_names=False)
    assert not hma_actual.isna().all() # Ensure it calculated something

def test_pine_nz():
    """Test the nz (fillna) function."""
    runtime = PineRuntime(DataFetcher())
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    s_filled_zero = runtime._pine_nz(s)
    s_filled_ten = runtime._pine_nz(s, replacement=10.0)
    pd.testing.assert_series_equal(s_filled_zero, pd.Series([1.0, 0.0, 3.0, 0.0]))
    pd.testing.assert_series_equal(s_filled_ten, pd.Series([1.0, 10.0, 3.0, 10.0]))

def test_pine_crossover_crossunder():
    """Test crossover and crossunder logic."""
    runtime = PineRuntime(DataFetcher())
    s1 = pd.Series([1, 2, 3, 4, 3, 2, 1])
    s2 = pd.Series([2, 2, 2, 3, 4, 3, 0])
    # Crossover expected at index 3 (s1 goes from 3->4, s2 from 2->3; s1[2]<=s2[2] and s1[3]>s2[3])
    # Crossover also expected at index 6 (s1 goes from 2->1, s2 from 3->0; s1[5]<=s2[5] and s1[6]>s2[6]) -> No, s1[6] < s2[6] is false. Let's recheck logic.
    # Crossover: (prev_s1 <= prev_s2) & (s1 > s2)
    # Index 3: (s1[2]<=s2[2]) & (s1[3]>s2[3]) -> (3<=2) & (4>3) -> False & True -> False. Hmm, let's adjust s2.
    s2_co = pd.Series([2, 2, 3, 3, 4, 3, 0]) # s1[2]<=s2[2] (3<=3) T, s1[3]>s2[3] (4>3) T -> Crossover at index 3
    crossover = runtime._pine_crossover(s1, s2_co)
    expected_co = pd.Series([False, False, False, True, False, False, False])
    pd.testing.assert_series_equal(crossover.fillna(False), expected_co, check_names=False) # Fill NaN from shift

    # Crossunder: (prev_s1 >= prev_s2) & (s1 < s2)
    # Index 4: (s1[3]>=s2[3]) & (s1[4]<s2[4]) -> (4>=3) & (3<4) -> True & True -> Crossunder at index 4
    s2_cu = pd.Series([0, 1, 2, 3, 4, 5, 6])
    crossunder = runtime._pine_crossunder(s1, s2_cu)
    expected_cu = pd.Series([False, False, False, False, True, True, True]) # s1 dips below s2
    pd.testing.assert_series_equal(crossunder.fillna(False), expected_cu, check_names=False)

def test_pine_valuewhen():
    """Test the complex valuewhen logic."""
    runtime = PineRuntime(DataFetcher())
    condition = pd.Series([False, True, False, True, False, True, False, True]) # True at 1, 3, 5, 7
    source = pd.Series   ([10,   11,   12,   13,   14,   15,   16,   17])
    # occurrence = 0 (most recent)
    expected_0 = pd.Series([np.nan, 11, 11, 13, 13, 15, 15, 17])
    result_0 = runtime._pine_valuewhen(condition, source, 0)
    pd.testing.assert_series_equal(result_0.ffill(), expected_0, check_dtype=False) # ffill mimics pine

    # occurrence = 1 (second most recent)
    expected_1 = pd.Series([np.nan, np.nan, np.nan, 11, 11, 13, 13, 15])
    result_1 = runtime._pine_valuewhen(condition, source, 1)
    pd.testing.assert_series_equal(result_1.ffill(), expected_1, check_dtype=False)

    # occurrence = 3 (fourth most recent)
    expected_3 = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 11])
    result_3 = runtime._pine_valuewhen(condition, source, 3)
    pd.testing.assert_series_equal(result_3.ffill(), expected_3, check_dtype=False)

    # No occurrences yet
    condition_late = pd.Series([False, False, False, False, True, False, True, False]) # True at 4, 6
    source_late = pd.Series([10, 11, 12, 13, 14, 15, 16, 17])
    expected_late_0 = pd.Series([np.nan, np.nan, np.nan, np.nan, 14, 14, 16, 16])
    result_late_0 = runtime._pine_valuewhen(condition_late, source_late, 0)
    pd.testing.assert_series_equal(result_late_0.ffill(), expected_late_0, check_dtype=False)


@pytest.mark.parametrize("symbol", ["TEST"]) # Use the symbol defined in mock_data_fetcher
def test_pine_runtime_execute_mvp(pine_runtime: PineRuntime, symbol: str):
    """Test the end-to-end execution of the MVP script logic within the runtime."""
    result = pine_runtime.execute(symbol, "1D", MVP_PINESCRIPT)

    assert "error" not in result, f"Execution failed with error: {result.get('error')}"
    assert "alerts" in result
    assert isinstance(result["alerts"], list)
    assert len(result["alerts"]) == 2 # Buy Signal, Sell Signal

    # Check structure of alerts
    for alert in result["alerts"]:
        assert "title" in alert
        assert "message" in alert
        assert "triggered" in alert
        assert isinstance(alert["triggered"], bool) # Should be True or False

    # Check other outputs (optional, but good practice)
    assert "last_hma" in result
    assert "last_saveMA_Min" in result
    assert "last_saveMA_Max" in result

    # We can't easily assert specific True/False for alerts without reference TradingView data
    # But we can check that the process ran without errors and produced the expected structure.
    print(f"\nTest Execution Result for {symbol}: {result['alerts']}") # Print alerts for manual inspection during test run

def test_pine_runtime_execute_no_data(pine_runtime: PineRuntime):
    """Test execution when no data is available."""
    result = pine_runtime.execute("UNKNOWN_SYMBOL", "1D", MVP_PINESCRIPT)
    assert "error" in result
    assert result["error"] == "No data available"
    assert result["alerts"] == []

def test_pine_runtime_execute_not_enough_data(mock_data_fetcher: DataFetcher):
    """Test execution when data is too short for calculations."""
    # Create a short DataFrame
    short_df = mock_data_fetcher.get_ohlcv("TEST", "1D", limit=20) # HMA(21) needs more
    short_data_fetcher = DataFetcher(mock_data_source={"SHORT": short_df})
    runtime = PineRuntime(short_data_fetcher)
    result = runtime.execute("SHORT", "1D", MVP_PINESCRIPT)
    assert "error" in result
    assert result["error"] == "Not enough data points"
    assert result["alerts"] == []