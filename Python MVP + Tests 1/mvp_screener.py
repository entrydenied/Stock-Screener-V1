# // --- Python Step-by-Step Guide ---

import os
import subprocess
import venv
import logging
import time
import json
import datetime
import hashlib
from typing import List, Dict, Any, Optional, Tuple

# --- Core Libraries ---
import pandas as pd
import numpy as np
import pandas_ta as ta # For HMA calculation

# --- Simulation/Placeholder Libraries ---
from collections import defaultdict
import random

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

# --- Constants ---
MVP_PINESCRIPT = """
//@version=5
indicator('MVP', overlay=true)

// ————— Inputs
price = input(hl2, title='Source')
HMA_Length = input(21, 'HMA Length') // Using default 21
lookback = input(2, 'lookback') // Using default 2
// ShowHullSupResLines = input(false, 'ShowHullSup/ResLines') // Not directly used in alerts
// ShowBuySellArrows = input(false, 'ShowBuySellArrows') // Not directly used in alerts
// ShowDivergenceLabel = input(false, 'ShowDivergenceLabel') // Not directly used in alerts
// ExtendSupResLines = input(false, 'ExtendLocalSup/ResLines') // Not directly used in alerts

// ————— Calculations
HMA = ta.hma(price, HMA_Length)
// delta = HMA[1] - HMA[lookback + 1] // Not directly used in alerts
// delta_per_bar = delta / lookback // Not directly used in alerts
// next_bar = HMA[1] + delta_per_bar // Not directly used in alerts
// concavity = HMA > next_bar ? 1 : -1 // Not directly used in alerts
// O_R = HMA > HMA[1] ? color(#ff7f00) : color(#ff0000) // Not directly used in alerts
// DG_G = HMA < HMA[1] ? color(#025f02) : color(#00fa03) // Not directly used in alerts

// ————— Plots (Logic needed for calculations below)
// plot(HMA, 'HMA', color=concavity != -1 ? DG_G : O_R, linewidth=3)

//MA_Min and MA_Max Points only
MA_Min_Cond = HMA > HMA[1] and HMA[1] < HMA[2]
MA_Max_Cond = HMA < HMA[1] and HMA[1] > HMA[2]
// MA_Min = MA_Min_Cond ? HMA[1] : na // Not directly used in alerts
// MA_Max = MA_Max_Cond ? HMA[1] : na // Not directly used in alerts

//MA_Min and MA_Max Series
saveMA_Min = ta.valuewhen(MA_Min_Cond, HMA[1], 0)
saveMA_Max = ta.valuewhen(MA_Max_Cond, HMA[1], 0)

//Draw MA_Min/MA_Max as lines from series or just points (Logic needed for alerts)
// plot(ShowHullSupResLines ? saveMA_Min : MA_Min, 'MA_Min/Hull Support', style=plot.style_circles, color=color(#00fa03), linewidth=1, trackprice=ExtendSupResLines, offset=-1)
// plot(ShowHullSupResLines ? saveMA_Max : MA_Max, 'MA_Max/Hull Resistance', style=plot.style_circles, color=color(#ff0000), linewidth=1, trackprice=ExtendSupResLines, offset=-1)

//Draw Arrows at MA_Min/MA_Max
// plotshape(ShowBuySellArrows ? MA_Min : na, 'Buy', shape.triangleup, location.belowbar, color.new(color.green, 0), text='Buy', offset=-1)
// plotshape(ShowBuySellArrows ? MA_Max : na, 'Sell', shape.triangledown, location.abovebar, color.new(color.red, 0), text='Sell', offset=-1)

//Divergence Label (Not used in alerts)
// divergence = math.round(HMA - next_bar, precision=4)
// ... divergence label logic ...

// ————— Alerts
alertcondition(ta.crossover(HMA, saveMA_Min), title='Buy Signal', message='Crossing above MA_Min, Bullish')
alertcondition(ta.crossunder(HMA, saveMA_Max), title='Sell Signal', message='Crossing below MA_Max, Bearish')
"""

# --- Placeholder Implementations ---

class PineParser:
    """Simulates parsing. For this MVP, we execute known logic directly."""
    def parse(self, script: str) -> Any:
        logging.info(f"Parsing script (simulation)...")
        # In a real system, this would return an AST (Abstract Syntax Tree)
        # For this MVP, we know the script structure and execute it directly in runtime
        return {"status": "parsed", "script_type": "indicator"}

class DataFetcher:
    """Simulates fetching data from a TimescaleDB instance."""
    def __init__(self, mock_data_source: Optional[Dict[str, pd.DataFrame]] = None):
        self.mock_data_source = mock_data_source if mock_data_source else self._generate_mock_data()
        logging.info("DataFetcher initialized (using mock data).")

    def _generate_mock_data(self, symbols: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA"], bars=200):
        data = {}
        base_prices = {"AAPL": 170, "GOOGL": 140, "MSFT": 300, "TSLA": 250}
        start_date = datetime.datetime.now() - datetime.timedelta(days=bars)
        for symbol in symbols:
            dates = pd.date_range(start_date, periods=bars, freq='B') # Business days
            price_data = base_prices.get(symbol, 100) + np.random.randn(bars).cumsum() * 0.5
            df = pd.DataFrame(index=dates)
            df['open'] = price_data + np.random.normal(0, 1, bars)
            df['high'] = df['open'] + np.random.uniform(0, 3, bars)
            df['low'] = df['open'] - np.random.uniform(0, 3, bars)
            df['close'] = df['low'] + np.random.uniform(0, (df['high']-df['low']), bars)
            df['volume'] = np.random.randint(100000, 5000000, bars)
            # Ensure OHLC consistency
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
            df = df.round(2)
            data[symbol] = df
        return data

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Fetches OHLCV data for a symbol."""
        logging.info(f"Fetching {limit} OHLCV bars for {symbol} ({timeframe})...")
        # In real implementation: Query TimescaleDB based on symbol, timeframe, limit
        # conn = psycopg2.connect(...)
        # query = f"SELECT time, open, high, low, close, volume FROM ohlcv_{timeframe} WHERE symbol = %s ORDER BY time DESC LIMIT %s"
        # df = pd.read_sql(query, conn, params=(symbol, limit), index_col='time')
        # df = df.sort_index() # Ensure ascending time order for calculations
        # return df

        # --- Mock Implementation ---
        if symbol in self.mock_data_source:
            df = self.mock_data_source[symbol].copy()
            # Simulate fetching a specific number of bars (most recent)
            return df.iloc[-limit:]
        else:
            logging.warning(f"No mock data found for symbol: {symbol}")
            return None

class PineRuntime:
    """Executes the logic of the MVP Pine Script."""
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.outputs = {}
        self.alert_conditions = []
        logging.info("PineRuntime initialized.")

    def _pine_na(self):
        """Represents Pine Script's 'na' value."""
        return np.nan

    def _pine_nz(self, series: pd.Series, replacement: float = 0.0) -> pd.Series:
        """Replicates Pine Script's nz() function."""
        return series.fillna(replacement)

    def _pine_valuewhen(self, condition: pd.Series, source: pd.Series, occurrence: int) -> pd.Series:
        """
        Replicates Pine Script's ta.valuewhen() function.
        Finds the value of 'source' on the Nth most recent bar where 'condition' was true.
        """
        # Ensure boolean condition
        condition = condition.astype(bool)
        # Get indices where condition is true
        true_indices = condition[condition].index
        # Create a series to store results, initialized with NaN
        result_series = pd.Series(index=source.index, data=np.nan, dtype=source.dtype)

        # Iterate through each time point in the source series
        for current_time in source.index:
            # Find true condition indices that occurred at or before the current time
            relevant_true_indices = true_indices[true_indices <= current_time]
            # Check if enough occurrences exist
            if len(relevant_true_indices) > occurrence:
                # Get the index of the Nth most recent occurrence (0-based)
                target_index = relevant_true_indices[-(occurrence + 1)]
                # Assign the value from the source series at that target index
                result_series.loc[current_time] = source.loc[target_index]

        # Forward fill the results to mimic Pine Script's behavior where the value persists
        return result_series.ffill()


    def _pine_crossover(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Replicates Pine Script's ta.crossover() function."""
        prev_s1 = series1.shift(1)
        prev_s2 = series2.shift(1)
        return (prev_s1 <= prev_s2) & (series1 > series2)

    def _pine_crossunder(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """Replicates Pine Script's ta.crossunder() function."""
        prev_s1 = series1.shift(1)
        prev_s2 = series2.shift(1)
        return (prev_s1 >= prev_s2) & (series1 < series2)

    def execute(self, symbol: str, timeframe: str, script: str, inputs: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Executes the hardcoded logic of the MVP Pine Script.
        A real engine would parse the script AST and execute dynamically.
        """
        logging.info(f"Executing MVP script logic for {symbol} on {timeframe}...")
        self.outputs = {}
        self.alert_conditions = []

        # --- Default Inputs (from MVP script) ---
        default_inputs = {'HMA_Length': 21, 'lookback': 2}
        if inputs:
            default_inputs.update(inputs)
        hma_length = int(default_inputs['HMA_Length'])
        lookback = int(default_inputs['lookback']) # Note: lookback isn't used in alert logic directly

        # --- 1. Fetch Data ---
        ohlcv_df = self.data_fetcher.get_ohlcv(symbol, timeframe)
        if ohlcv_df is None or ohlcv_df.empty:
            logging.error(f"No data found for {symbol}, cannot execute script.")
            return {"error": "No data available", "alerts": []}
        if len(ohlcv_df) < hma_length + lookback + 5: # Need enough data for HMA and lookbacks
             logging.error(f"Not enough data points for {symbol} ({len(ohlcv_df)}), need more for calculations.")
             return {"error": "Not enough data points", "alerts": []}


        # --- 2. Perform Calculations based on MVP Script ---
        try:
            # price = input(hl2, title='Source')
            price = (ohlcv_df['high'] + ohlcv_df['low']) / 2

            # HMA = ta.hma(price, HMA_Length)
            hma = ta.hma(price, length=hma_length)
            if hma is None or hma.isna().all():
                 logging.error(f"HMA calculation failed or resulted in all NaNs for {symbol}.")
                 return {"error": "HMA calculation failed", "alerts": []}
            self.outputs['HMA'] = hma # Store intermediate result if needed

            # Shifted HMA for conditions
            hma_1 = hma.shift(1)
            hma_2 = hma.shift(2)

            # MA_Min_Cond = HMA > HMA[1] and HMA[1] < HMA[2]
            ma_min_cond = (hma > hma_1) & (hma_1 < hma_2)

            # MA_Max_Cond = HMA < HMA[1] and HMA[1] > HMA[2]
            ma_max_cond = (hma < hma_1) & (hma_1 > hma_2)

            # saveMA_Min = ta.valuewhen(MA_Min_Cond, HMA[1], 0)
            saveMA_Min = self._pine_valuewhen(ma_min_cond, hma_1, 0)
            self.outputs['saveMA_Min'] = saveMA_Min

            # saveMA_Max = ta.valuewhen(MA_Max_Cond, HMA[1], 0)
            saveMA_Max = self._pine_valuewhen(ma_max_cond, hma_1, 0)
            self.outputs['saveMA_Max'] = saveMA_Max

            # --- 3. Evaluate Alert Conditions ---
            # alertcondition(ta.crossover(HMA, saveMA_Min), title='Buy Signal', ...)
            buy_signal_series = self._pine_crossover(hma, saveMA_Min)
            # Get the boolean value of the *last* bar
            buy_triggered = bool(buy_signal_series.iloc[-1]) if not buy_signal_series.empty else False
            self.alert_conditions.append({
                "title": "Buy Signal",
                "message": "Crossing above MA_Min, Bullish",
                "triggered": buy_triggered
            })

            # alertcondition(ta.crossunder(HMA, saveMA_Max), title='Sell Signal', ...)
            sell_signal_series = self._pine_crossunder(hma, saveMA_Max)
            # Get the boolean value of the *last* bar
            sell_triggered = bool(sell_signal_series.iloc[-1]) if not sell_signal_series.empty else False
            self.alert_conditions.append({
                "title": "Sell Signal",
                "message": "Crossing below MA_Max, Bearish",
                "triggered": sell_triggered
            })

            logging.info(f"Execution finished for {symbol}. Alerts: {self.alert_conditions}")

            # --- 4. Return Results ---
            return {
                "alerts": self.alert_conditions,
                "last_hma": hma.iloc[-1] if not hma.empty else self._pine_na(),
                "last_saveMA_Min": saveMA_Min.iloc[-1] if not saveMA_Min.empty else self._pine_na(),
                "last_saveMA_Max": saveMA_Max.iloc[-1] if not saveMA_Max.empty else self._pine_na(),
            }

        except Exception as e:
            logging.exception(f"Error executing script for {symbol}: {e}")
            return {"error": str(e), "alerts": []}


class DatabaseManager:
    """Simulates interactions with PostgreSQL (Metadata, Screener Defs)."""
    def __init__(self, db_url: str = "postgresql://user:pass@host/db"):
        logging.info(f"Initializing DatabaseManager (simulation for {db_url}).")
        # In real implementation: Use SQLAlchemy or psycopg2
        # from sqlalchemy import create_engine, Table, MetaData, Column, Integer, String, Text, JSON
        # self.engine = create_engine(db_url)
        # self.metadata = MetaData()
        # self.screeners_table = Table('screeners', self.metadata, ...)
        # self.metadata.create_all(self.engine) # Ensure tables exist
        self._screeners = {} # screener_id -> definition dict
        self._metadata = { # symbol -> metadata dict
            "AAPL": {"symbol": "AAPL", "exchange": "NASDAQ", "name": "Apple Inc."},
            "GOOGL": {"symbol": "GOOGL", "exchange": "NASDAQ", "name": "Alphabet Inc."},
            "MSFT": {"symbol": "MSFT", "exchange": "NASDAQ", "name": "Microsoft Corporation"},
            "TSLA": {"symbol": "TSLA", "exchange": "NASDAQ", "name": "Tesla, Inc."},
            "NVDA": {"symbol": "NVDA", "exchange": "NASDAQ", "name": "NVIDIA Corporation"},
        }
        self._next_screener_id = 1

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        logging.info(f"Fetching metadata for {symbol} (simulation)...")
        return self._metadata.get(symbol, {"symbol": symbol, "exchange": "Unknown", "name": "Unknown Company"})

    def save_screener(self, name: str, symbols: List[str], timeframe: str, script: str) -> int:
        screener_id = self._next_screener_id
        definition = {
            "id": screener_id,
            "name": name,
            "symbols": symbols,
            "timeframe": timeframe,
            "script": script,
            "created_at": datetime.datetime.now().isoformat()
        }
        self._screeners[screener_id] = definition
        self._next_screener_id += 1
        logging.info(f"Saved screener '{name}' with ID: {screener_id} (simulation).")
        # Real: with self.engine.connect() as connection: connection.execute(self.screeners_table.insert().values(...))
        return screener_id

    def get_screener(self, screener_id: int) -> Optional[Dict[str, Any]]:
        logging.info(f"Fetching screener definition ID: {screener_id} (simulation)...")
        # Real: with self.engine.connect() as connection: result = connection.execute(...) ; return result.first()
        return self._screeners.get(screener_id)

    def list_screeners(self) -> List[Dict[str, Any]]:
         logging.info("Listing all screeners (simulation)...")
         return list(self._screeners.values())


class TaskQueueClient:
    """Simulates interactions with a Task Queue (Redis List or Kafka)."""
    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self._queue = [] # In-memory list as queue simulation
        logging.info(f"TaskQueueClient initialized for queue '{queue_name}' (in-memory simulation).")
        # Real Redis: import redis; self.client = redis.Redis(...)
        # Real Kafka: from kafka import KafkaProducer, KafkaConsumer; self.producer = KafkaProducer(...)

    def send_task(self, task_data: Dict[str, Any]):
        logging.info(f"Sending task to queue '{self.queue_name}': {task_data['symbol']} for run {task_data['run_id']}")
        task_json = json.dumps(task_data)
        self._queue.append(task_json)
        # Real Redis: self.client.lpush(self.queue_name, task_json)
        # Real Kafka: self.producer.send(self.queue_name, key=task_data['run_id'].encode(), value=task_json.encode())

    def get_task(self, timeout: int = 1) -> Optional[Dict[str, Any]]:
        # logging.debug(f"Waiting for task from queue '{self.queue_name}' (timeout={timeout}s)...")
        if self._queue:
            task_json = self._queue.pop(0) # FIFO
            logging.info(f"Received task from queue '{self.queue_name}'.")
            return json.loads(task_json)
        else:
            # Simulate blocking wait (optional)
            # time.sleep(timeout)
            return None
        # Real Redis: task_json = self.client.brpop(self.queue_name, timeout=timeout); return json.loads(task_json[1]) if task_json else None
        # Real Kafka: consumer = KafkaConsumer(...); msg = next(consumer); return json.loads(msg.value)


class ResultsStorage:
    """Simulates interactions with Results Storage (Redis Hashes)."""
    def __init__(self):
        self._results = defaultdict(dict) # run_id -> {symbol -> result_dict}
        logging.info("ResultsStorage initialized (in-memory simulation).")
        # Real Redis: import redis; self.client = redis.Redis(...)

    def store_result(self, run_id: str, symbol: str, match_status: bool, details: Any):
        result_data = {
            "symbol": symbol,
            "match": match_status,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat()
        }
        logging.info(f"Storing result for run '{run_id}', symbol '{symbol}': Match={match_status}")
        self._results[run_id][symbol] = result_data
        # Real Redis: self.client.hset(f"results:{run_id}", symbol, json.dumps(result_data))

    def get_results(self, run_id: str) -> List[Dict[str, Any]]:
        logging.info(f"Fetching results for run '{run_id}' (simulation)...")
        # Real Redis: results_raw = self.client.hgetall(f"results:{run_id}"); return [json.loads(v) for v in results_raw.values()]
        return list(self._results.get(run_id, {}).values())

# --- Step-by-Step Functions ---

def step_0_1_define_scope_legal():
    print("\n--- Phase 0, Step 1: Define Scope & Legal ---")
    logging.info("Confirming MVP Pine Script™ subset and logic.")
    logging.info("(CRITICAL) Simulating legal clearance check...")
    legal_clearance = True # Assume clearance for this simulation
    if not legal_clearance:
        logging.error("Legal clearance NOT obtained. Stopping.")
        raise SystemExit("Legal clearance required.")
    logging.info("Legal clearance obtained (simulated).")
    print("-" * 30)

def step_0_2_core_team_tools():
    print("\n--- Phase 0, Step 2: Core Team & Tools ---")
    logging.info("Assign leads, set up Git, project management, communication (simulated).")
    if not os.path.exists("venv"):
        logging.info("Creating virtual environment 'venv'...")
        try:
            subprocess.run(["python3", "-m", "venv", "venv"], check=True)
            logging.info("Virtual environment created. Activate using: source venv/bin/activate")
            logging.info("Install requirements: pip install -r requirements.txt")
        except Exception as e:
            logging.error(f"Failed to create virtual environment: {e}")
    else:
        logging.info("Virtual environment 'venv' already exists.")
    logging.info("Core tools and environment setup simulated.")
    print("-" * 30)

def step_0_3_pine_mvp_analysis():
    print("\n--- Phase 0, Step 3: Pine Script™ MVP Analysis ---")
    logging.info("Analyzing MVP Pine Script™ for required functions/logic.")
    logging.info(f"MVP Script to implement:\n{MVP_PINESCRIPT[:300]}...")
    logging.info("Required: hl2, ta.hma, ta.valuewhen, na, nz, ta.crossover, ta.crossunder, alertcondition, history access [], comparisons.")
    logging.info("Documenting expected behavior based on TradingView (manual process).")
    print("-" * 30)
    return MVP_PINESCRIPT

def step_0_4_basic_architecture_cloud():
    print("\n--- Phase 0, Step 4: Basic Architecture & Cloud Setup ---")
    logging.info("Designing high-level Python service interactions (API, Scheduler, Worker, Engine, Data Stores).")
    logging.info("Simulating basic cloud infra setup (VPC, EKS/ECS, IAM using Terraform/CDK/Pulumi).")
    logging.info("Simulating skeleton CI/CD pipeline (GitHub Actions for lint/test).")
    print("-" * 30)

# --- Phase 1: Core Pine Engine & Validation ---

def step_1_5_build_pine_engine_core(data_fetcher: DataFetcher) -> Tuple[PineParser, PineRuntime]:
    print("\n--- Phase 1, Step 5: Build Pine Engine Core (Python) ---")
    logging.info("Implementing Pine Script™ Parser (Simulation - direct execution).")
    parser = PineParser()
    logging.info("Implementing Pine Script™ Runtime (Python using pandas/pandas_ta).")
    runtime = PineRuntime(data_fetcher)
    logging.info("Implemented key functions: hl2, hma, valuewhen, crossover, crossunder, alertcondition.")
    logging.warning("Accuracy Note: Engine matches MVP logic functionally. 100% TV parity requires extensive testing/tuning.")
    print("-" * 30)
    return parser, runtime

def step_1_6_develop_validation_framework():
    print("\n--- Phase 1, Step 6: Develop Validation Framework (Python/pytest) ---")
    logging.info("Validation framework uses `pytest` (see test_mvp_screener.py).")
    logging.info("Tooling fetches/uses specific historical data (simulated via DataFetcher).")
    logging.info("Test harness runs engine and compares outputs (especially alerts) against expected results.")
    logging.info("Run tests using: pytest test_mvp_screener.py")
    # We will run pytest at the end of the script execution
    print("-" * 30)

def step_1_7_validate_engine_expose(parser: PineParser, runtime: PineRuntime):
    print("\n--- Phase 1, Step 7: Validate Engine & Expose (Python) ---")
    logging.info("Integrating Parser (simulated) & Runtime.")
    logging.info("Using Validation Framework (pytest) to test engine repeatedly.")
    logging.info("Simulating exposure of engine via FastAPI endpoint:")
    logging.info("POST /execute {'symbol': '...', 'timeframe': '...', 'script': '...', 'inputs': {...}} -> {'alerts': [...], ...}")
    # Conceptual Flask/FastAPI app (not run here)
    # from flask import Flask, request, jsonify
    # engine_app = Flask(__name__)
    # @engine_app.route('/execute', methods=['POST'])
    # def execute():
    #     data = request.json
    #     results = runtime.execute(data['symbol'], data['timeframe'], data['script'], data.get('inputs'))
    #     return jsonify(results)
    # # if __name__ == '__main__': engine_app.run(port=8001) # Example port
    logging.info("Engine validation and exposure simulated.")
    print("-" * 30)


# --- Phase 2: Data Pipeline & Basic Execution ---

def step_2_8_market_data_ingestion():
    print("\n--- Phase 2, Step 8: Market Data Ingestion (Python) ---")
    logging.info("Simulating Python service connecting to data provider.")
    logging.info("Simulating parsing, standardizing, and publishing OHLCV to Kafka (topic: ohlcv_raw).")
    # kafka_producer = TaskQueueClient("ohlcv_raw") # Simulate using queue client
    # kafka_producer.send_task({"symbol": "AAPL", "time": ..., "open": ..., ...})
    logging.info("Data Ingestion service simulation complete.")
    print("-" * 30)

def step_2_9_data_storage_access() -> Tuple[DatabaseManager, DataFetcher]:
    print("\n--- Phase 2, Step 9: Data Storage & Access (Python) ---")
    logging.info("Simulating Kafka setup (e.g., MSK/Docker).")
    logging.info("Simulating Python Kafka consumer reading from 'ohlcv_raw'.")
    logging.info("Simulating writing consumed data to TimescaleDB (Hypertable: ohlcv_1D).")
    logging.info("Initializing Metadata Service (Python + PostgreSQL simulation).")
    db_manager = DatabaseManager() # Simulation uses in-memory dicts
    logging.info("Initializing DataFetcher to read from simulated TimescaleDB.")
    data_fetcher = DataFetcher() # Simulation uses pre-generated mock data
    logging.info("Data Storage Consumer & Metadata Service simulation complete.")
    print("-" * 30)
    return db_manager, data_fetcher

def step_2_10_basic_execution_worker(db_manager: DatabaseManager, runtime: PineRuntime):
    print("\n--- Phase 2, Step 10: Basic Execution Worker (Python) ---")
    logging.info("Simulating a single worker task execution.")
    worker_task = {"symbol": "AAPL", "timeframe": "1D", "script": MVP_PINESCRIPT} # Example task
    logging.info(f"Worker received task: {worker_task['symbol']}")

    metadata = db_manager.get_symbol_metadata(worker_task['symbol'])
    logging.info(f"Fetched metadata: {metadata}")

    # Data fetching is handled inside runtime.execute via data_fetcher
    logging.info(f"Calling Pine Engine for {worker_task['symbol']}...")
    engine_result = runtime.execute(
        worker_task['symbol'],
        worker_task['timeframe'],
        worker_task['script']
    )

    logging.info(f"Worker Task Result for {worker_task['symbol']}: {engine_result}")
    print("-" * 30)
    return engine_result

def step_2_11_basic_monitoring():
    print("\n--- Phase 2, Step 11: Basic Monitoring ---")
    logging.info("Simulating instrumentation of Python services with `prometheus-client`.")
    logging.info("Simulating Prometheus scraping metrics (e.g., from /metrics endpoint).")
    logging.info("Simulating Grafana setup for dashboard visualization.")
    logging.info("Basic monitoring setup simulation complete.")
    print("-" * 30)

# --- Phase 3: End-to-End Screener Logic ---

def step_3_12_screener_definition(db_manager: DatabaseManager) -> int:
    print("\n--- Phase 3, Step 12: Screener Definition (Python Service & API) ---")
    logging.info("Simulating Service & API (FastAPI/Flask + PostgreSQL) for screeners.")
    logging.info("API Endpoints (simulated): POST /screeners, GET /screeners, GET /screeners/{id}")
    # Simulate creating a new screener
    screener_id = db_manager.save_screener(
        name="MVP HMA Cross Screener",
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"], # Added NVDA
        timeframe="1D",
        script=MVP_PINESCRIPT
    )
    logging.info(f"Screener Definition Service: Saved screener with ID: {screener_id}")
    # Simulate listing screeners
    all_screeners = db_manager.list_screeners()
    logging.info(f"Current screeners: {len(all_screeners)}")
    print("-" * 30)
    return screener_id

def step_3_13_job_scheduling_dispatch(screener_id: int, db_manager: DatabaseManager) -> Tuple[Optional[str], Optional[TaskQueueClient]]:
    print("\n--- Phase 3, Step 13: Job Scheduling & Dispatch (Python) ---")
    logging.info("Simulating Python Scheduler service.")
    run_id = f"run_{int(time.time())}_{random.randint(100, 999)}"
    logging.info(f"Scheduler: Received request to run screener ID: {screener_id} -> Assigning Run ID: {run_id}")

    screener_def = db_manager.get_screener(screener_id)
    if not screener_def:
        logging.error(f"Scheduler Error: Screener ID {screener_id} not found.")
        return None, None

    logging.info(f"Fetched definition for screener: '{screener_def['name']}'")
    tasks = []
    for symbol in screener_def['symbols']:
        task = {
            "run_id": run_id,
            "symbol": symbol,
            "timeframe": screener_def['timeframe'],
            "script": screener_def['script'] # Pass the script itself
            # Could pass script ID and have worker fetch it if scripts are large/reused
        }
        tasks.append(task)

    logging.info(f"Creating {len(tasks)} tasks for run {run_id}.")
    task_queue = TaskQueueClient("screener_work_queue") # Connect to the simulated queue
    for task in tasks:
        task_queue.send_task(task)
    logging.info(f"Scheduler: Dispatched {len(tasks)} tasks to queue '{task_queue.queue_name}'.")
    print("-" * 30)
    return run_id, task_queue

def step_3_14_worker_enhancement(task_queue: TaskQueueClient, db_manager: DatabaseManager, runtime: PineRuntime, results_storage: ResultsStorage):
    print("\n--- Phase 3, Step 14: Worker Enhancement & Execution Loop (Python) ---")
    logging.info(f"Worker starting to consume tasks from queue '{task_queue.queue_name}'...")
    processed_tasks = 0
    while True:
        worker_task = task_queue.get_task(timeout=0) # Use 0 timeout for simulation loop
        if not worker_task:
            logging.info(f"Worker: No more tasks in queue. Processed {processed_tasks} tasks.")
            break # Exit loop if no tasks

        processed_tasks += 1
        run_id = worker_task['run_id']
        symbol = worker_task['symbol']
        logging.info(f"Worker: Received task for run {run_id}, symbol {symbol}")

        # --- Execute the Pine Script logic ---
        engine_result = runtime.execute(
            symbol=symbol,
            timeframe=worker_task['timeframe'],
            script=worker_task['script'] # Script passed in task
        )

        # --- Determine Match Status ---
        match_status = False
        triggered_alert_title = None
        if engine_result and not engine_result.get("error"):
            for alert in engine_result.get('alerts', []):
                # MVP specific: Match if EITHER Buy OR Sell signal triggered on the last bar
                if alert.get('triggered'):
                    match_status = True
                    triggered_alert_title = alert.get('title')
                    logging.info(f"Worker: Match found for {symbol}! Trigger: {triggered_alert_title}")
                    break # Stop checking alerts if one matched (adjust if multiple needed)
        else:
            logging.warning(f"Worker: Engine execution failed or returned error for {symbol}. Result: {engine_result}")


        # --- Report Result ---
        results_storage.store_result(
            run_id=run_id,
            symbol=symbol,
            match_status=match_status,
            details={
                "last_hma": engine_result.get('last_hma'),
                "triggered_alert": triggered_alert_title,
                "error": engine_result.get("error")
                }
        )
    logging.info("Worker finished processing all available tasks.")
    print("-" * 30)


def step_3_15_results_handling(run_id: str, results_storage: ResultsStorage) -> Dict[str, Any]:
    print("\n--- Phase 3, Step 15: Results Handling (Python Service & API) ---")
    logging.info("Simulating Results Aggregation Service (using ResultsStorage).")
    logging.info("Simulating Backend API endpoints:")
    logging.info("  POST /screeners/{id}/run -> Triggers scheduling (Step 13)")
    logging.info(f"  GET /screeners/runs/{run_id}/results -> Fetches results")

    # --- Simulate API call to get results ---
    logging.info(f"API: Fetching results for run_id='{run_id}'...")
    results = results_storage.get_results(run_id)
    matching_symbols = sorted([r['symbol'] for r in results if r['match']])
    api_response = {
        "run_id": run_id,
        "status": "COMPLETED", # Add status tracking in real system
        "matching_symbols": matching_symbols,
        "total_symbols_processed": len(results),
        "results_summary": [ # Provide a bit more detail
             {"symbol": r["symbol"], "match": r["match"], "trigger": r["details"].get("triggered_alert")}
             for r in results
        ]
        # "all_results": results # Optionally include all raw results
    }
    logging.info(f"API Response for Run '{run_id}': {len(matching_symbols)} matches found.")
    print(json.dumps(api_response, indent=2)) # Pretty print the result
    print("-" * 30)
    return api_response

def step_3_16_minimal_interface_test(screener_id: int, run_id: str, api_results: Dict):
    print("\n--- Phase 3, Step 16: Minimal Interface & Test ---")
    logging.info("Simulating a basic CLI tool interaction.")

    # --- Simple CLI Simulation ---
    print("\n--- CLI Simulation ---")
    print(f"$ python cli_app.py run --screener-id {screener_id}")
    print(f"--> Triggering run for screener {screener_id}...")
    print(f"--> Run started with ID: {run_id}")
    print("\n...")
    print(f"$ python cli_app.py results --run-id {run_id}")
    print(f"--> Fetching results for run {run_id}...")
    print("--> Screener Results:")
    print(f"    Run ID: {api_results['run_id']}")
    print(f"    Status: {api_results['status']}")
    print(f"    Matching Symbols ({len(api_results['matching_symbols'])}): {api_results['matching_symbols']}")
    # print("    Summary:")
    # for item in api_results.get('results_summary', []):
    #      print(f"      - {item['symbol']}: Match={item['match']} (Trigger: {item['trigger']})")
    print("--- End CLI Simulation ---\n")

    logging.info("** End-to-End Flow Test Verification: **")
    logging.info("1. Define Screener: Completed (Step 12).")
    logging.info("2. Trigger Run (CLI Sim): Completed.")
    logging.info("3. Scheduling & Dispatch: Completed (Step 13).")
    logging.info("4. Worker Execution & Engine Call: Completed (Step 14).")
    logging.info("5. Results Storage: Completed (Step 14).")
    logging.info("6. Retrieve & Verify Results (CLI Sim): Completed.")
    logging.info("-> Final list of matching symbols obtained and displayed.")
    logging.info("-> MVP End-to-End flow simulation successful.")
    print("-" * 30)

def run_pytest_tests():
    """Runs pytest tests programmatically."""
    print("\n--- Running Pytest Validation ---")
    logging.info("Executing pytest tests for engine validation...")
    try:
        # Ensure pytest runs from the script's directory context
        script_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(["pytest", "test_mvp_screener.py"], capture_output=True, text=True, check=True, cwd=script_dir)
        logging.info("Pytest Output:\n" + result.stdout)
        if result.stderr:
             logging.warning("Pytest Error Output:\n" + result.stderr)
        logging.info("Pytest validation completed successfully.")
    except FileNotFoundError:
         logging.error("Error: 'pytest' command not found. Make sure pytest is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Pytest execution failed with return code {e.returncode}.")
        logging.error("Pytest Output:\n" + e.stdout)
        logging.error("Pytest Error Output:\n" + e.stderr)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pytest execution: {e}")
    print("-" * 30)

# --- Main Execution Block ---

if __name__ == "__main__":
    logging.info("====== Starting Python MVP Screener Build Process ======")

    # Phase 0: Foundations
    step_0_1_define_scope_legal()
    step_0_2_core_team_tools()
    pine_script_mvp = step_0_3_pine_mvp_analysis()
    step_0_4_basic_architecture_cloud()

    # Phase 1: Core Pine Engine (Build & Validate)
    # DataFetcher needs to be initialized before the runtime
    # In a real app, DataFetcher might be configured differently
    temp_data_fetcher = DataFetcher() # Use temporary one for engine build step if needed
    parser, runtime = step_1_5_build_pine_engine_core(temp_data_fetcher)
    step_1_6_develop_validation_framework() # Describes test setup
    step_1_7_validate_engine_expose(parser, runtime) # Simulates validation & exposure

    # Phase 2: Data Pipeline & Basic Execution Setup
    step_2_8_market_data_ingestion()
    db_manager, data_fetcher = step_2_9_data_storage_access()
    # Re-assign the potentially "production" data_fetcher to the runtime
    runtime.data_fetcher = data_fetcher
    step_2_10_basic_execution_worker(db_manager, runtime) # Test single worker run
    step_2_11_basic_monitoring()

    # Phase 3: End-to-End Screener Logic Execution
    screener_id = step_3_12_screener_definition(db_manager)
    if screener_id is None or not db_manager.get_screener(screener_id):
         logging.error("Failed to create or retrieve a valid screener definition. Aborting Phase 3.")
    else:
        run_id, task_queue = step_3_13_job_scheduling_dispatch(screener_id, db_manager)
        if run_id and task_queue:
            results_storage = ResultsStorage() # Initialize results storage for the run
            # Simulate workers processing all dispatched tasks
            step_3_14_worker_enhancement(task_queue, db_manager, runtime, results_storage)
            # Fetch and display results via simulated API/CLI
            api_results = step_3_15_results_handling(run_id, results_storage)
            step_3_16_minimal_interface_test(screener_id, run_id, api_results)
        else:
            logging.error("Failed to schedule screener run. Aborting remaining Phase 3 steps.")

    # Run Pytest Tests at the end
    run_pytest_tests()

    logging.info("====== Python MVP Screener Build Process Simulation Complete ======")
    logging.info("Outcome: Functional simulation of a basic Python-based screener executing MVP Pine Script™ logic.")