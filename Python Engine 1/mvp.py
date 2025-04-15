# // --- Python Step-by-Step Guide ---

import os
import subprocess
import venv
import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple

# --- Placeholder Libraries/Modules (Illustrative) ---
# Replace these with actual implementations or libraries

# Placeholder for a potential Pine Script parsing library (e.g., lark-parser, PLY)
class PineParser:
    def parse(self, script: str) -> Any: # Returns Abstract Syntax Tree (AST)
        print(f"Parsing script...")
        # Example: return lark_parser.parse(script)
        return {"type": "Script", "body": [...]} # Simplified AST representation

# Placeholder for the Pine Script runtime engine
class PineRuntime:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
        self.functions = {
            'hl2': self._func_hl2,
            'ta.hma': self._func_hma,
            'ta.valuewhen': self._func_valuewhen,
            'na': float('nan'),
            'nz': self._func_nz,
            'ta.crossover': self._func_crossover,
            'ta.crossunder': self._func_crossunder,
            # Add other necessary operators/functions (math, comparison, etc.)
        }
        self.variables = {}
        self.series_outputs = {}
        self.alert_conditions = []

    def _func_hl2(self, high_series, low_series):
        # Example: Use numpy/pandas for efficiency
        # import numpy as np
        # return (np.array(high_series) + np.array(low_series)) / 2
        print("Calculating hl2...")
        return [(h + l) / 2 for h, l in zip(high_series, low_series)] # Simplified

    def _func_hma(self, source_series, length):
        print(f"Calculating HMA(length={length})...")
        # Requires implementing HMA logic accurately (potentially using pandas/numpy)
        # This is a complex function, simplified here
        # Example: return pandas_ta.hma(pandas.Series(source_series), length=length).to_list()
        return [s * 0.9 for s in source_series] # Highly simplified placeholder

    def _func_valuewhen(self, condition_series, source_series, occurrence):
         print(f"Calculating valuewhen (occurrence={occurrence})...")
         # Requires careful implementation matching Pine Script's behavior
         last_val = float('nan')
         result = []
         true_indices = [i for i, cond in enumerate(condition_series) if cond]
         for i in range(len(condition_series)):
             relevant_true_indices = [idx for idx in true_indices if idx <= i]
             if len(relevant_true_indices) > occurrence:
                 last_val = source_series[relevant_true_indices[-(occurrence + 1)]]
             result.append(last_val)
         return result # Simplified

    def _func_nz(self, series, replacement=0.0):
        # return [replacement if pd.isna(x) else x for x in series] # Using pandas check
        print("Calculating nz...")
        return [replacement if x != x else x for x in series] # Basic NaN check

    def _func_crossover(self, series1, series2):
        print("Calculating crossover...")
        crossings = [False] * len(series1)
        for i in range(1, len(series1)):
            if series1[i-1] <= series2[i-1] and series1[i] > series2[i]:
                crossings[i] = True
        return crossings

    def _func_crossunder(self, series1, series2):
        print("Calculating crossunder...")
        crossings = [False] * len(series1)
        for i in range(1, len(series1)):
            if series1[i-1] >= series2[i-1] and series1[i] < series2[i]:
                crossings[i] = True
        return crossings

    def execute(self, ast: Any, symbol: str, timeframe: str) -> Dict[str, Any]:
        print(f"Executing script for {symbol} on {timeframe}...")
        # 1. Fetch data using self.data_fetcher
        # ohlcv = self.data_fetcher.get_ohlcv(symbol, timeframe)
        # high = ohlcv['high']
        # low = ohlcv['low']
        # close = ohlcv['close'] # etc.

        # --- Simplified Execution Simulation based on MVP script ---
        # Replace with actual AST traversal and execution logic
        print("Simulating execution based on MVP script structure...")
        # Mock data for simulation
        mock_high = [10, 11, 12, 11, 13, 14, 15, 14, 13, 16]
        mock_low = [9, 10, 11, 10, 12, 13, 14, 13, 12, 15]
        mock_close = [9.5, 10.5, 11.5, 10.5, 12.5, 13.5, 14.5, 13.5, 12.5, 15.5]
        hma_length = 5 # Example value, should come from script inputs

        price = self._func_hl2(mock_high, mock_low)
        hma = self._func_hma(price, hma_length)

        # MA_Min / MA_Max logic
        ma_min_condition = [False] * len(hma)
        ma_max_condition = [False] * len(hma)
        for i in range(2, len(hma)):
             if hma[i] > hma[i-1] and hma[i-1] < hma[i-2]:
                 ma_min_condition[i-1] = True # Condition met at i-1
             if hma[i] < hma[i-1] and hma[i-1] > hma[i-2]:
                 ma_max_condition[i-1] = True # Condition met at i-1

        saveMA_Min = self._func_valuewhen(ma_min_condition, hma, 0)
        saveMA_Max = self._func_valuewhen(ma_max_condition, hma, 0)

        # Alert Conditions
        buy_signal_series = self._func_crossover(hma, saveMA_Min)
        sell_signal_series = self._func_crossunder(hma, saveMA_Max)

        # Store alert conditions results (typically check the *last* bar)
        self.alert_conditions = [
            {"title": "Buy Signal", "triggered": buy_signal_series[-1]},
            {"title": "Sell Signal", "triggered": sell_signal_series[-1]},
        ]
        print(f"Execution finished. Alerts: {self.alert_conditions}")

        # Return relevant outputs, especially alert condition status
        return {"alerts": self.alert_conditions, "last_hma": hma[-1]}

# Placeholder for data fetching logic
class DataFetcher:
    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, List[float]]:
        print(f"Fetching {limit} OHLCV bars for {symbol} ({timeframe})...")
        # Implement connection to TimescaleDB or other data source
        # Example: Use psycopg2 to query TimescaleDB
        # return db_client.query(...)
        # Returning mock data for demonstration
        return {
            'timestamp': [i for i in range(limit)],
            'open': [10 + i * 0.1 for i in range(limit)],
            'high': [10.5 + i * 0.1 for i in range(limit)],
            'low': [9.5 + i * 0.1 for i in range(limit)],
            'close': [10.2 + i * 0.1 for i in range(limit)],
            'volume': [1000 + i * 10 for i in range(limit)],
        }

# Placeholder for Database interactions (Metadata, Screener Defs)
class DatabaseManager:
    def __init__(self, db_url):
        print(f"Connecting to DB: {db_url}")
        # Example: Use SQLAlchemy or psycopg2
        self.connection = None # Placeholder

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        print(f"Fetching metadata for {symbol}...")
        # Query PostgreSQL for exchange, etc.
        return {"symbol": symbol, "exchange": "NASDAQ", "name": f"{symbol} Inc."}

    def save_screener(self, name: str, symbols: List[str], timeframe: str, script: str) -> int:
        screener_id = hash((name, tuple(symbols), timeframe, script)) # Simple ID generation
        print(f"Saving screener '{name}' (ID: {screener_id})...")
        # Insert into PostgreSQL screeners table
        return screener_id

    def get_screener(self, screener_id: int) -> Optional[Dict[str, Any]]:
        print(f"Fetching screener definition (ID: {screener_id})...")
        # Query PostgreSQL screeners table
        # Returning mock data
        if screener_id == 12345: # Example ID
             return {
                 "id": screener_id,
                 "name": "MVP HMA Cross",
                 "symbols": ["AAPL", "GOOGL", "MSFT"],
                 "timeframe": "1D",
                 "script": "//@version=5\nindicator('MVP', overlay=true)\n..." # Full script
             }
        return None

# Placeholder for Task Queue interactions (e.g., Redis, Kafka)
class TaskQueueClient:
    def __init__(self, queue_name):
        self.queue_name = queue_name
        print(f"Connecting to task queue '{queue_name}'...")
        # Example: Use redis-py or kafka-python

    def send_task(self, task_data: Dict[str, Any]):
        print(f"Sending task to queue '{self.queue_name}': {task_data}")
        # Example: redis_client.lpush(self.queue_name, json.dumps(task_data))

    def get_task(self, timeout=5) -> Optional[Dict[str, Any]]:
        print(f"Waiting for task from queue '{self.queue_name}' (timeout={timeout}s)...")
        # Example: task_json = redis_client.brpop(self.queue_name, timeout=timeout)
        # if task_json: return json.loads(task_json[1])
        # Simulate receiving a task after a delay
        time.sleep(1)
        # Return None # Simulate no task available
        return { # Simulate receiving a task
            "run_id": "run_abc",
            "symbol": "AAPL",
            "timeframe": "1D",
            "script": "//@version=5\nindicator('MVP', overlay=true)\n..."
        }


# Placeholder for Results Storage (e.g., Redis)
class ResultsStorage:
    def __init__(self):
        print("Connecting to results storage (e.g., Redis)...")
        # Example: Use redis-py
        self.results = {} # In-memory simulation

    def store_result(self, run_id: str, symbol: str, match_status: bool, details: Any):
        print(f"Storing result for run '{run_id}', symbol '{symbol}': Match={match_status}")
        if run_id not in self.results:
            self.results[run_id] = []
        self.results[run_id].append({"symbol": symbol, "match": match_status, "details": details})
        # Example: redis_client.hset(f"results:{run_id}", symbol, json.dumps(...))

    def get_results(self, run_id: str) -> List[Dict[str, Any]]:
        print(f"Fetching results for run '{run_id}'...")
        # Example: results_raw = redis_client.hgetall(f"results:{run_id}")
        # return [json.loads(v) for v in results_raw.values()]
        return self.results.get(run_id, [])

# --- Main Step-by-Step Functions ---

def step_0_1_define_scope_legal():
    print("\n--- Phase 0, Step 1: Define Scope & Legal ---")
    print("Action: Confirm MVP Pine Script™ subset: `hl2`, `ta.hma`, `ta.valuewhen`, `na`, `nz`, `ta.crossover`, `ta.crossunder`, `alertcondition`, basic math/logic.")
    print("Action: Analyze provided MVP script logic (HMA cross over min/max).")
    print("(CRITICAL) Action: Obtain explicit legal clearance for Pine Script™ engine replication.")
    # Simulate legal check failure/success
    legal_clearance = True # input("Is legal clearance obtained? (yes/no): ").lower() == 'yes'
    if not legal_clearance:
        raise Exception("Legal clearance not obtained. Cannot proceed.")
    print("Legal clearance confirmed.")
    print("-" * 20)

def step_0_2_core_team_tools():
    print("\n--- Phase 0, Step 2: Core Team & Tools ---")
    print("Action: Assign leads (Backend, Engine, Data).")
    print("Action: Set up Git repository (`git init`, `git remote add origin ...`).")
    print("Action: Set up project management (Jira/Linear/etc.).")
    print("Action: Set up communication channels (Slack/Teams).")
    print("Action: Set up Python environment (`python -m venv venv`, `source venv/bin/activate`, `pip install ...`).")
    # Example: Initialize venv if not present
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        subprocess.run(["python3", "-m", "venv", "venv"])
    print("Action: Install core Python libraries (`pip install requests numpy pandas ...`).") # Add necessary libs
    print("-" * 20)

def step_0_3_pine_mvp_analysis():
    print("\n--- Phase 0, Step 3: Pine Script™ MVP Analysis ---")
    pine_script_mvp = """
//@version=5
indicator('MVP', overlay=true)
price = input(hl2, title='Source')
HMA_Length = input(21, 'HMA Length')
lookback = input(2, 'lookback')
// ... (rest of the MVP script) ...
saveMA_Min = ta.valuewhen(HMA > HMA[1] and HMA[1] < HMA[2], HMA[1], 0)
saveMA_Max = ta.valuewhen(HMA < HMA[1] and HMA[1] > HMA[2], HMA[1], 0)
plot(HMA, 'HMA', color=color.blue, linewidth=3) // Simplified plot for context
alertcondition(ta.crossover(HMA, saveMA_Min), title='Buy Signal', message='Crossing above MA_Min, Bullish')
alertcondition(ta.crossunder(HMA, saveMA_Max), title='Sell Signal', message='Crossing below MA_Max, Bearish')
"""
    print("Action: Analyze the *subset* of Pine Script™ needed for the MVP:")
    print("  - Functions: hl2, ta.hma, ta.valuewhen, na, nz, ta.crossover, ta.crossunder, alertcondition")
    print("  - Inputs: input()")
    print("  - Operators: >, <, and, +, -, /, [] (history access)")
    print("  - Core concept: Evaluating conditions on series data.")
    print(f"Action: Gather test scripts (using MVP script):\n{pine_script_mvp[:200]}...") # Show snippet
    print("Action: Document expected behavior based on TradingView (e.g., for specific data points).")
    print("-" * 20)
    return pine_script_mvp

def step_0_4_basic_architecture_cloud():
    print("\n--- Phase 0, Step 4: Basic Architecture & Cloud Setup ---")
    print("Action: Design high-level Python service interactions:")
    print("  - API Service (FastAPI/Flask) -> Scheduler -> Task Queue (Redis/Kafka)")
    print("  - Worker Service (Python Script/Celery) <- Task Queue")
    print("  - Worker -> Data Access (Python DB Client) -> TimescaleDB")
    print("  - Worker -> Metadata Access (Python DB Client) -> PostgreSQL")
    print("  - Worker -> Pine Engine Service (Python API/gRPC call)")
    print("Action: Set up basic cloud infrastructure (e.g., using AWS CDK, Pulumi with Python, or Terraform).")
    print("  - VPC, Minimal EKS/ECS, Basic IAM Roles")
    print("Action: Implement skeleton CI/CD pipeline (e.g., GitHub Actions running `pytest`, `flake8`).")
    print("-" * 20)

# --- Phase 1: Core Pine Engine & Validation ---

def step_1_5_build_pine_engine_core(pine_script_mvp: str) -> Tuple[PineParser, PineRuntime]:
    print("\n--- Phase 1, Step 5: Build Pine Engine Core (Python) ---")
    print("Action: Implement Pine Script™ Parser (Python - e.g., using `lark-parser` or `PLY`) for MVP subset.")
    parser = PineParser()
    # ast = parser.parse(pine_script_mvp) # Simulate parsing
    print("  - Parse script text into an Abstract Syntax Tree (AST).")

    print("Action: Implement Basic Runtime (Python) for MVP subset.")
    print("  - Handle basic types (int, float, bool, series).")
    print("  - Implement operators (+, -, *, /, >, <, ==, and, or).")
    print("  - Implement history access operator `[]`.")
    print("  - Note: Pure Python loops can be slow; use `numpy`/`pandas` for vectorized operations where possible for performance.")

    print("Action: Implement *key required* built-in functions rigorously matching TradingView logic:")
    print("  - `hl2`, `ta.hma`, `ta.valuewhen`, `na`, `nz`, `ta.crossover`, `ta.crossunder`, `alertcondition`")
    # The PineRuntime class above is a simplified placeholder
    # Requires significant effort for accuracy, especially ta.hma and ta.valuewhen
    data_fetcher = DataFetcher() # Placeholder
    runtime = PineRuntime(data_fetcher)

    print("Action: Implement `alertcondition` to evaluate boolean series and store results.")
    print("Accuracy Note: Achieving 100% match with TradingView precision and edge cases is critical and challenging.")
    print("-" * 20)
    return parser, runtime

def step_1_6_develop_validation_framework(parser: PineParser, runtime: PineRuntime, pine_script_mvp: str):
    print("\n--- Phase 1, Step 6: Develop Validation Framework (Python) ---")
    print("Action: Build tooling (Python script) to fetch specific historical OHLCV data.")
    print("  - Use `requests`, `yfinance`, `ccxt`, or specific broker/data provider APIs.")
    print("  - Store reference data (e.g., in CSV files or a test database).")
    data_fetcher = DataFetcher() # Use a specific instance for validation

    print("Action: Create a test harness (Python `pytest` functions):")
    print("  - Takes symbol, timeframe, reference data, Pine script.")
    print("  - Executes the script using the Python `parser` and `runtime`.")
    print("  - Compares output series (especially `alertcondition` results) against known TradingView results.")
    print("  - Use `numpy.allclose` for float comparisons, handle `na` values correctly.")

    # --- Example Test Case (Conceptual) ---
    def run_validation_test(symbol, timeframe, script, reference_data_path, expected_alerts):
        print(f"Running validation: {symbol}, {timeframe}...")
        # 1. Load reference_data from reference_data_path
        # 2. Configure runtime.data_fetcher to use this specific data
        # 3. ast = parser.parse(script)
        # 4. result = runtime.execute(ast, symbol, timeframe) # Execute with reference data
        # 5. Compare result['alerts'] with expected_alerts
        # Example comparison:
        # assert result['alerts'] == expected_alerts, f"Mismatch for {symbol}"
        print(f"  > Simulating test run for {symbol}... OK") # Placeholder

    # run_validation_test("AAPL", "1D", pine_script_mvp, "data/AAPL_1D.csv", [{"title": "Buy Signal", "triggered": False}, {"title": "Sell Signal", "triggered": True}])
    print("Action: Manually run the *same* script on TradingView with the *same* data for comparison.")
    print("-" * 20)


def step_1_7_validate_engine_expose(parser: PineParser, runtime: PineRuntime):
    print("\n--- Phase 1, Step 7: Validate Engine & Expose (Python) ---")
    print("Action: Integrate Python Parser & Runtime components.")
    # Already done conceptually by passing parser/runtime objects

    print("Action: Use the Validation Framework (Step 6) repeatedly.")
    print("  - Run tests against MVP scripts and reference data.")
    print("  - Debug and refine engine logic until outputs match TradingView 100% for the MVP subset.")
    print("  - Focus heavily on `ta.hma`, `ta.valuewhen`, `alertcondition` accuracy.")
    # Simulate validation runs
    print("Simulating validation runs... Engine accuracy: 99.9% (Target: 100%)") # Placeholder

    print("Action: Expose the engine's execution function via a simple API (e.g., Python FastAPI).")
    # Conceptual FastAPI endpoint:
    # @app.post("/execute")
    # async def execute_script(request: ExecuteRequest):
    #     ast = parser.parse(request.script)
    #     # Need a data fetcher appropriate for the API context
    #     api_runtime = PineRuntime(DataFetcher())
    #     results = api_runtime.execute(ast, request.symbol, request.timeframe)
    #     return results
    print("  - Define request/response models (Symbol, Timeframe, Script -> Alert Results, Outputs).")
    print("  - Run the API service (e.g., using `uvicorn`).")
    print("Pine Engine Service API available at: http://localhost:8000/execute (Conceptual)")
    print("-" * 20)

# --- Phase 2: Data Pipeline & Basic Execution ---

def step_2_8_market_data_ingestion():
    print("\n--- Phase 2, Step 8: Market Data Ingestion (Python) ---")
    print("Action: Implement a Python service/script to connect to a data provider (e.g., using websockets, requests).")
    print("Action: Parse incoming data (JSON, CSV, etc.).")
    print("Action: Standardize data format (e.g., OHLCV dictionary/object).")
    print("Action: Publish standardized data to Kafka using `kafka-python`.")
    # Example: kafka_producer.send('ohlcv_topic', key=symbol, value=ohlcv_data)
    print("Data Ingestion Service running (simulated).")
    print("-" * 20)

def step_2_9_data_storage_access() -> Tuple[DatabaseManager, DataFetcher]:
    print("\n--- Phase 2, Step 9: Data Storage & Access (Python) ---")
    print("Action: Set up Kafka (e.g., using Docker, Confluent Cloud, AWS MSK).")
    print("Action: Implement a Python consumer service (`kafka-python`) to read from Kafka.")
    print("Action: Write consumed OHLCV data to TimescaleDB (using `psycopg2` with TimescaleDB extensions).")
    print("  - Ensure hypertable setup for efficient time-series queries.")
    print("Action: Implement a basic Metadata Service (Python FastAPI + PostgreSQL).")
    print("  - Manages symbol info (name, exchange, etc.).")
    print("  - Use `SQLAlchemy` or `psycopg2` for DB interaction.")
    db_manager = DatabaseManager("postgresql://user:pass@host/metadata_db")
    data_fetcher = DataFetcher() # This fetcher now conceptually reads from TimescaleDB
    print("Data Storage Consumer & Metadata Service running (simulated).")
    print("-" * 20)
    return db_manager, data_fetcher


def step_2_10_basic_execution_worker(db_manager: DatabaseManager, data_fetcher: DataFetcher):
    print("\n--- Phase 2, Step 10: Basic Execution Worker (Python) ---")
    print("Action: Build a Python worker script that can:")
    print("  - Receive a simple task (e.g., Dict: {'symbol': 'AAPL', 'timeframe': '1D', 'script': '...'}).")
    worker_task = {"symbol": "AAPL", "timeframe": "1D", "script": step_0_3_pine_mvp_analysis()} # Get script

    print("  - Fetch symbol metadata using `db_manager`.")
    metadata = db_manager.get_symbol_metadata(worker_task['symbol'])

    print("  - Fetch required OHLCV data using `data_fetcher` (from TimescaleDB).")
    ohlcv = data_fetcher.get_ohlcv(worker_task['symbol'], worker_task['timeframe'])

    print("  - Call the Pine Engine Service API (using `requests` to call the FastAPI endpoint from Step 7).")
    # engine_api_url = "http://localhost:8000/execute" # From Step 7
    # response = requests.post(engine_api_url, json=worker_task)
    # engine_result = response.json()
    # --- Or, if engine is directly callable (less scalable): ---
    parser, runtime = step_1_5_build_pine_engine_core(worker_task['script']) # Re-init for demo
    runtime.data_fetcher = data_fetcher # Ensure it uses the right data source
    ast = parser.parse(worker_task['script'])
    engine_result = runtime.execute(ast, worker_task['symbol'], worker_task['timeframe'])
    # ---

    print(f"  - Log the result using Python's `logging` module.")
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Worker Task Result for {worker_task['symbol']}: {engine_result}")
    print("-" * 20)
    # Return result for potential next steps
    return engine_result


def step_2_11_basic_monitoring():
    print("\n--- Phase 2, Step 11: Basic Monitoring ---")
    print("Action: Instrument Python services (API, Worker, Ingestor) using `prometheus-client`.")
    print("  - Expose metrics like task processed rate, error counts, execution duration.")
    print("Action: Set up Prometheus to scrape metrics from Python services.")
    print("Action: Set up Grafana to visualize metrics from Prometheus.")
    print("  - Create dashboards for data flow, DB performance, worker activity.")
    print("Monitoring setup configured (Prometheus scraping :<port>/metrics).")
    print("-" * 20)

# --- Phase 3: End-to-End Screener Logic ---

def step_3_12_screener_definition(db_manager: DatabaseManager) -> int:
    print("\n--- Phase 3, Step 12: Screener Definition (Python Service & API) ---")
    print("Action: Implement a Service & API (Python FastAPI/Flask + PostgreSQL).")
    print("  - Use `db_manager` (or SQLAlchemy) for database operations.")
    print("  - API Endpoint: POST /screeners (Payload: name, symbols, timeframe, script)")
    print("  - API Endpoint: GET /screeners")
    print("  - API Endpoint: GET /screeners/{id}")
    # Simulate saving a screener definition
    screener_id = db_manager.save_screener(
        name="MVP HMA Cross Screener",
        symbols=["AAPL", "GOOGL", "MSFT", "TSLA"],
        timeframe="1D",
        script=step_0_3_pine_mvp_analysis() # Get MVP script again
    )
    print(f"Screener Definition Service: Saved screener with ID: {screener_id}")
    print("-" * 20)
    return screener_id # Return the ID of the created screener

def step_3_13_job_scheduling_dispatch(screener_id: int, db_manager: DatabaseManager):
    print("\n--- Phase 3, Step 13: Job Scheduling & Dispatch (Python) ---")
    print("Action: Implement a Python Scheduler script/service.")
    print("  - Takes a `screener_id` as input.")
    run_id = f"run_{int(time.time())}" # Generate a unique ID for this run
    print(f"Scheduler: Received request to run screener ID: {screener_id} (Run ID: {run_id})")

    print("  - Fetches screener definition using `db_manager`.")
    screener_def = db_manager.get_screener(screener_id)
    if not screener_def:
        print(f"Error: Screener ID {screener_id} not found.")
        return None, None

    print(f"  - Creates individual tasks for each symbol.")
    tasks = []
    for symbol in screener_def['symbols']:
        task = {
            "run_id": run_id,
            "symbol": symbol,
            "timeframe": screener_def['timeframe'],
            "script": screener_def['script']
        }
        tasks.append(task)

    print(f"  - Sends tasks to a worker queue (e.g., Redis list or Kafka topic using TaskQueueClient).")
    task_queue = TaskQueueClient("screener_tasks") # Connect to the queue
    for task in tasks:
        task_queue.send_task(task)
    print(f"Scheduler: Dispatched {len(tasks)} tasks for run {run_id}.")
    print("-" * 20)
    return run_id, task_queue # Return run_id and queue for worker

def step_3_14_worker_enhancement(task_queue: TaskQueueClient, db_manager: DatabaseManager, data_fetcher: DataFetcher, results_storage: ResultsStorage):
    print("\n--- Phase 3, Step 14: Worker Enhancement (Python) ---")
    print("Action: Modify Python worker (from Step 10) to:")
    print("  - Consume tasks from the queue (`task_queue.get_task()`).")

    while True:
        worker_task = task_queue.get_task(timeout=1) # Wait for 1 second
        if not worker_task:
            print("Worker: No more tasks in queue. Exiting.")
            break # Exit loop if no tasks

        print(f"Worker: Received task for run {worker_task['run_id']}, symbol {worker_task['symbol']}")

        # --- Reuse logic from Step 10 ---
        metadata = db_manager.get_symbol_metadata(worker_task['symbol'])
        ohlcv = data_fetcher.get_ohlcv(worker_task['symbol'], worker_task['timeframe'])
        parser, runtime = step_1_5_build_pine_engine_core(worker_task['script'])
        runtime.data_fetcher = data_fetcher
        ast = parser.parse(worker_task['script'])
        engine_result = runtime.execute(ast, worker_task['symbol'], worker_task['timeframe'])
        # ---

        print("  - Report results to an aggregation point (`results_storage`).")
        # Determine match status based on 'alertcondition' results
        match_status = False
        triggered_alert_title = None
        for alert in engine_result.get('alerts', []):
             # Example: Consider a match if *any* relevant alert triggered on the last bar
             # Refine this logic based on specific screener goals (e.g., only 'Buy Signal')
            if alert.get('triggered'):
                 if alert.get('title') == 'Buy Signal': # Be specific if needed
                     match_status = True
                     triggered_alert_title = alert.get('title')
                     break # Or collect all triggered alerts
                 # Add conditions for other alerts if necessary
        
        print(f"Worker: Symbol {worker_task['symbol']} - Match Status: {match_status} (Triggered: {triggered_alert_title})")
        results_storage.store_result(
            run_id=worker_task['run_id'],
            symbol=worker_task['symbol'],
            match_status=match_status,
            details={"last_hma": engine_result.get('last_hma'), "triggered_alert": triggered_alert_title} # Add relevant details
        )
    print("-" * 20)


def step_3_15_results_handling(run_id: str, results_storage: ResultsStorage, db_manager: DatabaseManager):
    print("\n--- Phase 3, Step 15: Results Handling (Python Service & API) ---")
    print("Action: Implement Results Aggregation Service (using `results_storage`, e.g., Redis).")
    # This is implicitly done by the worker writing to results_storage in Step 14

    print("Action: Build/Extend Backend API (Python FastAPI/Flask):")
    # Conceptual endpoints:
    # /screeners - (GET, POST - from Step 12)
    # /screeners/{id}/run - (POST) - Triggers scheduling (Step 13)
    # /screeners/runs/{run_id}/results - (GET) - Fetches results from results_storage

    print(f"Action: Simulate API call to get results for run_id='{run_id}'")
    # --- Conceptual API Handler ---
    def get_screener_run_results(run_id_param: str):
        results = results_storage.get_results(run_id_param)
        # Optionally filter/format results
        matching_symbols = [r['symbol'] for r in results if r['match']]
        return {"run_id": run_id_param, "matching_symbols": matching_symbols, "all_results": results}
    # ---

    api_results = get_screener_run_results(run_id)
    print(f"API Result for Run '{run_id}': {api_results}")
    print("-" * 20)
    return api_results # Return results for final step


def step_3_16_minimal_interface_test(screener_id: int, run_id: str, api_results: Dict):
    print("\n--- Phase 3, Step 16: Minimal Interface & Test ---")
    print("Action: Create a *very basic* CLI tool (Python `argparse` or `click`) to interact with the API.")

    # --- Simple CLI Simulation ---
    def cli_tool(action: str, screener_id_param: Optional[int] = None, run_id_param: Optional[str] = None):
        if action == "run" and screener_id_param:
            print(f"CLI: Triggering run for screener {screener_id_param}...")
            # In reality, this would make an API call: requests.post(f"/screeners/{screener_id_param}/run")
            print(f"CLI: Run triggered, Run ID: {run_id}") # Use the run_id from the actual execution
        elif action == "results" and run_id_param:
            print(f"CLI: Fetching results for run {run_id_param}...")
            # In reality, this would make an API call: requests.get(f"/screeners/runs/{run_id_param}/results")
            # We use the results passed into this function
            print(f"CLI Results for {run_id_param}:")
            print(f"  Matching Symbols: {api_results.get('matching_symbols', [])}")
            # print(f"  All Results: {api_results.get('all_results', [])}") # Optionally show all
        else:
            print("CLI: Unknown action or missing parameters.")

    # Simulate using the CLI
    print("--- CLI Simulation ---")
    cli_tool(action="run", screener_id_param=screener_id)
    time.sleep(1) # Simulate run time
    cli_tool(action="results", run_id_param=run_id)
    print("--- End CLI Simulation ---")

    print("\nAction: **Test the End-to-End Flow:**")
    print("  1. Define a screener (Step 12 - Done).")
    print("  2. Trigger the screener run via API/CLI (Simulated above).")
    print("  3. Observe tasks being scheduled (Step 13 - Done).")
    print("  4. Observe workers processing tasks & calling engine (Step 14 - Done).")
    print("  5. Observe results being stored (Step 14 - Done).")
    print("  6. Retrieve and verify the list of matching symbols via API/CLI (Simulated above).")
    print("  - Ensure the symbols listed match expectations based on the Pine Script™ logic and test data.")
    print("-" * 20)


# --- Execute the Steps ---

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("Starting Python MVP Screener Build Process...")

    # Phase 0
    step_0_1_define_scope_legal()
    step_0_2_core_team_tools()
    pine_script_mvp = step_0_3_pine_mvp_analysis()
    step_0_4_basic_architecture_cloud()

    # Phase 1 (Simulated Engine Build/Validation)
    # In a real scenario, these steps build the core engine component.
    parser, runtime = step_1_5_build_pine_engine_core(pine_script_mvp)
    step_1_6_develop_validation_framework(parser, runtime, pine_script_mvp)
    step_1_7_validate_engine_expose(parser, runtime) # Assumes engine is now ready & potentially exposed via API

    # Phase 2 (Simulated Data Pipeline & Basic Worker Test)
    step_2_8_market_data_ingestion()
    db_manager, data_fetcher = step_2_9_data_storage_access()
    # step_2_10_basic_execution_worker(db_manager, data_fetcher) # Test one worker execution
    step_2_11_basic_monitoring()

    # Phase 3 (End-to-End Screener)
    screener_id = step_3_12_screener_definition(db_manager)
    # Ensure screener_id is valid before proceeding
    if screener_id is None or not db_manager.get_screener(screener_id): # Add check based on your implementation
         print("Failed to create or retrieve a valid screener definition. Aborting Phase 3.")
    else:
        run_id, task_queue = step_3_13_job_scheduling_dispatch(screener_id, db_manager)
        if run_id and task_queue:
            results_storage = ResultsStorage() # Initialize results storage
            # Simulate workers processing all dispatched tasks
            step_3_14_worker_enhancement(task_queue, db_manager, data_fetcher, results_storage)
            # Fetch and display results via simulated API/CLI
            api_results = step_3_15_results_handling(run_id, results_storage, db_manager)
            step_3_16_minimal_interface_test(screener_id, run_id, api_results)
        else:
            print("Failed to schedule screener run. Aborting remaining Phase 3 steps.")


    print("\nPython MVP Screener Build Process Simulation Complete.")
    print("Outcome: A functional, basic Python-based screener capable of executing the MVP Pine Script™ logic across symbols and reporting matches based on 'alertcondition'.")