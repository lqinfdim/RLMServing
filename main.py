import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import sys
import argparse
import asyncio
import time
import traceback
from datetime import datetime
import aiohttp
import json # Added for logging args

# Import components from other modules
from asu_perf_benchmark_suite.metrics import MetricsCollector
from asu_perf_benchmark_suite.datasets import get_benchmark_class, BENCHMARK_CLASSES # Factory function
from asu_perf_benchmark_suite.utils import report_results

# Set default HuggingFace endpoint if necessary


def get_args():
    """Parses command-line arguments for the benchmark."""
    parser = argparse.ArgumentParser(description='Run LLM benchmarks.')

    # --- Core Arguments ---
    parser.add_argument('--dataset', type=str, required=True, choices=list(BENCHMARK_CLASSES.keys()), # Use keys from dict + variants
                        help='Name of the dataset to benchmark.')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
                        help='Name of the model being benchmarked (passed to API).')
    parser.add_argument('--output_path', type=str, default='./outputs/benchmark_output',
                        help='Base path for the raw output file (dataset name and timestamp will be added).')
    parser.add_argument('--csv_path', type=str, default='./logs/benchmark_results.csv',
                        help='Path to the CSV file for logging benchmark results.')
    parser.add_argument('--run_name', type=str, default="default",
                        help="A name for this specific run (for logging).")

    # --- API Connection ---
    parser.add_argument('--url', type=str, default="127.0.0.5",
                        help='Hostname or IP address for the API service.')
    parser.add_argument('--port', type=int, default=8005,
                        help='Port number for the API service.')
    parser.add_argument('--request_timeout', type=int, default=600,
                        help='Timeout in seconds for each individual API request.')
    parser.add_argument('--api_key', type=str, default=os.getenv("RLLM_API_KEY", "rllm-key"),
                        help='API key used for Authorization header when calling the serving endpoint.')

    # --- Benchmark Control ---
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing questions.')
    parser.add_argument('--capacity', type=int, default=100,
                        help='Number of data points from the dataset to use (-1 for all).')
    # repeat 3 times
    parser.add_argument('--repeat_time', type=int, default=3,
                        help='Number of times to repeat processing each question.')
    parser.add_argument('--benchmark_performance_only', type=bool, default=False,
                        help='If set, only measure performance metrics, skip accuracy calculation.')

    # --- Model Generation Parameters ---
    
    # new version temperature 0.6 , not the original 0.0
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for language model generation.")
    # new version top_p 0.95, not the original 1.0
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p probability for language model generation.")
    # new version max_token_num 8192
    parser.add_argument('--max_token_num', type=int, default=8192,
                        help='Max number of tokens in API requests (adjust per model/task).')

    # --- Reasoning/CoT Flags ---
    parser.add_argument('--is_reasoning_llm', type=bool, default=True,
                         help='Expect reasoning_content field, adjust token counting and TTFT.')
    parser.add_argument('--is_cot_visible', type=bool, default=False, 
                        help='Only relevant if --is_reasoning_llm. If set, TTFT uses content token only.')

    # --- Dataset Specific ---
    parser.add_argument('--prompt_path', type=str, default=None,
                        help='Path to a custom prompt file (overrides dataset default if provided).')
    parser.add_argument('--dataset_split', type=str, default='test',
                        help='Dataset split to use (e.g., train, test, validation).')
    parser.add_argument('--hf_token', type=str, default="token_xxxxxxxxxxxxx",
                        help='Hugging Face API token (if required for dataset access).')

    # --- Other ---
    parser.add_argument("--serving_framework", type=str, default="sglang", choices=["vllm", "sglang"],
                        help="Serving framework used (for logging).")

    args = parser.parse_args()

    # --- Dynamic Path Generation ---
    # (Moved from the previous main function to keep it with arg processing)
    try:
        # Insert dataset name into the default path if not already specified uniquely
        if args.output_path == './outputs/benchmark_output':
             dir_name = './outputs'
             # Use the user-provided dataset name for the path
             base_name = f"{args.dataset}_output_raw.txt"
             args.output_path = os.path.join(dir_name, base_name)
             print(f"Default output path adjusted for dataset: {args.output_path}")

        # Add timestamp and run_name
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        dir_name = os.path.dirname(args.output_path)
        base_name = os.path.basename(args.output_path)
        name, ext = os.path.splitext(base_name)
        if not ext: 
            ext = ".txt"; 
        name = base_name
        new_base_name = f"{name}_{args.run_name}_{timestamp}{ext}"
        args.output_path = os.path.join(dir_name, new_base_name)
        print(f"Timestamped output file path: {args.output_path}")

    except Exception as e:
        print(f"Warning: Could not format output path '{args.output_path}'. Error: {e}")

    # Dynamic CSV Path
    if args.csv_path == './logs/benchmark_results.csv':
        args.csv_path = f'./logs/{args.serving_framework}/benchmark_{args.dataset}_{args.run_name}_results.csv'
        print(f"Default CSV path adjusted for run name: {args.csv_path}")
    elif args.csv_path == './pilot/benchmark_results.csv':
        args.csv_path = f'./pilot/logs/{args.serving_framework}/benchmark_{args.dataset}_{args.run_name}_results.csv'
        

    # Create directories
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)

    # Handle GPQA dataset name variations for class lookup
    if args.dataset.startswith("gpqa_"):
        args.dataset_name_for_class = "gpqa"
        # Keep original name for loading within the class
        args.dataset_name = args.dataset # Store the original name if needed later
    else:
        args.dataset_name_for_class = args.dataset
        # args.dataset_name = args.dataset # Already set

    return args

class BenchmarkRunner:
    """Orchestrates the benchmark execution flow."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.metrics_collector = MetricsCollector()
        self.benchmark_instance = self._create_benchmark_instance()

    def _create_benchmark_instance(self):
        """Factory method to create the dataset-specific benchmark instance."""
        BenchmarkClass = get_benchmark_class(self.args.dataset_name_for_class)
        if BenchmarkClass is None:
            print(f"Error: Unsupported dataset '{self.args.dataset}'. Available: {list(BENCHMARK_CLASSES.keys())}")
            sys.exit(1)
        print(f"\n--- Initializing benchmark for dataset: {self.args.dataset} ---")
        return BenchmarkClass(self.args, self.metrics_collector)

    def _setup_output_file(self, start_time_str: str) -> bool:
        """Checks writability and writes header to the output file."""
        try:
             with open(self.args.output_path, 'w', encoding='utf-8') as fd_check:
                  fd_check.write(f"# Benchmark Run Start: {start_time_str}\n")
                  fd_check.write(f"# Run Name: {self.args.run_name}\n")
                  fd_check.write(f"# Dataset: {self.args.dataset}\n")
                  args_dict = vars(self.args)
                  args_str = json.dumps(args_dict, indent=2, sort_keys=True)
                  fd_check.write(f"# Args:\n{args_str}\n\n")
             print(f"Checked writability and wrote header to: {self.args.output_path}")
             return True
        except IOError as e:
             print(f"Error: Cannot write header to output file {self.args.output_path}: {e}")
             return False
        except Exception as e:
             print(f"Error during output file header check: {e}")
             return False

    async def run_benchmark(self):
        """Executes the complete benchmark process."""
        print(f"Run Name: {self.args.run_name}")
        print(f"Model: {self.args.model_name}, Dataset: {self.args.dataset}, Batch Size: {self.args.batch_size}, Capacity: {self.args.capacity}")
        print(f"Temperature: {self.args.temperature}, Top_p: {self.args.top_p}, Request Timeout: {self.args.request_timeout}s")
        print(f"Token Budget: {self.args.max_token_num}")
        print(f"Performance Only Mode: {self.args.benchmark_performance_only}")
        print(f"Reasoning LLM Mode: {self.args.is_reasoning_llm}")
        if self.args.is_reasoning_llm:
            # Note: is_cot_visible=True means default (flag absent), considers faster token
            # is_cot_visible=False means flag present, considers only content token
            print(f"CoT Visible (for TTFT): {self.args.is_cot_visible} (True means faster of R/C, False means Content only)")
        print(f"Outputting raw responses to: {self.args.output_path}") # Path now includes timestamp
        print(f"Logging results to CSV: {self.args.csv_path}")
        print(f"LLM API is aviable at : http://{self.args.url}:{self.args.port}")
        
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Use consistent start time

        if not self._setup_output_file(start_time_str):
            sys.exit(1)

        if not self.benchmark_instance: # Should have exited in constructor if failed
             print("Error: Benchmark instance not created.")
             sys.exit(1)

        print(f"\n--- Preparing benchmark for dataset: {self.args.dataset} ---")
        if not self.benchmark_instance.prepare_benchmark():
            print(f"Error: Failed to prepare benchmark for dataset '{self.args.dataset}'. Exiting.")
            sys.exit(1)
        print("--- Benchmark preparation complete ---")

        # --- Run Benchmark Loop ---
        print(f"\n--- Starting benchmark API calls for: {self.args.dataset} ---")
        stats = {}
        run_error = False

        connector = aiohttp.TCPConnector(limit_per_host=self.args.batch_size)
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                # Open the output file in append mode
                with open(self.args.output_path, 'a', encoding='utf-8') as fd:
                    stats = await self.benchmark_instance.run_benchmark(session, fd)
            except IOError as e:
                 print(f"Error writing to output file during run: {self.args.output_path}: {e}")
                 run_error = True
            except Exception as e:
                 print(f"Critical error during benchmark run: {e}")
                 traceback.print_exc()
                 run_error = True

        # If there was an error during the run, get collected stats and force perf reporting
        if run_error:
             stats = self.metrics_collector.calculate_statistics()
             self.args.benchmark_performance_only = True
             print("Warning: Benchmark run encountered errors. Reporting performance metrics only.")

        # --- Parse and Evaluate ---
        num_parsed = 0
        num_correct = None

        if not self.args.benchmark_performance_only and stats.get("total_requests", 0) > 0:
            print(f"\n--- Parsing and Evaluating Results for: {self.args.dataset} ---")
            try:
                num_parsed, num_correct = self.benchmark_instance.parse_and_evaluate()
            except Exception as eval_err:
                print(f"Error during parsing and evaluation: {eval_err}")
                traceback.print_exc()
                num_parsed = 0
                num_correct = 'Error'
        elif self.args.benchmark_performance_only:
             print("\n--- Skipping Accuracy Evaluation (Performance Only Mode) ---")
             num_parsed = stats.get("total_requests", 0)
             num_correct = 'N/A (Perf Only)'
        else:
             print("\n--- Skipping Accuracy Evaluation (No requests processed) ---")
             num_parsed = 0
             num_correct = 0

        # --- Report Results ---
        print(f"\n--- Reporting Results ---")
        try:
            report_results(stats, self.args, num_parsed, num_correct)
        except Exception as report_err:
            print(f"Error during results reporting: {report_err}")
            traceback.print_exc()


async def main_async():
    """Asynchronous main function to setup and run the benchmark."""
    program_start_time = time.time()
    print(f"--- Benchmark Run Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    args = get_args()
    runner = BenchmarkRunner(args)

    try:
        await runner.run_benchmark()
    except Exception as e:
        print(f"\nUnhandled error during benchmark execution: {e}")
        traceback.print_exc()
        # Optionally log minimal info to CSV on critical failure
    finally:
        program_end_time = time.time()
        print(f"\n--- Benchmark Run Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"--- Total Execution Time: {program_end_time - program_start_time:.2f} seconds ---")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as main_err:
        # This catches errors during asyncio.run itself or potentially unhandled exceptions
        print(f"\nCritical error setting up or running the benchmark: {main_err}")
        traceback.print_exc()
        sys.exit(1) # Exit with error code
