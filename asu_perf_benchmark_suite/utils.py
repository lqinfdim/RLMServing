# benchmark_utils.py

import csv
import os
import traceback
import time
from datetime import datetime
import argparse # Import argparse to use args type hint
from typing import Dict, Optional

def report_results(stats: Dict, args: argparse.Namespace, num_parsed: int, num_correct: Optional[int]):
    """Prints statistics to console and appends results to a CSV file."""

    # --- Calculate Accuracy ---
    accuracy_val_str = 'N/A'
    if isinstance(num_correct, int) and num_parsed > 0:
        accuracy_val = float(num_correct / num_parsed)
        accuracy_val_str = f"{accuracy_val:.4f}"
    elif args.benchmark_performance_only:
        accuracy_val_str = 'N/A (Perf Only)'
        num_correct = 'N/A (Perf Only)' # Set num_correct string for reporting
    else:
        # num_correct might be 0 or 'Error' from parsing
        pass # Keep num_correct as is, accuracy_val_str remains 'N/A' or 'Error'

    # --- Print Statistics ---
    print("\n=====================================")
    print(f"[INFO] Benchmark Results ({args.run_name})")
    print("=====================================")
    print(f"Model Name: {args.model_name}")
    print(f"Serving Framework: {args.serving_framework}")
    print(f"Dataset Name: {args.dataset}")
    print(f"Token Budget: {args.max_token_num}")
    if hasattr(args, 'dataset_split'): # GPQA has split
        print(f"Dataset Split: {args.dataset_split}")
    print(f"Total Questions Attempted (API calls): {stats.get('total_requests', 0)}")
    print(f"Questions Parsed from Output: {num_parsed}")
    print(f"Correct Answers (parsed): {num_correct}") # Will show N/A or Error if applicable
    print(f"Accuracy (Correct/Parsed): {accuracy_val_str}")

    print(f"Batch Size: {args.batch_size}")
    print(f"Repetitions: {args.repeat_time}")
    print(f"Performance Only Mode: {args.benchmark_performance_only}")
    print(f"Reasoning LLM Mode: {args.is_reasoning_llm}")
    if args.is_reasoning_llm:
        print(f"CoT Visible (for TTFT): {args.is_cot_visible} (True=Faster R/C, False=Content Only)")
    print(f"Temperature: {args.temperature}, Top_p: {args.top_p}, Request Timeout: {args.request_timeout}s")
    print("-------------------------------------")

    print("\nRequest-level Metrics (Averages):")
    print(f"Requests per second: {stats.get('requests_per_second', 0):.2f}")
    print(f"Tokens per second (total input + total output / e2e time): {stats.get('tokens_per_second', 0):.2f}")
    print(f"Average request latency: {stats.get('avg_request_latency', 0):.4f} s")
    print(f"Average TTFT (Time to First Token, calculated based on flags): {stats.get('avg_ttft', 0):.4f} s")
    print(f"Average TBT (Time Between Tokens, based on calculated TTFT): {stats.get('avg_tbt', 0):.4f} s")
    print("-------------------------------------")

    print("\nSystem-level Metrics:")
    print(f"Total Generation Time (first calculated token to last token): {stats.get('total_generation_time', 0):.2f} s")
    print(f"End-to-End Latency (first request start to last token): {stats.get('end_to_end_latency', 0):.2f} s")
    print(f"System Tokens per Second (total output tokens / gen time): {stats.get('system_tokens_per_second', 0):.2f}")
    print(f"System TPOT (Time Per Output Token, based on total output tokens): {stats.get('system_tpot', 0):.4f} s")
    print("-------------------------------------")

    print("\nToken Statistics (Totals):")
    print(f"Total Input Tokens (estimated): {stats.get('total_input_tokens', 0)}")
    print(f"Total Output Tokens (counted, includes reasoning based on flags): {stats.get('total_output_tokens', 0)}")
    if args.is_reasoning_llm:
        total_output = stats.get('total_output_tokens', 0)
        total_reasoning = stats.get('total_reasoning_tokens', 0)
        content_tokens = max(0, total_output - total_reasoning)
        print(f"-- Content Tokens (calculated): {content_tokens}")
        print(f"-- Reasoning Tokens (counted): {total_reasoning}")
    print("=====================================")

    # --- Write to CSV ---
    print(f"\nAppending results to CSV: {args.csv_path}")
    csv_file_path = args.csv_path
    # Define fieldnames consistently
    fieldnames = [
        'run_start_time', 'run_name', 'model_name', 'serving_framework',
        'dataset_name', 'dataset_split', # Added split (will be empty if not applicable)
        'capacity', 'repeat_time',
        'num_questions_processed', # Use total_requests from stats
        'num_questions_parsed',
        'num_correct', 'accuracy',
        'batch_size', 'performance_only', 'reasoning_llm', 'is_cot_visible',
        'temperature', 'top_p', 'request_timeout', "token_budget",
        'requests_per_second', 'tokens_per_second', 'avg_request_latency',
        'avg_ttft', 'avg_tbt', 'total_requests',
        'total_generation_time', 'end_to_end_latency',
        'system_tokens_per_second', 'system_tpot',
        'total_input_tokens', 'total_output_tokens', 'total_reasoning_tokens'
    ]

    # Prepare row data using calculated values and args
    row_data = {
        'run_start_time': datetime.fromtimestamp(stats.get('first_request_start_time', time.time())).strftime('%Y-%m-%d %H:%M:%S') if stats.get('first_request_start_time') else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': args.run_name ,
        'model_name': args.model_name,
        'serving_framework': args.serving_framework,
        'dataset_name': args.dataset,
        'dataset_split': getattr(args, 'dataset_split', ''), # Add split if exists
        'capacity': args.capacity,
        'repeat_time': args.repeat_time,
        'num_questions_processed': stats.get('total_requests', 0), # Total requests sent
        'num_questions_parsed': num_parsed,
        'num_correct': num_correct, # Use the value determined above (int, N/A, Error)
        'accuracy': accuracy_val_str, # Use the formatted string
        'batch_size': args.batch_size,
        'performance_only': args.benchmark_performance_only,
        'reasoning_llm': args.is_reasoning_llm,
        'is_cot_visible': args.is_cot_visible if args.is_reasoning_llm else 'N/A',
        'temperature': args.temperature,
        'top_p': args.top_p,
        'request_timeout': args.request_timeout,
        "token_budget": args.max_token_num,
        # Unpack the stats dictionary safely
        **{k: stats.get(k, '') for k in fieldnames if k in stats}
    }

    file_exists = os.path.isfile(csv_file_path)
    is_empty = (not file_exists) or (os.path.getsize(csv_file_path) == 0)

    try:
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if is_empty:
                writer.writeheader()

            formatted_row = {}
            for field in fieldnames:
                 value = row_data.get(field)
                 if isinstance(value, float):
                      formatted_row[field] = f"{value:.4f}" if value is not None else ''
                 elif value is None:
                      formatted_row[field] = ''
                 else:
                      formatted_row[field] = value

            writer.writerow(formatted_row)
        print("Successfully appended results to CSV.")
    except IOError as e:
        print(f"Error writing to CSV file {csv_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        traceback.print_exc()
