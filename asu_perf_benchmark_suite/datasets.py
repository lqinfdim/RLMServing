# benchmark_datasets.py


import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import re
import random
import asyncio
import time
import traceback

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import aiohttp
from tqdm import trange

from datasets import load_dataset

# Import metrics components (assuming they are in the same directory or PYTHONPATH)
from .metrics import MetricsCollector, RequestMetrics,completion_with_backoff

def test_answer_math500(pred_str, ans_str):
    """
    Compares the predicted answer (pred_str) with the gold answer (ans_str).
    Handles both numeric and text answers, extracting from full model response.
    """

    # --- Gold Answer Extraction ---
    gold_ans_extracted = None
    # Remove "A:" prefix and clean
    ans_str_cleaned = re.sub(r'^A:\s*', '', ans_str, flags=re.IGNORECASE).strip()

    # 1. Try extracting from \boxed{}
    boxed_match = re.search(r'\\boxed\{(.+?)\}', ans_str_cleaned, re.DOTALL)
    if boxed_match:
        gold_ans_extracted = boxed_match.group(1).strip()
        # print(f"DEBUG (Gold): Extracted from boxed: '{gold_ans_extracted}'")
    else:
        # 2. Fallback: Use the last non-empty line
        lines = [line for line in ans_str_cleaned.splitlines() if line.strip()]
        if lines:
            gold_ans_extracted = lines[-1].strip()
            # print(f"DEBUG (Gold): Extracted fallback (last line): '{gold_ans_extracted}'")
        else:
            gold_ans_extracted = ans_str_cleaned # Use the cleaned string itself if no lines/boxed
            # print(f"DEBUG (Gold): Extracted fallback (full cleaned string): '{gold_ans_extracted}'")

    if gold_ans_extracted is None:
         # print(f"Warning: Could not extract gold answer from: '{ans_str}'")
         return False


    # --- Prediction Answer Extraction ---
    pred_ans_extracted = None
    # Remove "A_model:" prefix and clean
    pred_str_cleaned = re.sub(r'^A_model:\s*', '', pred_str, flags=re.IGNORECASE).strip()

    # 1. Look for "Answer: $ANSWER" format (case-insensitive) - *PRIORITY*
    # Find all matches and take the *last* one, as it's the intended final answer line
    answer_matches = re.findall(r'^Answer:\s*(.*)', pred_str_cleaned, re.IGNORECASE | re.MULTILINE)
    if answer_matches:
        pred_ans_extracted = answer_matches[-1].strip()
        # print(f"DEBUG (Pred): Extracted from 'Answer:': '{pred_ans_extracted}'")
    else:
        # 2. Fallback: If "Answer:" marker not found (model didn't follow prompt),
        #    use the last non-empty line as a guess.
        lines = [line for line in pred_str_cleaned.splitlines() if line.strip()]
        if lines:
             pred_ans_extracted = lines[-1].strip()
             # print(f"DEBUG (Pred): Extracted fallback (last line): '{pred_ans_extracted}'")
        else:
             # If prediction is empty or only whitespace after cleaning
             # print(f"Warning: Prediction string is empty after cleaning: '{pred_str}'")
             return False # Cannot compare if prediction is effectively empty

    if pred_ans_extracted is None: # Should not happen with above logic
         # print(f"Warning: Could not extract prediction answer from: '{pred_str}'")
         return False


    # --- Normalization Helper ---
    def normalize_answer(text):
        text = str(text).strip()
        # Remove commas used for thousands separators
        text = text.replace(',', '')

        # Handle LaTeX fractions e.g., \frac{a}{b} -> a/b
        text = re.sub(r'\\frac\{(.+?)\}\{(.+?)\}', r'\1/\2', text)
        # Handle \text{}
        text = re.sub(r'\\text\{(.+?)\}', r'\1', text)
        # Remove $, \boxed, {, } and leading/trailing whitespace again
        text = text.replace('$', '').replace('\\boxed', '').replace('{', '').replace('}', '').strip()
        # Remove percentage signs
        text = text.replace('%', '').strip()

        # Attempt numeric conversion
        try:
            # Try converting to float directly
            num = float(text)
            # Check if it's essentially an integer
            if np.isclose(num, round(num)):
                return str(int(round(num))) # Return as integer string
            else:
                # Check for scientific notation representation like 1e+...
                # np.isclose handles these correctly during comparison
                return str(num) # Return as float string
        except ValueError:
            # Not a standard number, treat as text
            # Convert to lower case for case-insensitive comparison
            return text.lower()

    # --- Comparison ---
    gold_ans_normalized = normalize_answer(gold_ans_extracted)
    pred_ans_normalized = normalize_answer(pred_ans_extracted)

    # print(f"DEBUG: Normalized Gold: '{gold_ans_normalized}' | Normalized Pred: '{pred_ans_normalized}'")

    # Final Comparison Logic
    try:
        # Try numeric comparison first
        gold_num = float(gold_ans_normalized)
        pred_num = float(pred_ans_normalized)
        # Use tolerance for floating point comparison
        is_correct = np.isclose(pred_num, gold_num, rtol=1e-4, atol=1e-6)
        # print(f"DEBUG: Numeric comparison result: {is_correct}")
        return is_correct
    except ValueError:
        # If numeric conversion fails, perform case-insensitive string comparison
        # Normalization already converted text to lower case
        is_correct = (gold_ans_normalized == pred_ans_normalized)
        # print(f"DEBUG: String comparison result: {is_correct}")
        return is_correct

# --- AIME Specific Answer Testing ---
def test_answer_aime(pred_str, ans_str):
    if not pred_str or pred_str.strip() == "A_model:": return False
    gold_pattern = r"final answer is.*?\\boxed\{(\d{1,3})\}"
    gold_match = re.search(gold_pattern, ans_str, re.IGNORECASE | re.DOTALL)
    if gold_match: gold = gold_match.group(1).strip()
    else:
        gold_nums = re.findall(r'\d+', ans_str)
        if gold_nums: gold = gold_nums[-1].strip()
        else: return False
    answer_patterns = [
        r'final answer is.*?\\boxed\{(\d+)\}', r'(?:answer|final answer|result)(?:\s+is\s+|\s*:\s*)\$?(\d+)\b',
        r'\\boxed\{(\d+)\}', r'(?:therefore|thus|so)(?:,?\s+the\s+answer\s+is\s+|\s+we\s+get\s+)\$?(\d+)\b',
        r'(?:equals|equal to|=)\s*\$?(\d+)\b', r'is\s+\$?(\d+)\.?$', r'\b(\d+)$'
    ]
    pred_num_str = None
    for pattern in answer_patterns:
        matches = re.findall(pattern, pred_str, re.IGNORECASE | re.DOTALL)
        if matches: pred_num_str = matches[-1]; break
    if pred_num_str is None:
        all_numbers = re.findall(r'\d+', pred_str)
        if all_numbers: pred_num_str = all_numbers[-1]
        else: return False
    pred_num_str = pred_num_str.strip()
    if gold.isdigit() and pred_num_str.isdigit():
        try: return int(gold) == int(pred_num_str)
        except ValueError: return gold == pred_num_str
    else: return gold == pred_num_str

# --- GSM8K Specific Answer Testing ---
def test_answer_gsm8k(pred_str, ans_str):
    if not ans_str: return False
    pattern = r'\d*\.?\d+'
    pred_list = re.findall(pattern, pred_str)
    if not pred_list: return False
    pred = pred_list[-1].rstrip('.').replace(',', '').strip()
    ans_list = re.findall(pattern, ans_str)
    if not ans_list: return False # Gold answer must contain a number
    gold = ans_list[-1].rstrip('.').replace(',', '').strip()
    try:
        if not pred or not gold: return False # Handle empty strings after cleaning
        return float(pred) == float(gold)
    except ValueError: return pred == gold # Fallback to string comparison
    except Exception: return False


# --- GPQA Specific Constants ---
LETTER_INDICES = ['A', 'B', 'C', 'D']


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""
    def __init__(self, args, metrics_collector: MetricsCollector):
        self.args = args
        self.metrics_collector = metrics_collector
        self.dataset = None
        self.prompt_template = "" # Or load from file in prepare

    @abstractmethod
    def prepare_benchmark(self) -> bool:
        """Load dataset, prompt, and perform any preprocessing."""
        pass

    @abstractmethod
    async def run_benchmark(self, session: aiohttp.ClientSession, fd) -> Dict:
        """Run the benchmark loop, collect metrics, and return final stats."""
        pass

    @abstractmethod
    def _handle_item(self, session: aiohttp.ClientSession, item: Dict, fd):
        """Process a single item (format prompt, call API, handle response)."""
        pass

    @abstractmethod
    def parse_and_evaluate(self) -> Tuple[int, int]:
        """Parse the output file and calculate accuracy."""
        pass

    def _load_hf_dataset(self, dataset_id, subset=None, split='test'):
        """Helper to load Hugging Face datasets with common options."""
        print(f"Loading dataset '{dataset_id}' (Subset: {subset}, Split: {split})...")
        try:
            # full_dataset = load_dataset(dataset_id, subset, trust_remote_code=True, token=self.args.hf_token)
            full_dataset = load_dataset(dataset_id, subset,  token=self.args.hf_token)
            if split not in full_dataset:
                print(f"Error: Split '{split}' not found in dataset '{dataset_id}' (Subset: {subset}). Available splits: {list(full_dataset.keys())}")
                return None
            data = full_dataset[split]
            max_num = len(data)

            capacity = self.args.capacity
            if capacity == -1: selected_capacity = max_num
            elif capacity < -1 or capacity == 0: selected_capacity = 0
            else: selected_capacity = min(capacity, max_num)

            if selected_capacity == 0:
                 print("Capacity set to 0, no data will be loaded.")
                 return None

            print(f"Selecting first {selected_capacity} samples.")
            return data.select(range(selected_capacity))
        except Exception as e:
            print(f"Error loading dataset '{dataset_id}' (Subset: {subset}, Split: {split}): {e}")
            traceback.print_exc()
            return None

    def _read_prompt_file(self, default_path):
        """Reads prompt from file specified in args, or uses default."""
        prompt_path = getattr(self.args, 'prompt_path', default_path) # Use arg if exists
        if not prompt_path:
            print(f"Warning: No prompt path specified, using empty prompt.")
            return ""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Prompt file not found at {prompt_path}. Using empty prompt.")
            return ""
        except Exception as e:
            print(f"Error reading prompt file {prompt_path}: {e}. Using empty prompt.")
            return ""

    async def _run_benchmark_loop(self, session: aiohttp.ClientSession, fd):
        """Common benchmark loop structure."""
        if self.dataset is None:
            print("Error: Dataset not prepared. Call prepare_benchmark first.")
            return {}

        data_indices = list(range(len(self.dataset)))

        for rep in range(self.args.repeat_time):
            print(f"\n--- Starting repetition {rep+1}/{self.args.repeat_time} ---")
            progress_bar = trange(0, len(data_indices), self.args.batch_size, desc=f"Rep {rep+1} Batches")
            for i in progress_bar:
                batch_indices = data_indices[i:min(i + self.args.batch_size, len(data_indices))]
                if not batch_indices: continue

                batch_items = [self.dataset[idx] for idx in batch_indices]

                tasks = [
                    self._handle_item(session, item, fd)
                    for item in batch_items
                ]
                if tasks:
                    await asyncio.gather(*tasks)

        print("\nBenchmark run finished.")
        return self.metrics_collector.calculate_statistics()

# --- GSM8K ---
class GSM8kBenchmark(BenchmarkDataset):
    def prepare_benchmark(self) -> bool:
        self.dataset = self._load_hf_dataset('openai/gsm8k', 'main', 'test')
        if self.dataset is None: return False
        # Rename column if needed (already 'question' and 'answer' in this dataset)
        self.prompt_template = self._read_prompt_file('./prompt/gsm8k/prompt_original.txt') # Example default
        return True

    async def run_benchmark(self, session: aiohttp.ClientSession, fd) -> Dict:
        return await self._run_benchmark_loop(session, fd)

    async def _handle_item(self, session: aiohttp.ClientSession, item: Dict, fd):
        q = item.get('question', '[Question Missing]')
        a = item.get('answer', '') # Gold answer

        # Format prompt (specific to GSM8K structure)
        prompt_q = f"Follow the given examples format and answer the following question.\n\n{self.prompt_template}\nQuestion: {q}\n"

        response_content, metrics = await completion_with_backoff(session, prompt_q, self.args)

        if metrics:
            await self.metrics_collector.add_metrics(metrics) # Use await for async add

        # Process response and write output
        if response_content is not None:
            try:
                # GSM8K specific answer extraction (optional, if needed before writing)
                # ans_, residual = extract_ans_gsm8k(response_content) # Define if needed
                ans_ = response_content.strip() # Keep full response for now

                q_cleaned = q.strip()
                fd.write(f'Q: {q_cleaned}\n')
                fd.write(f'A_model:\n{ans_}\n')
                if not self.args.benchmark_performance_only:
                    a_cleaned = a.strip() if isinstance(a, str) else ""
                    fd.write(f'A:\n{a_cleaned}\n')
                fd.write('\n')
                fd.flush()
            except Exception as e:
                print(f"Error processing/writing GSM8K response for Q: {q[:50]}... Error: {e}")
                # Write error marker
        else:
            # Handle API failure
            q_cleaned = q.strip()
            error_line = f'Q: {q_cleaned}\nA_model:\n[ERROR: Failed to get response content from API]\n'
            if not self.args.benchmark_performance_only:
                 a_cleaned = a.strip() if isinstance(a, str) else ""
                 error_line += f'A:\n{a_cleaned}\n'
            error_line += '\n'
            fd.write(error_line)
            fd.flush()

    def parse_and_evaluate(self) -> Tuple[int, int]:
        num_q, acc = 0, 0
        filename = self.args.output_path
        performance_only = self.args.benchmark_performance_only

        try:
            with open(filename, 'r', encoding='utf-8') as fd: lines = fd.readlines()
        except FileNotFoundError: print(f"Error: Output file not found at {filename}"); return 0, 0
        if not lines: print(f"Warning: Output file '{filename}' is empty"); return 0, 0

        q, am, a = None, None, None
        current_mode = 'none'
        for l in lines:
            line_stripped = l.strip()
            if line_stripped.startswith('#') or not line_stripped: continue
            if line_stripped.startswith('Q:'):
                if q is not None:
                    am_str = am.strip() if am is not None else "A_model:"
                    a_str = a.strip() if a is not None and not performance_only else ""
                    if a_str and am_str != "A_model:":
                        try:
                            if test_answer_gsm8k(am_str, a_str): acc += 1
                        except Exception as e: print(f"Warning: Error during test_answer_gsm8k: {e}")
                q = l; am, a = None, None; num_q += 1; current_mode = 'q'
            elif line_stripped.startswith('A_model:'):
                if q is None: continue
                am = l; current_mode = 'am'
            elif line_stripped.startswith('A:') and not performance_only:
                if q is None: continue
                a = l; current_mode = 'a'
            else:
                if current_mode == 'q' and q is not None: q += l
                elif current_mode == 'am' and am is not None: am += l
                elif current_mode == 'a' and a is not None and not performance_only: a += l
        # Process last entry
        if q is not None:
            am_str = am.strip() if am is not None else "A_model:"
            a_str = a.strip() if a is not None and not performance_only else ""
            if a_str and am_str != "A_model:":
                 try:
                     if test_answer_gsm8k(am_str, a_str): acc += 1
                 except Exception as e: print(f"Warning: Error during test_answer_gsm8k: {e}")

        if not performance_only: print(f"Result Parsing (GSM8K): num_q={num_q}, correct={acc}, ratio={float(acc / num_q):.4f}" if num_q > 0 else "Result Parsing (GSM8K): num_q=0")
        else: print(f"Result Parsing (GSM8K): num_q={num_q} (Accuracy not calculated)")
        return num_q, acc

# --- AIME ---
class AIMEBenchmark(BenchmarkDataset):
    def prepare_benchmark(self) -> bool:
        # AIME dataset only has 'train' split
        self.dataset = self._load_hf_dataset('HuggingFaceH4/aime_2024', None, 'train')
        if self.dataset is None: return False
        try:
            self.dataset = self.dataset.rename_column('problem', 'question')
            if 'answer' not in self.dataset.column_names:
                print("Error: 'answer' column missing in AIME dataset.")
                return False
        except Exception as e:
            print(f"Error preparing AIME dataset columns: {e}")
            return False
        # AIME prompt is usually hardcoded in the handler
        self.prompt_template = "" # Not loaded from file for AIME typically
        return True

    async def run_benchmark(self, session: aiohttp.ClientSession, fd) -> Dict:
        return await self._run_benchmark_loop(session, fd)

    async def _handle_item(self, session: aiohttp.ClientSession, item: Dict, fd):
        q = item.get('question', '[Question Missing]')
        a = item.get('answer', '') # Gold answer

        # AIME specific prompt structure
        prompt_q = f"""
Solve the following math problem step by step. The final answer should be a non-negative integer. Output the final answer in the format \\boxed{{ANSWER}} on the last line.

Problem:
{q}

Solution:
""".strip()

        response_content, metrics = await completion_with_backoff(session, prompt_q, self.args)

        if metrics:
            await self.metrics_collector.add_metrics(metrics)

        # Process response and write output
        if response_content is not None:
            try:
                ans_ = response_content.strip() # Keep full response
                q_cleaned = q.strip()
                fd.write(f'Q: {q_cleaned}\n')
                fd.write(f'A_model:\n{ans_}\n')
                if not self.args.benchmark_performance_only:
                    a_cleaned = a.strip() if isinstance(a, str) else ""
                    fd.write(f'A:\n{a_cleaned}\n')
                fd.write('\n')
                fd.flush()
            except Exception as e:
                print(f"Error processing/writing AIME response for Q: {q[:50]}... Error: {e}")
                # Write error marker
        else:
            # Handle API failure
            q_cleaned = q.strip()
            error_line = f'Q: {q_cleaned}\nA_model:\n[ERROR: Failed to get response content from API]\n'
            if not self.args.benchmark_performance_only:
                 a_cleaned = a.strip() if isinstance(a, str) else ""
                 error_line += f'A:\n{a_cleaned}\n'
            error_line += '\n'
            fd.write(error_line)
            fd.flush()

    def parse_and_evaluate(self) -> Tuple[int, int]:
        # Uses the same parsing logic structure as GSM8K, but with test_answer_aime
        num_q, acc = 0, 0
        filename = self.args.output_path
        performance_only = self.args.benchmark_performance_only

        try:
            with open(filename, 'r', encoding='utf-8') as fd: lines = fd.readlines()
        except FileNotFoundError: print(f"Error: Output file not found at {filename}"); return 0, 0
        if not lines: print(f"Warning: Output file '{filename}' is empty"); return 0, 0

        q, am, a = None, None, None
        current_mode = 'none'
        for l in lines:
            line_stripped = l.strip()
            if line_stripped.startswith('#') or not line_stripped: continue
            if line_stripped.startswith('Q:'):
                if q is not None:
                    am_str = am.strip() if am is not None else "A_model:"
                    a_str = a.strip() if a is not None and not performance_only else ""
                    if a_str and am_str != "A_model:":
                        try:
                            if test_answer_aime(am_str, a_str): acc += 1
                        except Exception as e: print(f"Warning: Error during test_answer_aime: {e}")
                q = l; am, a = None, None; num_q += 1; current_mode = 'q'
            elif line_stripped.startswith('A_model:'):
                if q is None: continue
                am = l; current_mode = 'am'
            elif line_stripped.startswith('A:') and not performance_only:
                if q is None: continue
                a = l; current_mode = 'a'
            else:
                if current_mode == 'q' and q is not None: q += l
                elif current_mode == 'am' and am is not None: am += l
                elif current_mode == 'a' and a is not None and not performance_only: a += l
        # Process last entry
        if q is not None:
            am_str = am.strip() if am is not None else "A_model:"
            a_str = a.strip() if a is not None and not performance_only else ""
            if a_str and am_str != "A_model:":
                 try:
                     if test_answer_aime(am_str, a_str): acc += 1
                 except Exception as e: print(f"Warning: Error during test_answer_aime: {e}")

        if not performance_only: print(f"Result Parsing (AIME): num_q={num_q}, correct={acc}, ratio={float(acc / num_q):.4f}" if num_q > 0 else "Result Parsing (AIME): num_q=0")
        else: print(f"Result Parsing (AIME): num_q={num_q} (Accuracy not calculated)")
        return num_q, acc

# --- MATH ---
class MATHBenchmark(BenchmarkDataset):
    def prepare_benchmark(self) -> bool:
        self.dataset = self._load_hf_dataset('HuggingFaceH4/MATH-500', 'default', 'test')
        if self.dataset is None: return False
        try:
            self.dataset = self.dataset.rename_column('problem', 'question')
            if 'answer' not in self.dataset.column_names:
                 print("Error: 'answer' column missing in MATH dataset.")
                 return False
        except Exception as e:
            print(f"Error preparing MATH dataset columns: {e}")
            return False
        # MATH prompt is usually hardcoded
        self.prompt_template = ""
        return True

    async def run_benchmark(self, session: aiohttp.ClientSession, fd) -> Dict:
        return await self._run_benchmark_loop(session, fd)

    async def _handle_item(self, session: aiohttp.ClientSession, item: Dict, fd):
        q = item.get('question', '[Question Missing]')
        a = item.get('answer', '') # Gold answer

        # MATH specific prompt structure
        prompt_q = f"""
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{q}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()

        response_content, metrics = await completion_with_backoff(session, prompt_q, self.args)

        if metrics:
            await self.metrics_collector.add_metrics(metrics)

        # Process response and write output
        if response_content is not None:
            try:
                ans_ = response_content.strip() # Keep full response
                q_cleaned = q.strip()
                fd.write(f'Q: {q_cleaned}\n')
                fd.write(f'A_model:\n{ans_}\n')
                if not self.args.benchmark_performance_only:
                    a_cleaned = a.strip() if isinstance(a, str) else ""
                    fd.write(f'A:\n{a_cleaned}\n')
                fd.write('\n')
                fd.flush()
            except Exception as e:
                print(f"Error processing/writing MATH response for Q: {q[:50]}... Error: {e}")
                # Write error marker
        else:
            # Handle API failure
            q_cleaned = q.strip()
            error_line = f'Q: {q_cleaned}\nA_model:\n[ERROR: Failed to get response content from API]\n'
            if not self.args.benchmark_performance_only:
                 a_cleaned = a.strip() if isinstance(a, str) else ""
                 error_line += f'A:\n{a_cleaned}\n'
            error_line += '\n'
            fd.write(error_line)
            fd.flush()

    def parse_and_evaluate(self) -> Tuple[int, int]:
        # Uses the same parsing logic structure as GSM8K, but with test_answer_math
        num_q, acc = 0, 0
        filename = self.args.output_path
        performance_only = self.args.benchmark_performance_only

        try:
            with open(filename, 'r', encoding='utf-8') as fd: lines = fd.readlines()
        except FileNotFoundError: print(f"Error: Output file not found at {filename}"); return 0, 0
        if not lines: print(f"Warning: Output file '{filename}' is empty"); return 0, 0

        q, am, a = None, None, None
        current_mode = 'none'
        for l in lines:
            line_stripped = l.strip()
            if line_stripped.startswith('#') or not line_stripped: continue
            if line_stripped.startswith('Q:'):
                if q is not None:
                    am_str = am.strip() if am is not None else "A_model:"
                    a_str = a.strip() if a is not None and not performance_only else ""
                    if a_str and am_str != "A_model:":
                        try:
                            if test_answer_math500(am_str, a_str): acc += 1
                        except Exception as e: print(f"Warning: Error during test_answer_math: {e}")
                q = l; am, a = None, None; num_q += 1; current_mode = 'q'
            elif line_stripped.startswith('A_model:'):
                if q is None: continue
                am = l; current_mode = 'am'
            elif line_stripped.startswith('A:') and not performance_only:
                if q is None: continue
                a = l; current_mode = 'a'
            else:
                if current_mode == 'q' and q is not None: q += l
                elif current_mode == 'am' and am is not None: am += l
                elif current_mode == 'a' and a is not None and not performance_only: a += l
        # Process last entry
        if q is not None:
            am_str = am.strip() if am is not None else "A_model:"
            a_str = a.strip() if a is not None and not performance_only else ""
            if a_str and am_str != "A_model:":
                 try:
                     if test_answer_math500(am_str, a_str): acc += 1
                 except Exception as e: print(f"Warning: Error during test_answer_math: {e}")

        if not performance_only: print(f"Result Parsing (MATH): num_q={num_q}, correct={acc}, ratio={float(acc / num_q):.4f}" if num_q > 0 else "Result Parsing (MATH): num_q=0")
        else: print(f"Result Parsing (MATH): num_q={num_q} (Accuracy not calculated)")
        return num_q, acc

# --- GPQA ---
class GPQABenchmark(BenchmarkDataset):
    def prepare_benchmark(self) -> bool:
        self.dataset = self._load_hf_dataset('Idavidrein/gpqa', "gpqa_diamond", "train")
        if self.dataset is None: return False
        # Check required columns
        required_cols = ['Question', 'Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']
        missing_cols = [col for col in required_cols if col not in self.dataset.column_names]
        if missing_cols:
            print(f"Error: GPQA dataset missing required columns: {missing_cols}")
            return False
        # GPQA prompt is hardcoded in handler
        self.prompt_template = ""
        return True

    async def run_benchmark(self, session: aiohttp.ClientSession, fd) -> Dict:
        return await self._run_benchmark_loop(session, fd)

    async def _handle_item(self, session: aiohttp.ClientSession, item: Dict, fd):
        # GPQA Prompt Template
        GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

        question_text = "[Question Missing]"
        correct_letter = "[N/A]"
        start_time_handle = time.time()

        try:
            question_text = item.get("Question", "[Question Missing]")
            correct_answer = item["Correct Answer"]
            incorrect_answers = [item["Incorrect Answer 1"], item["Incorrect Answer 2"], item["Incorrect Answer 3"]]
            if len(incorrect_answers) != 3: raise ValueError("Incorrect number of choices")

            choices = incorrect_answers[:]
            gold_index = random.randint(0, 3)
            choices.insert(gold_index, correct_answer)
            correct_letter = LETTER_INDICES[gold_index]

            prompt_q = GPQA_QUERY_TEMPLATE.format(
                Question=question_text, A=choices[0], B=choices[1], C=choices[2], D=choices[3]
            )

        except Exception as e:
            print(f"Error preparing GPQA item ({question_text[:50]}...): {e}")
            error_line = f"Q: {question_text.strip()}\nA_model:\n[ERROR: Item preparation failed: {e}]\nA: {correct_letter}\n\n"
            fd.write(error_line); fd.flush()
            await self.metrics_collector.add_metrics(RequestMetrics(start_time=start_time_handle, end_time=time.time(), output_tokens=0))
            return

        response_content, metrics = await completion_with_backoff(session, prompt_q, self.args)

        if metrics:
            await self.metrics_collector.add_metrics(metrics)

        if response_content is not None:
            try:
                ans_ = response_content.strip()
                q_cleaned = question_text.strip()
                fd.write(f'Q: {q_cleaned}\n')
                fd.write(f'A_model:\n{ans_}\n')
                if not self.args.benchmark_performance_only:
                    fd.write(f'A: {correct_letter}\n') # Write the shuffled correct letter
                fd.write('\n')
                fd.flush()
            except Exception as e:
                print(f"Error processing/writing GPQA response for Q: {question_text[:50]}... Error: {e}")
                # Write error marker
        else:
            # Handle API failure
            q_cleaned = question_text.strip()
            error_line = f'Q: {q_cleaned}\nA_model:\n[ERROR: Failed to get response content from API]\n'
            if not self.args.benchmark_performance_only:
                 error_line += f'A: {correct_letter}\n'
            error_line += '\n'
            fd.write(error_line)
            fd.flush()

    def parse_and_evaluate(self) -> Tuple[int, int]:
        # Uses the specific GPQA parsing logic
        num_q, acc = 0, 0
        filename = self.args.output_path
        performance_only = self.args.benchmark_performance_only

        try:
            with open(filename, 'r', encoding='utf-8') as fd: content = fd.read()
        except FileNotFoundError: print(f"Error: Output file not found at {filename}"); return 0, 0
        if not content: print(f"Warning: Output file '{filename}' is empty"); return 0, 0

        entries = re.split(r'\n(?=Q:)', content.strip())
        answer_pattern = re.compile(r"Answer:\s*([A-D])\s*(?:#|$)", re.IGNORECASE | re.MULTILINE)
        initial_entry_offset = 1 if entries[0].strip().startswith("# Benchmark Run Start") else 0

        for entry in entries[initial_entry_offset:]:
            entry = entry.strip()
            if not entry.startswith('Q:'): continue
            num_q += 1
            model_output, correct_letter = None, None
            a_model_start_idx = entry.find('A_model:')
            a_gold_start_idx = entry.find('\nA:')

            if a_model_start_idx != -1:
                end_idx = a_gold_start_idx if a_gold_start_idx != -1 else len(entry)
                model_output = entry[a_model_start_idx + len('A_model:'):end_idx].strip()
            else: continue

            if not performance_only:
                if a_gold_start_idx != -1:
                    a_gold_line = entry[a_gold_start_idx:].strip().split('\n', 1)[0]
                    a_gold_match = re.match(r'A:\s*([A-D])\s*', a_gold_line.strip(), re.IGNORECASE)
                    if a_gold_match: correct_letter = a_gold_match.group(1).upper()
                    else: continue # Skip accuracy check if gold A: is unparseable
                else: continue # Skip accuracy check if gold A: is missing

                if model_output and correct_letter:
                    predicted_letter = None
                    prediction_match = answer_pattern.search(model_output)
                    if prediction_match: predicted_letter = prediction_match.group(1).upper()
                    else: # Fallback
                        last_char = model_output.strip()[-1:].upper()
                        if last_char in LETTER_INDICES: predicted_letter = last_char

                    if predicted_letter and predicted_letter == correct_letter: acc += 1

        if not performance_only: print(f"Result Parsing (GPQA): num_q={num_q}, correct={acc}, accuracy={float(acc / num_q):.4f}" if num_q > 0 else "Result Parsing (GPQA): num_q=0")
        else: print(f"Result Parsing (GPQA): num_q={num_q} (Accuracy not calculated)")
        return num_q, acc


# --- Factory to get the correct benchmark class ---
BENCHMARK_CLASSES = {
    "gsm8k": GSM8kBenchmark,
    "aime24": AIMEBenchmark,
    "math500": MATHBenchmark,
    "gpqa": GPQABenchmark,
}

def get_benchmark_class(dataset_name):
    # Allow matching variations like gpqa_diamond -> gpqa
    base_name = dataset_name.split('_')[0]
    return BENCHMARK_CLASSES.get(base_name)
