# benchmark_metrics.py

import time
import json
import asyncio
import aiohttp
import traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import argparse # Import argparse to use args type hint

@dataclass
class RequestMetrics:
    """Stores metrics for a single API request."""
    start_time: float
    first_content_token_time_raw: Optional[float] = None # Raw time for content
    first_reasoning_token_time_raw: Optional[float] = None # Raw time for reasoning
    first_token_time: Optional[float] = None # This will be the *calculated* TTFT time
    end_time: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None # This will be the *calculated* total output tokens
    reasoning_tokens: Optional[int] = field(default=None) # Explicitly counted reasoning tokens

    @property
    def latency(self) -> Optional[float]:
        """Calculate end-to-end latency for the request."""
        if self.end_time is None: return None
        # Ensure latency is non-negative
        return max(0.0, self.end_time - self.start_time)

    @property
    def ttft(self) -> Optional[float]:
        """Calculate Time To First Token based on the calculated first_token_time."""
        if self.first_token_time is None: return None
        # Ensure TTFT is non-negative relative to start_time
        return max(0.0, self.first_token_time - self.start_time)

    @property
    def tbt(self) -> Optional[float]:
        """Calculate Time Between Tokens based on calculated TTFT and output tokens."""
        if (self.end_time is None or self.first_token_time is None or
            self.output_tokens is None or self.output_tokens <= 1 or
            self.end_time <= self.first_token_time):
            return None # Not enough info or invalid times
        total_generation_time = max(0.0, self.end_time - self.first_token_time)
        # Use the calculated output_tokens which might include reasoning tokens
        # Avoid division by zero
        denominator = self.output_tokens - 1
        return total_generation_time / denominator if denominator > 0 else 0.0

@dataclass
class SystemMetrics:
    """Stores aggregated metrics for the entire benchmark run."""
    first_request_start_time: Optional[float] = None
    first_token_time: Optional[float] = None # Based on the calculated TTFT per request
    last_token_time: Optional[float] = None
    total_output_tokens: int = 0 # Aggregated calculated output_tokens
    total_reasoning_tokens: int = 0 # Aggregated explicit reasoning_tokens

    @property
    def total_generation_time(self) -> Optional[float]:
        """Time from the first calculated token arrival to the last token arrival."""
        if self.last_token_time is None or self.first_token_time is None: return None
        return max(0.0, self.last_token_time - self.first_token_time)

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """Time from the start of the first request to the arrival of the last token."""
        if self.last_token_time is None or self.first_request_start_time is None: return None
        return max(0.0, self.last_token_time - self.first_request_start_time)

    @property
    def system_tpot(self) -> Optional[float]:
        """System-wide Time Per Output Token."""
        gen_time = self.total_generation_time
        total_tokens_for_tpot = self.total_output_tokens
        if (gen_time is None or gen_time <= 0 or total_tokens_for_tpot == 0): return None
        return gen_time / total_tokens_for_tpot

class MetricsCollector:
    """Collects and aggregates request metrics."""
    def __init__(self):
        self.metrics: List[RequestMetrics] = []
        self.system_metrics = SystemMetrics()
        self._lock = asyncio.Lock() # Lock for thread-safe metric addition

    async def add_metrics(self, metric: RequestMetrics):
        """Adds a RequestMetrics object safely."""
        async with self._lock:
            # Basic validation
            if metric.end_time is not None and metric.end_time < metric.start_time:
                print(f"Warning: Ignoring metric with end_time ({metric.end_time}) before start_time ({metric.start_time}).")
                return
            if metric.first_token_time is not None and metric.first_token_time < metric.start_time:
                 print(f"Warning: Metric has calculated first_token_time ({metric.first_token_time}) before start_time ({metric.start_time}). TTFT property will clamp to 0.")

            self.metrics.append(metric)

            # Use the *calculated* first_token_time from the metric for system aggregation
            current_calculated_first_token_time = metric.first_token_time

            # Update system start time
            if self.system_metrics.first_request_start_time is None:
                self.system_metrics.first_request_start_time = metric.start_time
            else:
                self.system_metrics.first_request_start_time = min(self.system_metrics.first_request_start_time, metric.start_time)

            # Update system first token time (using the *calculated* TTFT time)
            if current_calculated_first_token_time is not None:
                 if self.system_metrics.first_token_time is None:
                      self.system_metrics.first_token_time = current_calculated_first_token_time
                 else:
                      # Only update if the current token arrived *after* its request started
                      if current_calculated_first_token_time >= metric.start_time:
                          self.system_metrics.first_token_time = min(self.system_metrics.first_token_time, current_calculated_first_token_time)

            # Update system last token time
            if metric.end_time is not None:
                if self.system_metrics.last_token_time is None:
                    self.system_metrics.last_token_time = metric.end_time
                else:
                    self.system_metrics.last_token_time = max(self.system_metrics.last_token_time, metric.end_time)

            # Aggregate the *calculated* output_tokens and the explicit reasoning_tokens
            self.system_metrics.total_output_tokens += metric.output_tokens or 0
            self.system_metrics.total_reasoning_tokens += metric.reasoning_tokens or 0

    def calculate_statistics(self) -> Dict:
        """Calculates final statistics based on collected metrics."""
        if not self.metrics:
             # Return default zero values
             return {
                "requests_per_second": 0, "tokens_per_second": 0,
                "avg_request_latency": 0, "avg_ttft": 0, "avg_tbt": 0,
                "total_requests": 0, "total_generation_time": 0,
                "end_to_end_latency": 0, "system_tokens_per_second": 0,
                "system_tpot": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "total_reasoning_tokens": 0,
            }

        # Use aggregated values from system_metrics where appropriate
        total_input_tokens = sum(m.input_tokens for m in self.metrics if m.input_tokens is not None)
        total_output_tokens = self.system_metrics.total_output_tokens # Uses aggregated calculated value
        total_reasoning_tokens = self.system_metrics.total_reasoning_tokens # Uses aggregated explicit value
        total_time_e2e = self.system_metrics.end_to_end_latency or 0.0
        total_gen_time = self.system_metrics.total_generation_time or 0.0

        # Calculate averages based on valid individual request metrics
        valid_latencies = [m.latency for m in self.metrics if m.latency is not None and m.latency >= 0]
        valid_ttfts = [m.ttft for m in self.metrics if m.ttft is not None and m.ttft >= 0] # Uses calculated ttft
        valid_tbts = [m.tbt for m in self.metrics if m.tbt is not None and m.tbt >= 0] # Uses calculated tbt

        # Total tokens for RPS calculation uses the calculated output tokens
        total_tokens_for_rps = total_input_tokens + total_output_tokens

        stats = {
            "requests_per_second": len(self.metrics) / total_time_e2e if total_time_e2e > 0 else 0,
            "tokens_per_second": total_tokens_for_rps / total_time_e2e if total_time_e2e > 0 else 0,
            "avg_request_latency": sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0,
            "avg_ttft": sum(valid_ttfts) / len(valid_ttfts) if valid_ttfts else 0, # Based on calculated TTFT
            "avg_tbt": sum(valid_tbts) / len(valid_tbts) if valid_tbts else 0,     # Based on calculated TBT
            "total_requests": len(self.metrics),
            "total_generation_time": total_gen_time,
            "end_to_end_latency": total_time_e2e,
            # system_tokens_per_second uses the calculated output tokens
            "system_tokens_per_second": (total_output_tokens / total_gen_time if total_gen_time > 0 else 0),
            "system_tpot": self.system_metrics.system_tpot or 0, # Uses calculated system_tpot
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens, # The calculated total output
            "total_reasoning_tokens": total_reasoning_tokens, # The explicit reasoning total
        }
        return stats


async def completion_with_backoff(session: aiohttp.ClientSession, input_prompt: str, args: argparse.Namespace): 
    """
    Sends a request to the completion API, handles streaming response, and calculates metrics.

    Args:
        session: The aiohttp ClientSession.
        input_prompt: The prompt string for the API call.
        args: The argparse Namespace containing configuration.

    Returns:
        A tuple containing:
        - The concatenated response content (str) or None on failure.
        - A RequestMetrics object populated with timing and token info, or None if metric creation failed.
    """
    url = f"http://{args.url}:{args.port}/v1/chat/completions"
    metrics = RequestMetrics(start_time=time.time())

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {args.api_key}'
    }

    data = {
        'model': args.model_name,
        'messages': [{"role": "user", "content": input_prompt}],
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_tokens': args.max_token_num,
        'stream': True,
        # Add other parameters like 'stop' if needed
    }

    final_content_response = []
    reasoning_content_response = []
    content_tokens = 0
    reasoning_tokens = 0
    first_content_token_time_raw = None
    first_reasoning_token_time_raw = None
    error_occurred = False
    response_status = None

    request_timeout = aiohttp.ClientTimeout(total=args.request_timeout)

    try:
        async with session.post(url, headers=headers, data=json.dumps(data), timeout=request_timeout) as response:
            response_status = response.status
            if response.status == 200:
                async for line in response.content:
                    current_chunk_time = time.time()
                    try:
                        line_str = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        print(f"Warning: UnicodeDecodeError processing chunk for prompt: {input_prompt[:50]}...")
                        error_occurred = True; continue

                    if line_str.startswith('data: '):
                        if line_str == 'data: [DONE]': break
                        json_payload_str = line_str[len('data: '):]
                        if not json_payload_str: continue

                        try:
                            chunk_data = json.loads(json_payload_str)
                            if 'choices' in chunk_data and isinstance(chunk_data['choices'], list) and len(chunk_data['choices']) > 0:
                                choice = chunk_data['choices'][0]
                                if isinstance(choice, dict):
                                    delta = choice.get('delta', {})
                                    if isinstance(delta, dict):
                                        content_piece = delta.get('content')
                                        reasoning_piece = delta.get('reasoning_content') if args.is_reasoning_llm else None

                                        # Track raw first token times
                                        if content_piece and first_content_token_time_raw is None:
                                            first_content_token_time_raw = current_chunk_time
                                        if reasoning_piece and first_reasoning_token_time_raw is None:
                                            first_reasoning_token_time_raw = current_chunk_time

                                        # Append content and count tokens
                                        if content_piece:
                                            final_content_response.append(content_piece)
                                            content_tokens += 1
                                        if reasoning_piece:
                                            reasoning_content_response.append(reasoning_piece)
                                            reasoning_tokens += 1
                        except json.JSONDecodeError:
                            print(f"Warning: JSONDecodeError processing chunk: '{json_payload_str}' for prompt: {input_prompt[:50]}...")
                            error_occurred = True; continue

                # ---- Processing finished ----
                metrics.end_time = time.time()
                content = "".join(final_content_response) if final_content_response else ""

                try:
                    metrics.input_tokens = len(input_prompt.split())
                except Exception: metrics.input_tokens = 0

                metrics.first_content_token_time_raw = first_content_token_time_raw
                metrics.first_reasoning_token_time_raw = first_reasoning_token_time_raw
                metrics.reasoning_tokens = reasoning_tokens

                # ---- Calculate final output tokens and TTFT based on args ----
                if args.is_reasoning_llm:
                    metrics.output_tokens = content_tokens + reasoning_tokens
                    if args.is_cot_visible:
                        valid_times = [t for t in [first_content_token_time_raw, first_reasoning_token_time_raw] if t is not None]
                        metrics.first_token_time = min(valid_times) if valid_times else None
                    else:
                        metrics.first_token_time = first_content_token_time_raw
                else:
                    metrics.output_tokens = content_tokens
                    metrics.first_token_time = first_content_token_time_raw

                return content, metrics

            else: # Handle non-200 initial response
                error_text = await response.text()
                print(f'--- API Error {response.status}: {error_text} for prompt: {input_prompt[:50]}...')
                metrics.end_time = time.time()
                try: metrics.input_tokens = len(input_prompt.split())
                except Exception: metrics.input_tokens = 0
                metrics.output_tokens = 0
                metrics.reasoning_tokens = 0
                metrics.first_token_time = None
                return None, metrics

    # Handle exceptions during the request
    except aiohttp.ClientConnectorError as e:
         print(f"--- Connection Error: {e} for URL: {url}")
         error_occurred = True
    except asyncio.TimeoutError:
        print(f"--- Request Timeout Error ({args.request_timeout}s) for URL: {url}")
        error_occurred = True
    except Exception as e:
        print(f'--- Request failed unexpectedly: {e}')
        traceback.print_exc()
        error_occurred = True

    # Common error handling / metric finalization
    if metrics:
        metrics.end_time = time.time()
        if metrics.input_tokens is None:
            try: metrics.input_tokens = len(input_prompt.split())
            except Exception: metrics.input_tokens = 0
        if metrics.output_tokens is None: metrics.output_tokens = 0
        if metrics.reasoning_tokens is None: metrics.reasoning_tokens = 0
        metrics.first_token_time = None # No valid first token on error
        return None, metrics
    else:
        # Should not happen, but return None if metric object creation failed
        return None, None
