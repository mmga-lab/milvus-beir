import logging
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


DEFAULT_MAX_THREADS = 128
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]


class QPSMeasurement:
    def __init__(self, test_duration: int, concurrency: int):
        self.test_duration = test_duration
        self.concurrency = concurrency
        self.total_queries = 0
        self.successful_queries = 0
        self.error_count = 0
        self.start_time = 0

    def process_future_result(self, future: Any) -> None:
        try:
            result = future.result()
            if result is not None:
                self.successful_queries += 1
            else:
                self.error_count += 1
        except Exception as e:
            logger.error(f"Error processing result: {e!s} {traceback.format_exc()}")
            self.error_count += 1

    def process_completed_futures(self, done_futures: set) -> None:
        for future in done_futures:
            self.process_future_result(future)

    def calculate_qps(self) -> float:
        elapsed_time = time.time() - self.start_time
        return self.successful_queries / elapsed_time if elapsed_time > 0 else 0

    def log_results(self, qps: float) -> None:
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"Concurrency {self.concurrency}: {qps:.2f} QPS "
            f"(total: {self.total_queries}, successful: {self.successful_queries}, "
            f"errors: {self.error_count}, time: {elapsed_time:.2f}s)"
        )


def run_concurrency_test(
    func: Callable,
    args: tuple,
    kwargs: dict,
    thread_count: int,
    concurrency: int,
    test_duration: int,
) -> Tuple[int, float]:
    measurement = QPSMeasurement(test_duration, concurrency)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        measurement.start_time = time.time()
        futures = []

        try:
            while time.time() - measurement.start_time < test_duration:
                while len(futures) < concurrency:
                    futures.append(executor.submit(func, *args, **kwargs))
                    measurement.total_queries += 1

                done, not_done = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)

                measurement.process_completed_futures(done)
                futures = list(not_done)

        except Exception as e:
            logger.error(f"Unexpected error during QPS test: {e!s}")
        measurement.process_completed_futures(set(futures))

    qps = measurement.calculate_qps()
    measurement.log_results(qps)
    return concurrency, qps


def measure_search_qps_decorator(
    concurrency_levels: List[int], test_duration: int = 10, max_threads: Optional[int] = None
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            default_max_threads = 128
            max_concurrent_threads = max_threads if max_threads is not None else default_max_threads
            logger.info(f"Maximum concurrent threads: {max_concurrent_threads}")

            results = []
            for concurrency in concurrency_levels:
                thread_count = min(concurrency, max_concurrent_threads)
                logger.info(f"Testing concurrency {concurrency} with {thread_count} threads")

                result = run_concurrency_test(
                    func=func,
                    args=args,
                    kwargs=kwargs,
                    thread_count=thread_count,
                    concurrency=concurrency,
                    test_duration=test_duration,
                )
                results.append(result)

            return results

        return wrapper

    return decorator
