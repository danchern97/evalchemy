import logging
from typing import List, Optional

from eval.chat_benchmarks.LiveCodeBench.eval_instruct import LiveCodeBenchBenchmark


class LiveCodeBenchV5OfficialBenchmark(LiveCodeBenchBenchmark):
    """Compatibility wrapper for the official LiveCodeBench v5 slice."""

    def __init__(
        self,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(
            debug=debug,
            seed=seed,
            max_tokens=max_tokens,
            version="v5",
            cache_dir="./",
            contest_months=["2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"],
            n_repeat=3,
            logger=logger,
            system_instruction=system_instruction,
        )
