import logging
from typing import List, Optional

from eval.chat_benchmarks.LiveCodeBench.eval_instruct import HF_HUB_CACHE, LiveCodeBenchBenchmark


class LiveCodeBenchV5Benchmark(LiveCodeBenchBenchmark):
    """Compatibility wrapper for the legacy LCBv5 dataset."""

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
            dataset_repo="mlfoundations-dev/LCBv5-v2",
            cache_dir=HF_HUB_CACHE,
            n_repeat=3,
            logger=logger,
            system_instruction=system_instruction,
        )
