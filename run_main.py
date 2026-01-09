#!/usr/bin/env python3
"""
Wrapper entry point for LangSAE training.

Some environments/tools can end up with a non-spawn multiprocessing start method,
which can break vLLM + CUDA (e.g. "Cannot re-initialize CUDA in forked subprocess").

This script forces 'spawn' *before* importing LangSAE, then exposes the same Fire
CLI as `python -m LangSAE.main`.
"""

from __future__ import annotations

import os
import multiprocessing


def _force_spawn() -> None:
    os.environ.setdefault("PYTHON_MULTIPROCESSING_START_METHOD", "spawn")
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set - that's fine.
        pass


def main() -> None:
    _force_spawn()

    import fire

    from LangSAE.main import main as train_main

    fire.Fire(train_main)


if __name__ == "__main__":
    main()

