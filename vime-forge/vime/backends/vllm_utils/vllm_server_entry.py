"""Entry for the vLLM server subprocess under the rollout interpreter.

Dual-venv setups run the trainer actors (and the multiprocessing-spawn child
that starts the vLLM server) under the TRAIN interpreter, while vLLM must run
under its own env. ``_run_vllm_server`` re-execs into this module via
``os.execve`` with the rollout interpreter when ``VIME_VLLM_EXECUTABLE``
points at a different one; argv[1] is a pickle of (server_kwargs, env).
"""

import os
import pickle
import sys


def main() -> None:
    path = sys.argv[1]
    try:
        with open(path, "rb") as f:
            kwargs, env = pickle.load(f)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    os.environ.update(env)
    from vime.backends.vllm_utils.vllm_engine import _run_vllm_server

    _run_vllm_server(kwargs, env)


if __name__ == "__main__":
    main()
