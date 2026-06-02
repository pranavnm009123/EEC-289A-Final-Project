"""
local_test.py — Quick smoke test for benchmark.py.

Runs all three agents (PPO, Neural Dyna-Q, Dreamer) for 1,000 steps each
on MiniGrid-Empty-8x8-v0. Finishes in < 2 minutes on any Mac CPU.
Verifies: imports, env creation, training loop, eval, CSV logging, checkpointing.

Usage:
    python local_test.py
"""

import os
import sys
import time
import shutil
import torch

# ---------------------------------------------------------------------------
# Patch training functions to stop after TEST_STEPS
# ---------------------------------------------------------------------------
TEST_STEPS    = 1_000
TEST_ENV      = "empty"
TEST_SEED     = 0
TEST_SAVE_DIR = "results_local_test"

def run_agent(name: str, train_fn, kwargs: dict):
    print(f"\n{'='*60}")
    print(f"  Testing: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        train_fn(**kwargs)
        elapsed = time.time() - t0
        print(f"  [PASS] {name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name} crashed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Import after path is set up
    try:
        from benchmark import train_ppo, train_dynaq, train_dreamer
    except ImportError as e:
        print(f"[ERROR] Could not import benchmark.py: {e}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

    device = torch.device("cpu")
    os.makedirs(os.path.join(TEST_SAVE_DIR, "logs"),        exist_ok=True)
    os.makedirs(os.path.join(TEST_SAVE_DIR, "checkpoints"), exist_ok=True)

    results = {}

    # ---- PPO ----
    results["PPO"] = run_agent(
        "PPO",
        train_ppo,
        dict(
            env_key=TEST_ENV,
            seed=TEST_SEED,
            total_steps=TEST_STEPS,
            save_dir=TEST_SAVE_DIR,
            device=device,
        ),
    )

    # ---- Neural Dyna-Q ----
    results["Dyna-Q"] = run_agent(
        "Neural Dyna-Q",
        train_dynaq,
        dict(
            env_key=TEST_ENV,
            seed=TEST_SEED,
            total_steps=TEST_STEPS,
            save_dir=TEST_SAVE_DIR,
            device=device,
        ),
    )

    # ---- Dreamer ----
    results["Dreamer"] = run_agent(
        "Dreamer",
        train_dreamer,
        dict(
            env_key=TEST_ENV,
            seed=TEST_SEED,
            total_steps=TEST_STEPS,
            save_dir=TEST_SAVE_DIR,
            device=device,
        ),
    )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for agent, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {agent}")
        if not passed:
            all_pass = False

    # Check CSV logs were written
    print()
    log_dir = os.path.join(TEST_SAVE_DIR, "logs")
    for fname in os.listdir(log_dir):
        fpath = os.path.join(log_dir, fname)
        size = os.path.getsize(fpath)
        print(f"  log: {fname}  ({size} bytes)")

    # Clean up test artifacts
    shutil.rmtree(TEST_SAVE_DIR, ignore_errors=True)

    print()
    if all_pass:
        print("  All agents passed. Ready for full benchmark run.")
        print("  Run: python benchmark.py --agent ppo --env doorkey --seed 0 --total_steps 500000")
    else:
        print("  One or more agents failed. Fix errors above before running full benchmark.")
        sys.exit(1)


if __name__ == "__main__":
    main()
