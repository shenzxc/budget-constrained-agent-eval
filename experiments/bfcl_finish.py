#!/usr/bin/env python3
"""BFCL 收尾脚本:跑完剩余模型的 multi_turn_base 前100任务。

持久化在仓库内(不依赖被清理的 scratchpad)。幂等:bfcl generate 自动跳过已完成任务。
连接错误自动重试。每模型 generate 后 evaluate。
"""
import subprocess
import sys
import time
from pathlib import Path

BFCL_ROOT = Path("/Users/shenweiming/Projects/lunwen/experiments/gorilla/berkeley-function-call-leaderboard")
BFCL_BIN = Path("/Users/shenweiming/Projects/lunwen/experiments/.venv/bin/bfcl")
LOGS = Path("/Users/shenweiming/Projects/lunwen/experiments/logs")

# 剩余5个模型(7/12已完成),(模型, 并发线程数)
REMAINING = [
    ("qwen3.6-flash-FC", 8),
    ("deepseek-v4-flash-FC", 8),
    ("qwen3.5-397b-a17b-FC", 3),
    ("qwen3.5-122b-a10b-FC", 3),
    ("qwen3.5-35b-a3b-FC", 3),
]
CATEGORY = "multi_turn_base"
MAX_RETRY = 4


def run(cmd, logf):
    logf.write(f"\n$ {' '.join(cmd)}\n")
    logf.flush()
    return subprocess.run(cmd, cwd=str(BFCL_ROOT), stdout=logf, stderr=subprocess.STDOUT).returncode


def main():
    for model, threads in REMAINING:
        log = LOGS / f"bfcl_{model}.log"
        with log.open("a") as logf:
            for attempt in range(1, MAX_RETRY + 1):
                print(f"[{model}] generate attempt {attempt}/{MAX_RETRY}", flush=True)
                rc = run([str(BFCL_BIN), "generate", "--model", model,
                          "--test-category", CATEGORY, "--num-threads", str(threads)], logf)
                if rc == 0:
                    break
                print(f"[{model}] generate rc={rc}, 等60秒重试", flush=True)
                time.sleep(60)
            print(f"[{model}] evaluate", flush=True)
            run([str(BFCL_BIN), "evaluate", "--model", model,
                 "--test-category", CATEGORY], logf)
        print(f"[{model}] 完成", flush=True)
    print("ALL_REMAINING_DONE", flush=True)


if __name__ == "__main__":
    main()
