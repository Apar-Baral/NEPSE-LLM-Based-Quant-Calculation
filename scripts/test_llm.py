#!/usr/bin/env python3
"""Test DeepSeek / configured LLM connection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.llm.analyst import llm_status, test_llm_connection

if __name__ == "__main__":
    status = llm_status()
    print("Provider:", status["provider"])
    print("Model:", status["model"])
    print("Ready:", status["ready"])
    if status.get("hint"):
        print("Hint:", status["hint"])
    print("-" * 40)
    result = test_llm_connection()
    print("OK:", result.get("ok"))
    print("Response:\n", result.get("response", ""))
