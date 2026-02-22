from __future__ import annotations

import json
import sys

from research_assistant.config import get_settings
from research_assistant.llm_client import LocalLLMClient


def main() -> int:
    settings = get_settings()
    client = LocalLLMClient(settings)
    status = client.check_server()

    print("LLM server check")
    print(json.dumps(status, indent=2))

    if status.get("ok"):
        print("\n✅ LLM server is up and usable.")
        return 0

    print("\n❌ LLM server check failed. See errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
