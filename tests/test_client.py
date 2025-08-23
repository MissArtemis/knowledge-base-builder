import os
import sys
from utils.config import ConfigManager

project_root = ConfigManager().project_root
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_run_query():
    question = "What is the meaning of life?"
    from utils.llm.client import LLMClient
    client = LLMClient()
    response = client.run_query(question)
    assert response is not None
    print(response)
