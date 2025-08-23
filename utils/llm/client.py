from volcenginesdkarkruntime import Ark
import re

from utils.config import ConfigManager


class LLMClient:
    def __init__(self):
        self.config = ConfigManager()
        self.client = Ark(api_key=self.config.llm_key)

    def run_query(self, query):
        completion = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return completion.choices[0].message.content

    def extract_answer_between_tags(self, text, tag):
        pattern = f'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
