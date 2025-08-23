from service.prompt import PromptService
from utils.llm.client import LLMClient


class LLMService:
    def __init__(self):
        self.name = "LLM Service"
        self.prompt_service = PromptService()
        self.llm = LLMClient()


    def generate_taxonomy_indexing(self, text:str, candidate_indexing:list) -> list:
        if not candidate_indexing:
            candidate_indexing = ["contract -> breach of contract -> remedies -> damages -> punitive damages"]
        prompt = self.prompt_service.build_taxonomy_indexing_prompt(
            candidate_terms="\n".join(candidate_indexing),
            text=text
        )
        llm_answer = self.llm.run_query(prompt)
        print(llm_answer)
        indexing_terms = self.llm.extract_answer_between_tags(llm_answer, "taxonomy_terms").split("\n")
        return indexing_terms