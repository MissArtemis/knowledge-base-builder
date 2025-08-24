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

    def generate_best_taxonomy_indexing(self, text:str, candidate_indexing:list) -> list:
        candidate_indexing_with_idx = [f"{i}. {term}" for i, term in enumerate(candidate_indexing)]
        prompt = self.prompt_service.build_best_taxonomy_indexing_prompt(
            candidate_terms="\n".join(candidate_indexing_with_idx),
            text=text
        )
        llm_answer = self.llm.run_query(prompt)
        print(llm_answer)
        best_index_str = self.llm.extract_answer_between_tags(llm_answer, "best_taxonomy_terms")
        best_indexes = [int(idx) for idx in best_index_str.split(",") if idx.isdigit() and int(idx) < len(candidate_indexing)]
        best_indexing_terms = [candidate_indexing[idx] for idx in best_indexes]
        return best_indexing_terms