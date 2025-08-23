import os

from tqdm import tqdm

from service.llm import LLMService
from service.rag import RAGService
from utils.config import ConfigManager
from utils.db.mock_db import MockDB

tqdm.pandas()

class KAGApp:
    def __init__(self):
        self.config = ConfigManager()
        project_root = self.config.project_root
        os.chdir(project_root)
        self.db = MockDB()
        self.llm_service = LLMService()
        self.rag_service = RAGService()

    def run(self):
        print("KAG app starting...")
        doc_df = self.db.doc_data.head(10)

        # 存储累积的indexing_terms
        accumulated_indexing_terms = []

        # 为每一行生成indexing_terms
        indexing_results = []
        for idx, row in tqdm(doc_df.iterrows(), total=len(doc_df), desc="Processing documents"):
            print(f"\nProcessing row {idx}")

            # 如果有累积的索引词，使用RAG召回相关的top 20作为候选
            if accumulated_indexing_terms:
                try:
                    # 构建索引（基于累积的索引词）
                    self.rag_service.build_document_recall_indexing(accumulated_indexing_terms)

                    # 召回与当前文档相关的top 20索引词
                    candidate_indexing = self.rag_service.recall_related_document(
                        query=row['text'],
                        top_n=10
                    )
                    print(f"Recalled {len(candidate_indexing)} candidate indexing terms for row {idx}")
                except Exception as e:
                    print(f"RAG recall failed for row {idx}: {e}")
                    # 如果召回失败，使用原有逻辑
                    candidate_indexing = accumulated_indexing_terms.copy()
            else:
                # 第一行没有累积词汇，使用空列表
                candidate_indexing = []
                print(f"No accumulated terms yet, using empty candidate_indexing for row {idx}")

            # generate_taxonomy_indexing 返回的是一个列表
            indexing_terms = self.llm_service.generate_taxonomy_indexing(
                text=row['text'],
                candidate_indexing=candidate_indexing
            )

            print(f"Generated indexing terms: {indexing_terms}")

            # 添加到结果列表
            indexing_results.append(indexing_terms)

            # 将当前行的结果添加到累积列表中，供下一行使用
            accumulated_indexing_terms.extend(indexing_terms)
            print(f"Accumulated indexing terms count: {len(accumulated_indexing_terms)}")

        # 将结果添加到DataFrame
        doc_df['indexing_terms'] = indexing_results

        # 显示最终结果
        print("\nFinal results:")
        for idx, row in doc_df.iterrows():
            print(f"Row {idx}: {len(row['indexing_terms'])} indexing terms")

        print(doc_df[['text', 'indexing_terms']].head())



if __name__ == "__main__":
    app = KAGApp()
    app.run()