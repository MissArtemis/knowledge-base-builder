import os

from tqdm import tqdm
import pandas as pd
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

    def build(self):
        print("KAG app starting...")
        doc_df = self.db.doc_data.head(10)

        # Store accumulated indexing terms
        accumulated_indexing_terms = []

        # Generate indexing terms for each row
        indexing_results = []
        for idx, row in tqdm(doc_df.iterrows(), total=len(doc_df), desc="Processing documents"):
            print(f"\nProcessing row {idx}")

            # If there are accumulated index terms, use RAG to recall related top 20 as candidates
            if accumulated_indexing_terms:
                try:
                    # Build index (based on accumulated indexing terms)
                    self.rag_service.build_document_recall_indexing(accumulated_indexing_terms)

                    # Recall top 20 indexing terms related to current document
                    candidate_indexing = self.rag_service.recall_related_document(
                        query=row['text'],
                        top_n=10
                    )
                    print(f"Recalled {len(candidate_indexing)} candidate indexing terms for row {idx}")
                except Exception as e:
                    print(f"RAG recall failed for row {idx}: {e}")
                    # If recall fails, use original logic
                    candidate_indexing = accumulated_indexing_terms.copy()
            else:
                # First row has no accumulated vocabulary, use empty list
                candidate_indexing = []
                print(f"No accumulated terms yet, using empty candidate_indexing for row {idx}")

            # generate_taxonomy_indexing returns a list
            indexing_terms = self.llm_service.generate_taxonomy_indexing(
                text=row['text'],
                candidate_indexing=candidate_indexing
            )

            print(f"Generated indexing terms: {indexing_terms}")

            # Add to results list
            indexing_results.append(indexing_terms)

            # Add current row results to accumulated list for next row
            accumulated_indexing_terms.extend(indexing_terms)
            print(f"Accumulated indexing terms count: {len(accumulated_indexing_terms)}")
        self.rag_service.build_document_recall_indexing(accumulated_indexing_terms)
        # Add results to DataFrame
        doc_df['indexing_terms'] = indexing_results

        # Display final results
        print("\nFinal results:")
        for idx, row in doc_df.iterrows():
            print(f"Row {idx}: {len(row['indexing_terms'])} indexing terms")

        print(doc_df[['text', 'indexing_terms']].head())

        indexing_df = doc_df.explode('indexing_terms')[['text', 'indexing_terms']].reset_index(drop=True)
        print(indexing_df)
        indexing_df.to_pickle(self.config.kb_indexing_path)

    def query(self, query_text):
        if not os.path.exists(self.config.kb_indexing_path):
            raise ValueError(f"Indexing file does not exist: {self.config.kb_indexing_path}, please run build() first.")

        indexing_df = pd.read_pickle(self.config.kb_indexing_path)
        print(f"Loaded indexing data with {len(indexing_df)} entries.")

        # Use RAG to recall related top 20 indexing_terms
        self.rag_service.build_document_recall_indexing(indexing_df['indexing_terms'].tolist())
        recalled_terms = self.rag_service.recall_related_document(
            query=query_text,
            top_n=20
        )
        print(f"Recalled {len(recalled_terms)} terms for the query.")

        # Use rerank to reorder recalled terms and get top 5
        reranked_terms = self.rag_service.rerank_related_document(
            query=query_text,
            documents=recalled_terms,
            top_n=5
        )
        print(f"Reranked to {len(reranked_terms)} top terms.")

        best_terms = self.llm_service.generate_best_taxonomy_indexing(
            text=query_text,
            candidate_indexing=reranked_terms)

        # Filter DataFrame to get rows corresponding to these top 5 indexing terms
        relevant_rows = indexing_df[indexing_df['indexing_terms'].isin(best_terms)]
        print(f"Found {len(relevant_rows)} relevant rows in the indexing data.")

        return relevant_rows



if __name__ == "__main__":
    app = KAGApp()
    app.build()
    query = "social security and tax"
    results = app.query(query)
    print(f"\nQuery results for '{query}':")
    print(results)