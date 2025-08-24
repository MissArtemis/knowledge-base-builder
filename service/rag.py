from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import os
import json

from service.custom_embeddings import CustomEmbeddings
from utils.config import ConfigManager


class RAGService:
    def __init__(self):
        self.config = ConfigManager()
        self.recall_embeddings = CustomEmbeddings(self.config.recall_base_model)
        # Initialize rerank model
        self.reranker = CrossEncoder(
            self.config.rerank_base_model,
            automodel_args={"torch_dtype": "auto"},
        )

    def build_document_recall_indexing(self, text_list):
        """
        Build document recall index and save to file

        Args:
            text_list: List of texts
        """
        # Deduplicate and keep original index mapping
        unique_texts = []
        original_indices = []
        seen_texts = set()

        for idx, text in enumerate(text_list):
            if text not in seen_texts:
                unique_texts.append(text)
                original_indices.append(idx)
                seen_texts.add(text)

        # Create document objects
        documents = [Document(page_content=text, metadata={'original_index': original_indices[idx]})
                    for idx, text in enumerate(unique_texts)]

        # Create vector store using FAISS
        vector_store = FAISS.from_documents(documents, self.recall_embeddings)

        # Save FAISS index to file
        vector_store.save_local(self.config.recall_index_save_path)

        # Save text list and index mapping to file
        text_data = {
            'texts': unique_texts,
            'original_indices': original_indices
        }
        text_save_path = os.path.join(os.path.dirname(self.config.recall_index_save_path), 'text_data.json')
        with open(text_save_path, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=2)

        print(f"Successfully built vector index containing {len(documents)} documents")
        print(f"FAISS index saved to: {self.config.recall_index_save_path}")
        print(f"Text data saved to: {text_save_path}")

    def recall_related_document(self, query, top_n=10, search_type='mmr', lambda_mult=0.25):
        """
        Load FAISS index from file and recall related documents

        Args:
            query: Query text
            top_n: Number of documents to return
            search_type: Search type
            lambda_mult: Diversity parameter

        Returns:
            List of recalled related document texts
        """
        # Check if index file exists
        if not os.path.exists(self.config.recall_index_save_path):
            raise ValueError(f"FAISS index file does not exist: {self.config.recall_index_save_path}, please call build_document_recall_indexing first to build the index")

        # Check if text data file exists
        text_save_path = os.path.join(os.path.dirname(self.config.recall_index_save_path), 'text_data.json')
        if not os.path.exists(text_save_path):
            raise ValueError(f"Text data file does not exist: {text_save_path}, please call build_document_recall_indexing first to build the index")

        # Load FAISS index from file
        vector_store = FAISS.load_local(self.config.recall_index_save_path, self.recall_embeddings,
                                         allow_dangerous_deserialization=True)

        # Load text data from file
        with open(text_save_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        texts = text_data['texts']

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={'k': top_n, 'lambda_mult': lambda_mult, 'fetch_k': top_n}
        )

        # Get relevant documents
        results = retriever.get_relevant_documents(query)

        # Extract recalled texts
        recalled_texts = [result.page_content for result in results]

        return recalled_texts

    def rerank_related_document(self, query, documents, top_n=10):
        """
        Rerank documents using CrossEncoder

        Args:
            query: Query text
            documents: List of documents
            top_n: Number of top n documents to return

        Returns:
            List of top n documents after reranking
        """
        if not documents:
            return []

        if len(documents) <= top_n:
            # If document count doesn't exceed top_n, still perform reranking
            top_n = len(documents)

        # Build query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Use CrossEncoder to calculate relevance scores
        scores = self.reranker.predict(pairs)

        # Pair documents with scores and sort by score in descending order
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top n documents
        reranked_documents = [doc for doc, score in doc_score_pairs[:top_n]]

        return reranked_documents
