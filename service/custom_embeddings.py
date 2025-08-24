from typing import List
from sentence_transformers import SentenceTransformer

class CustomEmbeddings:
    def __init__(self, model: str, batch_size: int = 2):
        print(f"Loading model: {model}")
        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.batch_size = batch_size  # Add a batch_size attribute

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Vectorize the input text list at once and process in batches (batch_size) within the model.
        SentenceTransformer.encode(texts, batch_size=...) returns a 2D numpy array,
        we just need to convert it to a list.
        """
        # If texts is very large, it will be processed in batches according to self.batch_size
        embeddings = self.model.encode(texts, batch_size=self.batch_size)
        return embeddings.tolist()  # Convert to list[List[float]]

    def embed_query(self, query: str) -> List[float]:
        """
        embed_query generally only processes a single text, can directly call batch_size=1 or irrelevant to batch_size.
        If you want to also utilize batch_size, you can pass in a list containing only the query.
        """
        embedding = self.model.encode([query], batch_size=self.batch_size)
        return embedding[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)