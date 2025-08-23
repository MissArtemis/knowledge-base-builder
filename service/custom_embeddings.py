from typing import List
from sentence_transformers import SentenceTransformer

class CustomEmbeddings:
    def __init__(self, model: str, batch_size: int = 2):
        print(f"Loading model: {model}")
        self.model = SentenceTransformer(model, trust_remote_code=True)
        self.batch_size = batch_size  # 新增一个 batch_size 属性

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将传入的文本列表一次性进行向量化，并在模型内部分批 (batch_size) 处理。
        SentenceTransformer.encode(texts, batch_size=...) 会返回一个 2D numpy array，
        我们只需转成 list 即可。
        """
        # 如果 texts 很大，就会根据 self.batch_size 分批处理
        embeddings = self.model.encode(texts, batch_size=self.batch_size)
        return embeddings.tolist()  # 转换为 list[List[float]]

    def embed_query(self, query: str) -> List[float]:
        """
        embed_query 一般只对单个文本处理，可直接调用 batch_size=1 或和 batch_size 无关。
        如果你想也利用 batch_size，可以传入一个只含 query 的列表。
        """
        embedding = self.model.encode([query], batch_size=self.batch_size)
        return embedding[0].tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)