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
        # 初始化重排序模型
        self.reranker = CrossEncoder(
            self.config.rerank_base_model,
            automodel_args={"torch_dtype": "auto"},
        )

    def build_document_recall_indexing(self, text_list):
        """
        构建文档召回索引并保存到文件

        Args:
            text_list: 文本列表
        """
        # 去重并保留原始索引映射
        unique_texts = []
        original_indices = []
        seen_texts = set()

        for idx, text in enumerate(text_list):
            if text not in seen_texts:
                unique_texts.append(text)
                original_indices.append(idx)
                seen_texts.add(text)

        # 创建文档对象
        documents = [Document(page_content=text, metadata={'original_index': original_indices[idx]})
                    for idx, text in enumerate(unique_texts)]

        # 使用FAISS创建向量存储
        vector_store = FAISS.from_documents(documents, self.recall_embeddings)

        # 保存FAISS索引到文件
        vector_store.save_local(self.config.recall_index_save_path)

        # 保存文本列表和索引映射到文件
        text_data = {
            'texts': unique_texts,
            'original_indices': original_indices
        }
        text_save_path = os.path.join(os.path.dirname(self.config.recall_index_save_path), 'text_data.json')
        with open(text_save_path, 'w', encoding='utf-8') as f:
            json.dump(text_data, f, ensure_ascii=False, indent=2)

        print(f"成功构建了包含 {len(documents)} 个文档的向量索引")
        print(f"FAISS索引已保存到: {self.config.recall_index_save_path}")
        print(f"文本数据已保存到: {text_save_path}")

    def recall_related_document(self, query, top_n=10, search_type='mmr', lambda_mult=0.25):
        """
        从文件加载FAISS索引并召回相关文档

        Args:
            query: 查询文本
            top_n: 返回的文档数量
            search_type: 搜索类型
            lambda_mult: 多样性参数

        Returns:
            召回的相关文档文本列表
        """
        # 检查索引文件是否存在
        if not os.path.exists(self.config.recall_index_save_path):
            raise ValueError(f"FAISS索引文件不存在: {self.config.recall_index_save_path}，请先调用build_document_recall_indexing构建索引")

        # 检查文本数据文件是否存在
        text_save_path = os.path.join(os.path.dirname(self.config.recall_index_save_path), 'text_data.json')
        if not os.path.exists(text_save_path):
            raise ValueError(f"文本数据文件不存在: {text_save_path}，请先调用build_document_recall_indexing构建索引")

        # 从文件加载FAISS索引
        vector_store = FAISS.load_local(self.config.recall_index_save_path, self.recall_embeddings,
                                         allow_dangerous_deserialization=True)

        # 从文件加载文本数据
        with open(text_save_path, 'r', encoding='utf-8') as f:
            text_data = json.load(f)

        texts = text_data['texts']

        # 创建检索器
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={'k': top_n, 'lambda_mult': lambda_mult, 'fetch_k': top_n}
        )

        # 获取相关文档
        results = retriever.get_relevant_documents(query)

        # 提取召回的文本
        recalled_texts = [result.page_content for result in results]

        return recalled_texts

    def rerank_related_document(self, query, documents, top_n=10):
        """
        使用CrossEncoder对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回的top n文档数量

        Returns:
            重排序后的top n文档列表
        """
        if not documents:
            return []

        if len(documents) <= top_n:
            # 如果文档数量不超过top_n，仍然进行重排序
            top_n = len(documents)

        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]

        # 使用CrossEncoder计算相关性分数
        scores = self.reranker.predict(pairs)

        # 将文档和分数配对并按分数降序排序
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 返回top n个文档
        reranked_documents = [doc for doc, score in doc_score_pairs[:top_n]]

        return reranked_documents
