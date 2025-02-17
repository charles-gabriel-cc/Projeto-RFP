from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle, Document
import logging
from llama_index.core.postprocessor import LLMRerank
from llama_index.readers.json import JSONReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore


class productRag():
    def __init__(self, data_dir: str = "docs") -> None:
        self.data_dir = data_dir
        self.documents: list[Document]
        self.index: VectorStoreIndex
        self.retriever: VectorIndexRetriever
        self.reranker: LLMRerank

    def _start(self):
        json_reader = JSONReader(
            levels_back=None,
            is_jsonl=True,
            clean_json=True,
        )
        reader = SimpleDirectoryReader(
            self.data_dir,
            file_extractor={"jsonl": json_reader},
            num_files_limit=1,
            recursive=True,
            filename_as_id=True
        )

        self.documents = reader.load_data()

        self.index = VectorStoreIndex.from_documents(
            self.documents,
            show_progress=True
        )

        self.reranker = LLMRerank(
            choice_batch_size=5,
            top_n=3,    
        )

    def retrieve(self, query: str) -> list[NodeWithScore]:
        query_bundle = QueryBundle(query)
        retrieved_nodes = self.retriever.retrieve(query_bundle)
        retrieved_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )
        relevant_documents = self._relevancy(query, retrieved_nodes)

        return relevant_documents