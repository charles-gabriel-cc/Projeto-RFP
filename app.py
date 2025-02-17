import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate, Settings
import gradio as gr
from llama_index.readers.json import JSONReader
from typing import Iterator
from rag import productRag
from chat import Chat

Settings.embed_model = HuggingFaceEmbedding(
    model_name="all-MiniLM-L6-v2",
    device="cuda:0",
)

Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    tokenizers_name="meta-llama/Llama-3.2-3B-Instruct",
    device_map="cuda:0",
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    max_new_tokens=100
)

json_reader = JSONReader(
    levels_back=None,
    is_jsonl=True,
    clean_json=True,
)

class Agent():
    def __init__(self, data_dir="docs", file_extractor: dict = None) -> None:
        self.ragreranker = productRag(data_dir, file_extractor)
        self.chat = Chat()
        self.retrieved_Documents = {}
            
    def query(self, query_str) -> Iterator[str]:
        context = self.ragreranker.retrieve_documents(query_str)
        documents = [node.node.get_text() for node in context]

        return self.chat.query(query_str)
    
agent = Agent(file_extractor={".jsonl": json_reader})

def chat_response(user_input):
    return agent.query(user_input)

def chat_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 LLama Rag Reranker")
        
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Digite sua pergunta aqui...")
        send_button = gr.Button("Enviar")
        
        def respond(chat_history, user_message):
            chat_history.append((user_message, ""))  # Adiciona mensagem do usuário
            bot_message = ""
            for chunk in chat_response(user_message):
                bot_message += chunk
                chat_history[-1] = (user_message, bot_message)
                yield chat_history  # Atualiza a interface dinamicamente
        
        send_button.click(respond, [chatbot, user_input], chatbot)
        user_input.submit(respond, [chatbot, user_input], chatbot)
        
    return demo

if __name__ == "__main__":
    chat_interface().launch(share=True)