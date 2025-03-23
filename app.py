import torch

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import PromptTemplate, Settings
import gradio as gr
from llama_index.readers.json import JSONReader
from typing import Iterator
from src import productRag, Chat, StructuredOutput
import instructor
from pydantic import BaseModel, Field

import base64
import requests
from PIL import Image
import os

from dotenv import load_dotenv
load_dotenv()

import openai


class output_format(BaseModel):
    product_list: list[str] = Field(..., description="All products the user wants to buy listed")

SYSTEM_PROMPT = """Act as an OCR assistant. Analyze the provided image and:
1. Recognize all visible text in the image as accurately as possible.
2. Maintain the original structure and formatting of the text.
Extract all the products listed in a list of str
Provide only the transcription without any additional comments."""

RECOMMENDATION_PROMPT = PromptTemplate(
    """
        You are a chatbot specialized in recommending products. The user will provide a list of desired products in their preferred language, and you have access to a predefined list of relevant products: {list}. Your task is to find similar products from this list and suggest the best alternative for each item requested by the user. 

        **IMPORTANT:**  
        - Only recommend a product if it exists in the provided list.  
        - Match the user's request as closely as possible.  
        - Use available product information to justify your recommendation.  
        - Do not describe your analysis process.  
        - You need to strict response as the format described below.
        - ALWAYS respond in the same language as the user and translate the response format to the user's language.

        Format your response for better readability using bullet points and bold text. Example format:

            **Recommended Alternative:**  
            - **{User's Product Name}**:
                - Suggested product: {Name of the recommended product}
                    - Feature 1  
                    - Feature 2
                    - Feature 3
                ...

            If there is no product in the list, inform the user that the product is not available.:
            
            - **{User's Product Name}**: 
                - Product is not available

        If multiple alternatives exist, suggest the most relevant one. If no suitable match is found, politely inform the user.  
    """
)

model = "phi4:latest"

Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

llm = OpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.7,
)

vision_model = OpenAI(model="gpt-4-turbo", temperature=0.2)

json_reader = JSONReader(
    levels_back=None,
    is_jsonl=True,
    clean_json=True,
)

def encode_image_to_base64(image_path):
    """Convert an image file to a base64 encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Agent():
    def __init__(self, data_dir="docs", ragreranker: productRag = None) -> None:
        if ragreranker is None:
            self.ragreranker = productRag(data_dir)
        else:
            self.ragreranker = ragreranker
        self.chat = Chat()
        self.retrieved_Documents = {}
            
    def query(self, query_str):
        context = self.ragreranker.retrieve(str(query_str))
        documents = [node.node.get_text() for node in context]

        prompt = RECOMMENDATION_PROMPT.format(list=documents)

        query_prompt = prompt + f"""
                                <User query>
                                {str(query_str)}
                                </User query>
                                """
        
        return Settings.llm.complete(prompt + 
                                     f"""
                                     <User query>
                                     {str(query_str)}
                                     </User query>
                                     """)

ragreranker = productRag("docs")

agent = Agent(ragreranker=ragreranker)

structured_output = StructuredOutput(output_format=output_format, system_prompt="Extract all the products listed in a list of str")

recommended_output = StructuredOutput(output_format=output_format, system_prompt= "You need to identify the products that are recommended, you will receive a list of products, the products indented just below the product are the recommended ones, create a list ONLY with the recommended products")

def chat_response(user_input):
    return agent.query(user_input)

def clear_and_restart():
    global agent
    agent = Agent(ragreranker=ragreranker)
    return None

def chat_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 OCR Assistant")
        
        image_input = gr.Image(type="filepath", label="Envie sua imagem aqui...")
        output_text = gr.Textbox(label="Resposta da LLM", interactive=False)
        
        def process_image(image_path):
            # Verificar se image_path é None
            if image_path is None:
                return "Nenhuma imagem foi enviada."

            # Criar a pasta 'images' se não existir
            os.makedirs('images', exist_ok=True)

            # Salvar a imagem na pasta 'images'
            save_path = os.path.join('images', os.path.basename(image_path))
            
            #query = SYSTEM_PROMPT.format(image=encoded_image)
            # Chamar a LLM e capturar a resposta
            
            client = openai.Client(api_key=os.environ['OPENAI_API_KEY'],) 

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Analyze the following image:"},
                    {"role": "user", "content": "Provide an analysis based on the image provided."}
                ],
                temperature=0.2
            )

            reponse_content = response.choices[0].message.content
            print("=====================")
            print(reponse_content)
            print("=====================")

            products = structured_output.query(reponse_content)

            full_response = agent.query(products.product_list)

            recomendation_list = recommended_output.query(full_response.text)
            
            # Debug: imprimir a resposta para verificar o que está sendo retornado
            print("OCR:", response.message.content)
            print("Instructor:", products.product_list)
            print("Resposta da llm:", full_response.text)
            print("Lista de recomendados: ", recomendation_list.product_list)

            result = f""""
            OCR: {response.message.content}
            Instructor: {products.product_list}
            Response: {full_response.text}
            Recommended list: {recomendation_list.product_list}
            """
            return result

        
        image_input.change(process_image, image_input, output_text)
        
    return demo

if __name__ == "__main__":
    chat_interface().launch(share=True)