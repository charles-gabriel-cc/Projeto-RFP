from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from typing import Iterator
from llama_index.core import Settings
from llama_index.core import PromptTemplate


RECOMMENDATION_PROMPT = PromptTemplate(
    """
You are a chatbot specialized in recommending products. The user will provide a list of desired products in their preferred language, and you have access to a predefined list of relevant products: {list}. Your task is to find similar products from this list and suggest the best alternative for each item requested by the user. 

**IMPORTANT:**  
- Only recommend a product if it exists in the provided list.  
- Match the user's request as closely as possible.  
- Use available product information to justify your recommendation.  
- Do not describe your analysis process.  
- **Respond in the same language as the user.**  

Format your response for better readability using bullet points and bold text. Example format:

**Recommended Alternative:**  
- **Product Name:** [Suggested Product]  
- **Key Features:**  
  - Feature 1  
  - Feature 2  
  - Feature 3  

If there is no product in the list, inform the user that the product is not available.
- **Product Name:** 
    - Product is not available

If multiple alternatives exist, suggest the most relevant one. If no suitable match is found, politely inform the user.  

**Always respond in the same language as the user.**
    """
)


#Chatbot with stream response
class Chat:
    def __init__(self, token_limit: int = 3000, **kwargs) -> None:
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000, **kwargs)
    
    def query(self, user_input: str, context: str) -> Iterator[str]:
        system_message = ChatMessage(role=MessageRole.SYSTEM, content=RECOMMENDATION_PROMPT.format(list=context))
        self.memory.put(system_message)
        user_message = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_message)
        
        chat_history = self.memory.get_all()
        
        response_gen = Settings.llm.stream_chat(chat_history)

        full_response = ""
        for chunk in response_gen:
            print(chunk.delta, end="", flush=True)
            full_response += chunk.delta
            yield chunk.delta
        
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        self.memory.put(assistant_message)