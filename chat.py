from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from typing import Iterator
from llama_index.core import Settings

#Chatbot with stream response
class Chat:
    def __init__(self, token_limit: int = 3000, **kwargs) -> None:
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=3000, **kwargs)
    
    def query(self, user_input: str) -> Iterator[str]:
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