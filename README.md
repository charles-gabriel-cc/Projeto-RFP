# Projeto-RFP
#### Grupo: Charles Gabriel, Eduardo Melo, Jailson Soares e Josef Jaeger

Aplicativo que implementa ocr + raga para assistência, validação e busca de produtos em listas de compras em um estoque, recomendando produtos existentes ou similares a produtos existentes na lista e contexto recuperado

# Requisitos
- **Python**: 3.10.0
- **CUDA**: 11.8 (se o modelo que você usará for executado localmente)
- **Ollama**: 11.8

## Instalação
```bash
# Clone o repositório
$ git clone https://github.com/charles-gabriel-cc/Projeto-RFP/tree/main
```

```bash
# Instale as dependências
$ pip install -r requirements.txt
```

```bash
# Para rodar localmente
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Uso

Para usar o pipeline você precisa configurar o modelo escolhido, aqui usamos um modelo ollama, mas você pode configurar um modelo huggingface ou um modelo openAI, veja a documentação do llama_index para mais informações

```python
from llama_index.core import Settings

model = "phi4:latest"

Settings.embed_model = OllamaEmbedding(
    model_name="all-minilm:latest",
    ollama_additional_kwargs={"mirostat": 0},
)

Settings.llm = Ollama(model=model, 
                      request_timeout=210,
                      temperature=0.2)
```
Depois deve-se criar uma pasta docs para armazenar os arquivos que serão processados no RAG, é possivel definir loaders especificos para outros formatos, mas deve-se usar JSON ou txt

Para rodar a interface com o gradio basta usar:
```
$ python app.py
```
