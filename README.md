# Projeto-RFP
#### Grupo: Charles Gabriel, Eduardo Melo, Jailson Soares e Josef Jaeger

Assistência para validação e busca de produtos em listas de compras em fornecedores versus lista de produtos

# Requisitos
- **Python**: 3.10.0
- **CUDA**: 11.8
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

## Usage

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

## Instruções para build 
Veja [Guia de Build](BUILD.md) para detalhes.

## Contribuição
Veja [Guia de Contribuição](CONTRIBUTING.md) para detalhes.
