# Ollama for Lyric Generation

As an alternative to running LLM models with GGUF format in `llama-cpp-python`, Ollama models can be used for lyric generation.

## Prerequisite

```
pip install ollama
```

## Usage

For using Ollama, add the following command line arguments:
- `--ollama`
- `--model_path <Ollama model name>`, e.g. `--model_path gemma3:12b-it-q4_K_M`
 