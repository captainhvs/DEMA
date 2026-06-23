# Results


# Setup Guide

This part describes how to prepare the environment, launch the required background services, install the project dependencies, and download the required checkpoints.

## 1. Prerequisites

Before running this project, install and configure the following tools:

- **Ollama**: local model runtime used to serve Llama 3.3 and embedding models.
  - Official website: <https://ollama.com/>
  - Documentation: <https://docs.ollama.com/>
  - Linux installation guide: <https://docs.ollama.com/linux>

- **Llama Stack**: local OpenAI-compatible API server and agent/RAG runtime.
  - PyPI: <https://pypi.org/project/llama-stack/>
  - Documentation: <https://llamastack.github.io/>
  - GitHub: <https://github.com/llamastack/llama-stack>


## 2. Install Ollama

On Linux, install Ollama with:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:

```bash
ollama serve
```

In another terminal, verify that Ollama is available:

```bash
ollama -v
```

## 3. Register / Download Llama 3.3 with Ollama

Pull the Llama 3.3 model:

```bash
ollama pull llama3.3
```

You can verify that the model has been downloaded with:

```bash
ollama list
```

Optional quick test:

```bash
ollama run llama3.3
```

## 4. Install Llama Stack

You can install Llama Stack using `uv`:

```bash
uv pip install llama-stack
```




## 5. Create the Main Project Environment

From the project root directory, create the main virtual environment from `requirements.txt`. We recommend using uv to manage venv.


```bash
uv sync
source .venv/bin/activate
```

The main project Python path is:

```text
./.venv/bin/python
```


## 6. Create a Separate CLAP Environment

The CLAP dependencies should be installed in a separate virtual environment. Do **not** reuse `.venv`.

Recommended environment name:

```text
.venv-clap
```

Create and install the CLAP dependencies:

```bash
python -m venv .venv-clap
source .venv-clap/bin/activate
pip install -r requirements-clap.txt
```

The relative Python path for subprocess calls is:

```text
./.venv-clap/bin/python
```



## 7. Download Checkpoints

Create a `ckpt/` directory in the project root:

```bash
mkdir -p ckpt
```

Recommended checkpoint layout:

```text
ckpt/
├── nomic-embed-text-v1.5/
├── Qwen-Audio-Chat/
└── clap/
    └── 630k-audioset-fusion-best.pt
```

### 7.1 Download `nomic-embed-text-v1.5`

Hugging Face model page:

<https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>

Download with Hugging Face CLI:

```bash
huggingface-cli download nomic-ai/nomic-embed-text-v1.5 \
  --local-dir ckpt/nomic-embed-text-v1.5
```


### 7.2 Download `Qwen-Audio-Chat`

Hugging Face model page:

<https://huggingface.co/Qwen/Qwen-Audio-Chat>

Download with Hugging Face CLI:

```bash
huggingface-cli download Qwen/Qwen-Audio-Chat \
  --local-dir ckpt/Qwen-Audio-Chat
```

### 7.3 Download CLAP checkpoint: `630k-audioset-fusion-best.pt`

Hugging Face file page:

<https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-fusion-best.pt>



Download with Hugging Face CLI:

```bash
huggingface-cli download lukewys/laion_clap 630k-audioset-fusion-best.pt \
  --local-dir ckpt/clap
```

## 8. Minimal Startup Checklist

This project expects both Ollama and Llama Stack to be running in different terminal continuously, you can manage them with tmux.

After completing the setup, start the required services:

```bash
# Terminal 1
ollama serve
```

```bash
# Terminal 2
llama stack run starter
```


Then activate the main project environment and run your project entrypoint:

```bash
source .venv/bin/activate
python your_main_script.py
```

When invoking CLAP-related scripts through a subprocess, use:

```text
./.venv-clap/bin/python
```

### Optional: Run with a custom Ollama endpoint

If you run Ollama on a custom host/port, for example `127.0.0.1:11445`, start Ollama with:

```bash
export OLLAMA_HOST=127.0.0.1:11445
ollama serve
```

If Ollama is running on a non-default port, export the endpoint before starting Llama Stack:

```bash
export OLLAMA_URL=http://127.0.0.1:11445/v1
llama stack run starter
```

## 10. Notes

- Keep the main project environment and the CLAP environment separate to avoid dependency conflicts.
- Make sure the checkpoint paths used in the code match the `ckpt/` layout shown above.
- If Ollama is already managed by `systemd`, `tmux`, `screen`, or another process manager, do not start a second `ollama serve` instance on the same port.
- If you use a non-default Ollama port, make sure both `OLLAMA_HOST` and `OLLAMA_URL` are configured consistently.








