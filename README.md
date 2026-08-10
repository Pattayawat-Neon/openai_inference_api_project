# OpenAI-Compatible Local Inference API Project

A self-hosted, GPU-accelerated local LLM inference pipeline using Docker and an **OpenAI-compatible API**, paired with a robust Python client suite for text summarization, language translation (EN <-> TH), and structured data extraction.

---

## 📁 Project Structure

```text
openai_inference_api_project/
├── app/
│   ├── client.py            # Python OpenAI SDK client (Summarization, Translation, Extraction)
│
├── .env           # Environment configuration template
└── docker-compose.yml   # Container orchestration for GPU-accelerated local LLM
├── .gitignore               # Git ignore rules for virtual environments, outputs, and caches
├── requirements.txt         # Python package dependencies (openai>=1.40.0)
└── README.md                # Project documentation
```

---

## ⚡ Key Features

- **OpenAI-Compatible API**: Self-hosted local LLM endpoint running on port `8000`.
- **NVIDIA GPU Acceleration**: Utilizes Docker GPU passthrough for low-latency, high-throughput inference.
- **Python Client Suite ([app/client.py](file:///c:/Users/neonp/Desktop/neon_project/openai_inference_api_project/app/client.py))**:
  - **Text Summarization**: Converts long text into formatted, factual bullet points.
  - **Language Translation**: English <-> Thai translation preserving tone and context.
  - **Structured Data Extraction**: Returns validated JSON objects conforming to predefined JSON schemas.
- **Resilience & Reliability**: Built-in timeout handling, bounded linear retry loops, and error logging.
- **Automated Output Storage**: Saves all client request outputs as timestamped JSON files in `./output/`.

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- Docker Desktop with WSL2 (Windows) or Docker Engine (Linux).
- NVIDIA GPU with up-to-date graphics drivers.
- Verify Docker can access your GPU:
  ```powershell
  docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
  ```

### 2. Configure Environment

Copy the template configuration file:

```bash
cp env.example .env
```

Default environment parameters:

| Variable | Description | Default Value |
| :--- | :--- | :--- |
| `MODEL_NAME` | Model ID to serve / request | `Qwen/Qwen2.5-7B-Instruct` |
| `HF_TOKEN` | Hugging Face access token (if using gated models) | `""` |
| `HF_CACHE_DIR` | Local directory for cached model weights | `./hf_cache` |

---

### 3. Launch the Local LLM Server

Navigate to the `app/` folder and start the container service using Docker Compose:

```bash
cd app
docker compose up -d
```

Verify container status and health check:

```bash
docker compose ps
```

Pull your target model (e.g., `qwen2.5:7b-instruct` or `qwen2.5:3b`):

```bash
docker exec -it local-llm-api ollama pull qwen2.5:7b-instruct
```

Verify endpoint health:

```bash
curl http://localhost:8000/api/tags
```

---

### 4. Setup Python Environment & Run Client

From the project root:

```bash
# Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the inference client demo
python app/client.py
```

---

## 💡 Python Client Usage

You can import and use the client functions in your own Python scripts:

```python
from app.client import summarize_text, translate_text, extract_structured_data

# 1. Summarization
summary = summarize_text("Your long text here...", bullet_points=5)
if summary.ok:
    print(summary.data)

# 2. Translation (EN -> TH)
translation = translate_text("Hello, how are you?", target_lang="th")
if translation.ok:
    print(translation.data)

# 3. Structured Extraction (JSON Schema)
extraction = extract_structured_data("Meeting notes or report text...")
if extraction.ok:
    print(extraction.data)  # Validated Python dictionary
```

### Environment Variable Overrides for Client

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LLM_API_BASE_URL` | Base URL of the OpenAI-compatible endpoint | `http://localhost:8000/v1` |
| `LLM_MODEL_NAME` | Target model name | `Qwen/Qwen2.5-7B-Instruct` |
| `LLM_TIMEOUT_S` | Request timeout in seconds | `120` |
| `LLM_MAX_RETRIES` | Maximum retry attempts | `3` |
| `LLM_OUTPUT_DIR` | Output JSON log directory | `./output` |

---

## 🛠️ Troubleshooting

- **Connection Refused (`APIConnectionError`)**: Ensure the Docker container is running (`docker compose ps` in `app/`).
- **GPU Not Detected**: Verify NVIDIA container runtime pass-through (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`).
- **Model Load Delay**: The initial request after pulling or starting may take a few seconds while loading weights into VRAM.
