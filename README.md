# KIT Toolbox LLM Setup Project

A clean, interactive, and production-ready Python project to experiment with open-source LLM APIs from the KIT toolbox.

## Features

- **Virtual Environment (`.venv`)**: Pre-configured environment to keep dependencies isolated.
- **Environment Variables (`.env`)**: Credentials stored securely in a local `.env` file (ignored by git for security).
- **Flexible CLI (`main.py`)**: Easily query different models, customize prompts, or change system behaviors using command-line arguments.

---

## Getting Started

### 1. Virtual Environment Setup
A virtual environment has already been created in this workspace, and the required dependencies (`openai`, `python-dotenv`) have been installed. 

To activate the virtual environment in your terminal:

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```powershell
.venv\Scripts\activate
```

---

## How to Run

You can run the script using the environment's Python interpreter directly:

### 1. Run the default query (Rayleigh-Streuung)
Using the default model `kit.gemma4-31b-it`:
```bash
python main.py
```

### 2. See all CLI options and available models
```bash
python main.py --help
```

### 3. Query a different model
For example, using the `kit.mistral-small-4-119b-a8b` model:
```bash
python main.py --model kit.mistral-small-4-119b-a8b
```

### 4. Custom Prompt and System Message
```bash
python main.py \
  --model kit.gemma4-31b-it \
  --system "Du bist ein präziser Physiker." \
  --prompt "Was ist die Lichtgeschwindigkeit im Vakuum?"
```

---

## Configuration

The API credentials are loaded from the `.env` file in the root of the directory:
- `KIT_API_KEY`: Your KIT Employee API Key.
- `KIT_BASE_URL`: The KIT Toolbox gateway endpoint (`https://ki-toolbox.scc.kit.edu/api/v1`).

---

## Available Models

- `kit.gemma4-31b-it` (default)
- `kit.gpt-oss-120b`
- `kit.minimax-m2.5-229b`
- `kit.minimax-m2.7-229b`
- `kit.mistral-small-4-119b-a8b`
