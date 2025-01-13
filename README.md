# Car Manuals RAG Parameter Tuning

A tool for experimenting with RAG (Retrieval Augmented Generation) parameters on car manuals, helping find optimal chunk sizes and overlap values for document retrieval.

## Setup

1. **Create a Python Environment**

   Using venv:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

   Or using uv (faster):
   ```bash
   pip install uv
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Environment Variables**

   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage
```bash
streamlit run rag_finetune.py
```

Video reference here: https://youtu.be/nN8tAliaMW0