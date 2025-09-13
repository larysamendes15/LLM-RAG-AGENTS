FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy reqs first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Default envs
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODEL=llama3.1:8b
ENV EMBEDDINGS_MODEL=thenlper/gte-small
ENV VECTORSTORE=chroma
ENV PERSIST_DIR=data/chroma_reforma_textos_legais

# Health: show versions
RUN python -V && python -c "import langchain,langgraph,chromadb; print('OK: deps loaded')"

# Start app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
