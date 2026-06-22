#!/bin/bash
set -e
echo "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

echo "Starting Ollama server in background..."
ollama serve &
sleep 5

echo "Pulling LLM model (llama3.2:3b — ~2GB, free)..."
ollama pull llama3.2:3b

echo "Pulling embedding model (nomic-embed-text — ~274MB, free)..."
ollama pull nomic-embed-text

echo "Setup complete. Models ready."