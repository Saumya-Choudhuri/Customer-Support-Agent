#!/bin/bash
set -e
echo "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

echo "Starting Ollama server in background..."
ollama serve &
sleep 5

echo "Pulling LLM model (llama3.2:3b - ~2GB, free)..."
olllama pull llama3.2:3b
