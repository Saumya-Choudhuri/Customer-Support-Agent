#!/bin/bash
set -e
echo "Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

echo "Starting Ollama server in background..."
ollama serve &
sleep 5