#!/bin/bash
set -e

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -f http://localhost:11434/api/version > /dev/null 2>&1; do
    sleep 2
done

# Pull the llama3.2:3b model if it doesn't exist
echo "Checking for model: llama3.2:3b"
if ! ollama list | grep -q "llama3.2:3b"; then
    echo "Pulling model: llama3.2:3b (this may take several minutes...)"
    ollama pull llama3.2:3b
    echo "Model llama3.2:3b downloaded successfully"
else
    echo "Model llama3.2:3b already exists"
fi


echo "LLM service ready"

# Keep the container running
wait $OLLAMA_PID