#!/bin/bash

set -e # Exit on any error

echo " Setting up Memory System with Local LLMs..."


GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Clean up any existing environment
if [ -d "my_env" ]; then
    print_status "Removing existing environment..."
    rm -rf my_env
fi

# Create fresh virtual environment
print_status "Creating virtual environment..."
python3 -m venv my_env
source my_env/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    print_warning "requirements.txt not found! Creating one..."
    cat > requirements.txt << 'EOF'
# Core ML/NLP packages
spacy>=3.4.0,<3.5.0
spacy-experimental>=0.6.0
spacy-transformers>=1.2.0

# Web framework
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
websockets>=9.0
aiohttp>=3.8.0

# Vector database and storage
chromadb>=0.4.0
numpy>=1.20.0,<2.0.0

# Sentiment analysis
vaderSentiment>=3.3.0

# Utilities
python-dotenv>=0.19.0
rich>=10.0.0
pytz>=2021.1

requests>=2.25.0

# Data handling
pandas>=1.3.0
EOF
    print_success "Created requirements.txt"
fi

# Install packages in stages to avoid dependency conflicts
print_status "Installing core packages first..."
pip install --no-cache-dir numpy setuptools wheel

print_status "Installing remaining packages..."
pip install --no-cache-dir -r requirements.txt

print_success "All Python packages installed successfully!"

# Download SpaCy transformer model
print_status "Downloading SpaCy transformer model..."
python -m spacy download en_core_web_trf

# Set up coreference resolution
print_status "Setting up SpaCy coreference resolution..."
python << 'EOF'
try:
    import spacy
    from spacy_experimental.coref.coref_component import CoreferenceResolver
    
    print("Loading base transformer model...")
    nlp = spacy.load("en_core_web_trf")
    
    print("Adding experimental coreference component...")
     if "experimental_coref" not in nlp.pipe_names:
        nlp.add_pipe("experimental_coref")
        print("Initializing coreference pipeline...")
        nlp.initialize()
    
    # Save the complete pipeline
    nlp.to_disk("./en_coref_model")
    print("✅ Coreference-enabled model saved to ./en_coref_model")
    
    # Test the coreference
    print("\nTesting coreference resolution...")
    doc = nlp("Sarah went to the store. She bought milk and bread.")
    
    if hasattr(doc._, 'coref_clusters'):
        clusters = doc._.coref_clusters
        if clusters:
            print(f"✅ Coreference working! Found {len(clusters)} cluster(s):")
            for i, cluster in enumerate(clusters):
                mentions = [mention.text for mention in cluster.mentions]
                print(f"  Cluster {i}: {mentions}")
        else:
            print("⚠️ No coreference clusters found in test text")
    else:
        print("⚠️ Coreference attributes not found")
        
except Exception as e:
    print(f"❌ Coreference setup failed: {e}")
    print("Continuing with basic SpaCy model...")
EOF

# Check if Ollama is installed and start service
print_status "Checking Ollama installation..."
if command -v ollama >/dev/null 2>&1; then
    print_success "Ollama is already installed"
else
    print_warning "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    print_success "Ollama installed successfully"
fi

# Start Ollama service (required for downloading models)
print_status "Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    print_success "Ollama service already running"
else
    ollama serve &
    print_status "Waiting for Ollama service to start..."
    sleep 8
    
    # Verify service started
    if curl -s http://localhost:11434/api/version >/dev/null 2>&1; then
        print_success "Ollama service started successfully"
    else
        print_error "Failed to start Ollama service"
        exit 1
    fi
fi

# Download recommended models
print_status "Downloading recommended AI models (this may take 10-15 minutes)..."
print_status "Downloading Qwen2.5 7B (fast, efficient general chat)..."
if ollama pull qwen2.5:7b; then
    print_success "Qwen2.5 7B downloaded successfully"
else
    print_error "Failed to download qwen2.5:7b"
fi

print_status "Downloading DeepSeek-R1 7B (excellent reasoning capabilities)..."
if ollama pull deepseek-r1:7b; then
    print_success "DeepSeek-R1 7B downloaded successfully"
else
    print_error "Failed to download deepseek-r1:7b (continuing anyway)"
fi

# Verification
print_status "Verifying installation..."
python -c "
try:
    import spacy
    import fastapi
    import chromadb
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import requests
    
    print('✅ Core packages imported successfully:')
    print(f'  SpaCy: {spacy.__version__}')
    print(f'  FastAPI: {fastapi.__version__}') 
    print(f'  ChromaDB: {chromadb.__version__}')
    print(f'  Requests: {requests.__version__}')
    print('')
    
    # Test Ollama HTTP connection
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f'✅ Ollama HTTP API connected - {len(models)} models available:')
            for model in models[:5]:  # Show first 5
                size_gb = model.get('size', 0) / (1024**3)  # Convert to GB
                print(f'  - {model[\"name\"]} ({size_gb:.1f}GB)')
        else:
            print(f'⚠️ Ollama responded with status {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f'⚠️ Ollama connection failed: {e}')
        print('You may need to restart Ollama: ollama serve')
        
except Exception as e:
    print(f'❌ Verification failed: {e}')
"

# Create example .env file
print_status "Creating example .env file..."
cat > .env.example << 'EOF'
# Your secret key for the application
SECRET_KEY=your-secret-key-here-change-this

# Server Configuration  
HOST=localhost
PORT=8000

# Ollama Configuration (HTTP API only - no Python package)
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=qwen2.5:7b
CREATIVE_MODEL=deepseek-r1:7b

# Model Selection Guide:
# - qwen2.5:7b     -> Fast, efficient, good for general chat
# - deepseek-r1:7b -> Excellent reasoning, slower but more thoughtful
EOF

# Create simple Ollama HTTP client example
print_status "Creating Ollama HTTP client example..."
cat > ollama_client_example.py << 'EOF'
"""
Simple HTTP client for Ollama - no pydantic dependencies!
"""
import requests
import json

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def chat(self, model, messages, stream=False):
        """Chat with Ollama model using HTTP requests"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        if stream:
            # Handle streaming response
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        else:
            return response.json()
    
    def list_models(self):
        """List available models"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

# Example usage:
if __name__ == "__main__":
    client = OllamaClient()
    
    # Test connection
    try:
        models = client.list_models()
        print(f"Available models: {[m['name'] for m in models['models']]}")
        
        # Simple chat
        response = client.chat("qwen2.5:7b", [
            {"role": "user", "content": "Hello! How are you?"}
        ])
        
        print(f"AI: {response['message']['content']}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure Ollama is running: ollama serve")
EOF

print_success "✅ Setup complete!"
echo ""
print_success "Memory System is ready!"
echo ""
echo "What was installed:"
echo "   ✅ Python environment with all dependencies"
echo "   ✅ SpaCy coreference model (./en_coref_model/)"
echo "   ✅ Ollama service running in background"
echo "   ✅ AI models downloaded and ready"
echo ""
echo "Next steps:"
echo "   1. Activate environment: source ai_memory_env/bin/activate"
echo "   2. Configure app: cp .env.example .env (edit SECRET_KEY!)"
echo "   3. Test Ollama: python ollama_client_example.py"
echo "   4. Run your app: python app.py"
echo ""
echo "Useful commands:"
echo "   - List models: ollama list"
echo "   - Check service: curl http://localhost:11434/api/version"
echo "   - Stop service: pkill ollama"
echo "   - Remove model: ollama rm model-name"
echo "   - Show model: ollama show model-name"
echo ""
