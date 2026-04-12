# NIHSA AI Wrapper Service

Production-grade AI assistant service for the NIHSA Flood Intelligence Platform.

## Features

- **Speech-to-Text**: Local Whisper model (base/small) with persistent caching
- **Text Chat**: DeepSeek with function calling for app navigation
- **Text-to-Speech**: Google Cloud TTS with 30-day disk caching
- **Multi-language**: Supports English, Hausa, Yoruba, Igbo, French
- **Tutorials**: Built-in help content in all languages

## Deployment (Render)

1. Push this repository to GitHub
2. Create a new Web Service on Render
3. Connect your repository
4. Render will automatically use `render.yaml`
5. Add secrets:
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key
   - `GOOGLE_APPLICATION_CREDENTIALS_JSON`: Full GCP service account JSON

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
export NIHSA_API_URL="https://nihsa-backend-20hh.onrender.com/api"

# Run
uvicorn main:app --reload --port 8000