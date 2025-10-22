# Signal enabled confidential AI.

A Signal messenger bot that gives you AI agent in your chat.

## Features

- Chat with AI assistant via Signal messages
- **Local speech processing** with Sherpa-ONNX (STT/TTS)
- Voice message transcription and synthesis
- Graceful fallback to external APIs (Whisper/Kokoro)
- Role-based permission system with file watching
- Maintains conversation context
- Simple command interface
- Docker deployment ready
- Modular, organized codebase with src/ structure

## Quick Start

**One command setup and deployment:**

```bash
# Clone and setup
git clone <repository-url>
cd signal-bot
chmod +x setup.sh
./setup.sh

# Configure your Signal phone number
nano .env  # Edit SIGNAL_PHONE_NUMBER=+1234567890

# Deploy
docker compose up -d
```

That's it! The bot will start with local Sherpa-ONNX processing.

## First Time Setup

### 1. Initial Configuration

```bash
# Run the setup script (creates .env, directories, and downloads models)
./setup.sh

# Edit your phone number and settings
nano .env
```

**Required settings in `.env`:**
```env
SIGNAL_PHONE_NUMBER=+1234567890
```

**Optional settings:**
```env
# Use local processing (recommended)
USE_SHERPA_STT=true
USE_SHERPA_TTS=true

# Voice activation phrase
VOICE_ACTIVATION_PHRASE=hey jarvis
```

### 2. Link Signal Account

```bash
# Link your Signal account (first time only)
docker compose run --rm signal-cli-rest-api signal-cli link -n "Signal Bot"
```

Scan the QR code with your Signal app when prompted.

### 3. Start the Bot

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f signal-bot
```

## Usage

Send messages to the bot:

- `!chat <message>` - Chat with AI assistant
- `!clear` - Clear conversation history
- `!models` - List available AI models
- `!help` - Show available commands
- Any message without command prefix is treated as chat
- **Voice messages** - Send or forward voice memos for automatic transcription

## Available Models

The bot can use any model available through PrivateMode.ai API. Use `!models` command to see available models. Example models:
- `ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4`

## Development

Run locally:
```bash
pip install -r requirements.txt
python signal_bot.py
```

## Architecture

- Uses `signalbot` library for Signal integration
- Connects to PrivateMode.ai Chat Completions API endpoint
- **Local STT/TTS**: Sherpa-ONNX for offline speech processing (recommended)
- **Fallback STT/TTS**: External Whisper ASR and Kokoro TTS APIs
- Maintains conversation context per sender (last 10 messages)
- Supports docker deployment with signal-cli-rest-api
- No authentication required (follows PrivateMode.ai approach)

## Project Structure

The project is organized with a clean, modular structure:

```
signal-bot/
├── src/                          # Main application source code
│   ├── bot.py                   # Main bot application and entry point
│   ├── clients/                  # Client implementations
│   │   ├── stt.py              # Speech-to-Text client (Sherpa-ONNX + Whisper)
│   │   ├── tts.py              # Text-to-Speech client (Sherpa-ONNX + Kokoro)
│   │   └── api.py              # API clients (PrivateMode, Whisper, Kokoro)
│   ├── permissions/             # Permission system
│   │   └── manager.py          # Role-based access control with file watching
│   └── commands/                # Bot command implementations
│       ├── __init__.py
│       ├── chat.py             # Chat and conversation commands
│       ├── admin.py            # Administrative commands
│       └── voice.py            # Voice message handling
├── scripts/                     # Utility scripts
│   └── download_models.py     # Sherpa-ONNX model downloader
├── docs/                        # Documentation
│   ├── MIGRATION_GUIDE.md      # Migration instructions
│   └── DOCKER_OPTIMIZATION.md  # Docker optimization guide
├── docker/                      # Docker configuration files
├── config/                      # Configuration files
├── signal_bot.py               # Main entry point (redirects to src/)
├── setup.sh                    # Automated setup script
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker deployment configuration
├── Dockerfile                  # Multi-stage Docker build
├── .env.example                # Configuration template
├── permissions.json            # User roles and permissions
└── README.md                   # This file
```

**Key Design Principles:**
- **Modularity**: Each component has a single responsibility
- **Separation of Concerns**: Clear boundaries between UI, business logic, and infrastructure
- **Graceful Degradation**: Local processing with external API fallbacks
- **Security**: Non-root execution, role-based permissions
- **Maintainability**: Clean imports, organized structure, comprehensive documentation

## Speech Processing (STT/TTS)

The bot supports two approaches for speech processing:

### 1. Sherpa-ONNX Local Processing (Recommended)

**Benefits**: No external dependencies, faster response, better reliability, multi-language support

**Supported Models**:
- **STT**: SenseVoice (Chinese/English/Japanese/Korean/Cantonese), Zipformer Bilingual, Paraformer Trilingual, FireRedASR
- **TTS**: VITS Chinese, Melo Chinese/English, Kokoro Multi-language (53 speakers)

**Setup**:
1. Download models to `./models` directory:
```bash
# Example: Download SenseVoice ASR model
mkdir -p models
cd models
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

# Download VITS TTS model
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-hf-theresa.tar.bz2
tar xf vits-zh-hf-theresa.tar.bz2

# Download VAD model (required for offline ASR models)
mkdir -p silero_vad
cd silero_vad
curl -SL -o silero_vad.onnx https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

2. Configure environment variables (see `.env.example`)

### 2. External API Fallback (Legacy)

**Whisper ASR Integration**: The bot supports voice message transcription using external Whisper ASR instances.

**Network Setup**: The docker-compose configuration uses `network_mode: "host"` to ensure the Signal bot can access Whisper instances running on:

- **Host machine**: Use `http://localhost:9000` or `http://127.0.0.1:9000`
- **Docker containers (bridge network)**: Use `http://172.17.0.1:9000` (Docker's default bridge gateway)
- **Docker containers (host network)**: Use `http://localhost:9000`
- **External servers**: Use the server's URL (e.g., `http://whisper.example.com:9000`)

### Multiple Instances

Configure multiple instances for automatic failover:
```bash
# For Sherpa-ONNX: Not applicable (local processing)
# For external APIs:
WHISPER_ASR_URLS=http://localhost:9000,http://localhost:9001,http://backup-server:9000
KOKORO_TTS_URLS=http://localhost:8880,http://backup-tts:8880
```

### Voice Message Processing

1. User sends or forwards a voice message to the Signal chat
2. Bot detects the voice attachment
3. **Sherpa-ONNX**: Local processing with selected ASR model
4. **Fallback**: Audio is sent to external Whisper/Kokoro instance
5. Transcribed text is returned with a 📝 prefix
6. Transcription is stored in conversation history for context
7. If transcription starts with voice activation phrase, AI response is generated
8. **TTS**: AI response can be spoken using local Sherpa-ONNX or external Kokoro TTS
