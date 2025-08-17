# Signal enabled confidential AI.

A Signal messenger bot that gives you AI agent in your chat.

## Features

- Chat with AI assistant via Signal messages
- Voice message transcription with Whisper ASR
- Voice commands with activation phrase 

## Requirements
- working [signal-cli-rest-api](https://github.com/bbernhard/signal-cli-rest-api) instance
- working [whisper-asr](https://github.com/ahmetoner/whisper-asr-webservice) instance (if you want voice support)
- account at [privatemode.ai](https://privatemode.ai)

## Setup

1. Copy `.env.example` to `.env` and configure:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your configuration:
   - `SIGNAL_PHONE_NUMBER`: Your Signal phone number (e.g., +1234567890)
   - `PRIVATEMODE_BASE_URL`: URL of PrivateMode.ai API (default: http://localhost:8080)
   - `PRIVATEMODE_MODEL`: Optional - specific model to use (if not set, uses first available)
   - `WHISPER_ASR_URLS`: Comma-separated list of Whisper ASR instances (e.g., http://localhost:9000,http://localhost:9001)
   - `WHISPER_OUTPUT_FORMAT`: Output format for transcription (default: text)
   - `WHISPER_VAD_FILTER`: Enable voice activity detection (default: true)
   - `WHISPER_LANGUAGE`: Optional - language for transcription (auto-detect if not set)

3. Link Signal account (first time only):
   ```bash
   docker compose run --rm signal-cli-rest-api signal-cli link -n "Signal Bot"
   ```
   Follow the instructions to scan QR code with Signal app.

4. Start the bot:
   ```bash
   docker compose up -d
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
- Connects to PrivateMode.ai API endpoint
- Integrates with Whisper ASR for voice transcription
- Maintains conversation context per sender (last 10 messages)
- Supports docker deployment with signal-cli-rest-api
- No authentication required (follows PrivateMode.ai approach)

## Whisper ASR Integration

The bot supports voice message transcription using Whisper ASR. The docker-compose configuration uses `network_mode: "host"` to ensure the Signal bot can access Whisper instances running on:

- **Host machine**: Use `http://localhost:9000` or `http://127.0.0.1:9000`
- **Docker containers (bridge network)**: Use `http://172.17.0.1:9000` (Docker's default bridge gateway)
- **Docker containers (host network)**: Use `http://localhost:9000`
- **External servers**: Use the server's URL (e.g., `http://whisper.example.com:9000`)

### Multiple Whisper Instances

Configure multiple Whisper instances for automatic failover:
```bash
WHISPER_ASR_URLS=http://localhost:9000,http://localhost:9001,http://backup-server:9000
```

If the first instance fails or is unavailable, the bot automatically tries the next one in the list.

### Voice Message Processing

1. User sends or forwards a voice message to the Signal chat
2. Bot detects the voice attachment
3. Audio is sent to the first available Whisper instance
4. Transcribed text is returned with a 📝 prefix
5. Transcription is stored in conversation history for context
