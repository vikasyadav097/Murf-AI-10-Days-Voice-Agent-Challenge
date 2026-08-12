# 🎙️ Murf AI Voice Agent Challenge

> Build a production-ready real-time AI Voice Agent powered by Murf Falcon TTS, LiveKit, Deepgram STT, and LLM intelligence.

This project creates a natural conversational AI voice assistant where users can speak through their microphone and receive intelligent AI-generated voice responses in real time.

The system combines speech recognition, large language models, and ultra-low latency text-to-speech technology to deliver a human-like voice interaction experience.

---

# 🚀 Features

## 🎤 Real-Time Voice Conversation
- Real-time microphone input
- AI understands user speech
- Instant voice responses
- Low latency conversational experience

## 🔊 Murf Falcon TTS
- High-quality AI voice generation
- Natural human-like speech
- Multiple voice support
- Fast text-to-speech processing

## 🧠 AI Intelligence
- Context-aware conversations
- Custom system prompts
- Flexible AI agent personality
- Supports multiple use cases

## 📝 Speech Recognition
- Deepgram Nova Speech-to-Text
- Accurate voice transcription
- Real-time processing

## ⚡ LiveKit Real-Time Communication
- WebRTC based audio streaming
- Scalable voice infrastructure
- Low-latency communication

---

# 🏗️ Architecture

```
User Speech
     |
     ↓
Microphone Input
     |
     ↓
Deepgram STT
(Speech To Text)
     |
     ↓
LLM Brain
(Gemini / OpenAI)
     |
     ↓
Murf Falcon TTS
(Text To Speech)
     |
     ↓
AI Voice Response
```

---

# 🛠️ Tech Stack

## Frontend
- Next.js
- TypeScript
- LiveKit Client SDK

## Backend
- Python
- LiveKit Agents Framework
- Async Programming

## AI Services
- Murf Falcon TTS
- Deepgram STT
- Google Gemini / OpenAI

## Development Tools
- Git
- GitHub
- uv Package Manager
- pnpm
- Docker

---

# 📂 Project Structure

```
Murf-AI-10-Days-Voice-Agent-Challenge/

│
├── backend/
│   ├── src/
│   │   └── agent.py
│   ├── .env.example
│   └── pyproject.toml
│
├── frontend/
│   ├── app/
│   ├── components/
│   └── package.json
│
├── start_app.ps1
├── start_app.sh
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

Make sure you have:

- Python 3.10+
- Node.js 18+
- uv package manager
- pnpm
- LiveKit Account

---

# Clone Repository

```bash
git clone https://github.com/vikasyadav097/Murf-AI-10-Days-Voice-Agent-Challenge.git

cd Murf-AI-10-Days-Voice-Agent-Challenge
```

---

# 🔐 Environment Setup

Create `.env` files.

## Backend Environment

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

MURF_API_KEY=
DEEPGRAM_API_KEY=

GOOGLE_API_KEY=
```

## Frontend Environment

```env
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
```

---

# 📦 Backend Setup

```bash
cd backend

uv sync

uv run python src/agent.py download-files
```

---

# 📦 Frontend Setup

```bash
cd frontend

pnpm install
```

---

# ▶️ Run Application

## Windows

From root directory:

```powershell
.\start_app.ps1
```

---

## Manual Run

### Terminal 1

Start LiveKit:

```bash
livekit-server --dev
```

### Terminal 2

Start AI Agent:

```bash
cd backend

uv run python src/agent.py dev
```

### Terminal 3

Start Frontend:

```bash
cd frontend

pnpm dev
```

Open:

```
http://localhost:3000
```

Click **Start Talking**, allow microphone access, and start speaking with your AI voice agent.

---

# 🎯 Use Cases

## 🤖 AI Customer Support Agent

- Handle customer queries
- Product assistance
- Technical support automation

## 📚 AI Language Tutor

- Speaking practice
- Pronunciation improvement
- Interactive learning

## 🏥 AI Receptionist

- Appointment scheduling
- Information assistance
- Voice-based services

## 💼 Business Voice Assistant

- Lead qualification
- Meeting assistant
- Workflow automation

---

# 🎙️ Murf Voice Options

| Voice | Language |
|---|---|
| Anisha | Indian English |
| Pooja | Indian English |
| Samar | Indian English |
| Amara | US English |
| Gordon | US English |
| Hazel | UK English |

---

# 🚀 Deployment

## Backend Deployment

Recommended platforms:

- Railway
- Render
- AWS

Required Variables:

```
MURF_API_KEY
DEEPGRAM_API_KEY
GOOGLE_API_KEY
LIVEKIT_URL
LIVEKIT_API_KEY
LIVEKIT_API_SECRET
```

---

## Frontend Deployment

Recommended platform:

- Vercel

Required Variables:

```
LIVEKIT_URL
LIVEKIT_API_KEY
LIVEKIT_API_SECRET
```

---

# 🔮 Future Improvements

- Voice cloning support
- Multi-language conversations
- AI memory system
- Emotion-aware responses
- Mobile application
- Custom AI personalities
- Advanced analytics dashboard

---

# 👨‍💻 Author

## Vikas Yadav

GitHub:
https://github.com/vikasyadav097

---

# 📜 License

MIT License

---

⭐ Built with Murf Falcon, LiveKit, Deepgram and Generative AI.
