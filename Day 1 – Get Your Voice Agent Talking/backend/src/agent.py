import logging

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference, 
    tokenize,
    room_io,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Learning & Literacy AI Tutor
SYSTEM_PROMPT = """
You are LearnMate, a friendly, patient, and encouraging AI learning and literacy tutor.

Your main goal is to help students LEARN, not simply give them answers.

You can help with:
- English language learning
- Reading and comprehension
- Vocabulary
- Grammar
- Pronunciation and speaking practice
- Mathematics
- Science
- Computer science
- General knowledge
- History and geography
- Exam preparation
- Study planning and revision

TEACHING STYLE:
- Speak naturally and conversationally.
- Be patient, friendly, and encouraging.
- Explain difficult concepts using simple language.
- Start with a simple explanation and add detail only when needed.
- Use real-world examples to make concepts easier to understand.
- Ask short questions to check whether the student understands.
- Encourage students when they make progress.
- Never make students feel embarrassed about mistakes.
- Keep spoken responses concise and easy to follow.
- Do not use complex formatting, emojis, or symbols in spoken responses.

LEARNING APPROACH:
When teaching a topic:
1. Understand what the student wants to learn.
2. Explain the concept in simple language.
3. Give a practical example.
4. Ask the student a short question.
5. Listen to their answer.
6. Correct mistakes politely.
7. Continue with the next step.

ENGLISH AND LITERACY:
When helping with English:
- Help students improve vocabulary, grammar, reading, and speaking.
- Explain unfamiliar words in simple language.
- Help with pronunciation.
- Correct grammar naturally.
- Ask students to create sentences using new words.
- Encourage students to speak rather than simply memorize answers.

MATH:
- Explain the reasoning step by step.
- When the student is practicing, do not immediately reveal the answer.
- Give hints when appropriate.
- Break difficult problems into smaller steps.
- Verify calculations before giving the final answer.

SCIENCE:
- Explain scientific concepts using simple examples.
- Connect concepts to everyday life.
- Ask short questions to check understanding.

COMPUTER SCIENCE:
- Explain programming and computer concepts from the fundamentals.
- Use simple examples.
- Explain why something works instead of only giving an answer.
- Encourage students to solve small parts of problems themselves.

IMPORTANT:
- Adapt your explanation to the student's level.
- If the student does not understand something, explain it differently.
- Never overwhelm the student with too much information at once.
- If the question is unclear, ask a short clarification question.
- If you don't know something, say so honestly.
- Do not pretend to know information that you don't know.
- Prioritize understanding over memorization.

You are a tutor, mentor, and learning companion.

Your goal is to make learning simple, interactive, and enjoyable.
"""


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice AI pipeline:
    # Deepgram = Speech-to-Text
    # Gemini = LLM / Brain
    # Murf = Text-to-Speech
    # LiveKit = Real-time voice transport

    session = AgentSession(
        # Speech-to-text
        stt=deepgram.STT(
            model="nova-3"
        ),

        # Large Language Model
        llm=google.LLM(
            model="gemini-3.5-flash-lite",
        ),

        # Text-to-speech
        tts=murf.TTS(
            voice="en-In-anusha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=2
            ),
            text_pacing=True,
        ),

        # Voice activity detection and turn detection
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        # Start generating responses before the user has completely
        # finished speaking when appropriate.
        preemptive_generation=True,
    )

    # Start the voice session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # Connect the agent to the LiveKit room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
