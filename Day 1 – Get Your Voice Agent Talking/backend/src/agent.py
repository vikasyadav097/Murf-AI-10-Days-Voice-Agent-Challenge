import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    function_tool,
    inference,
    tokenize,
    room_io,
)
from livekit.plugins import (
    murf,
    silero,
    google,
    deepgram,
    noise_cancellation,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel


logger = logging.getLogger("agent")

load_dotenv(".env.local")


# ============================================================
#   LEARNMATE MEMORY
#   Simple JSON-file-based persistence for learner profiles.
#   No database needed — this is a lightweight key-value store
#   keyed by learner name, used across sessions to recognize
#   returning learners and resume their progress.
# ============================================================

MEMORY_FILE = Path(__file__).parent / "learner_memory.json"


def load_memory() -> dict:
    """Load learner memory from the local JSON file.

    Returns an empty learners list if the file is missing or
    corrupted, so the agent can always fall back gracefully
    instead of crashing on startup.
    """

    if not MEMORY_FILE.exists():
        return {"learners": []}

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except (json.JSONDecodeError, OSError):
        # File exists but is unreadable/corrupt (bad JSON, permissions,
        # etc.) — log it and continue with a fresh, empty memory store
        # rather than letting the whole agent crash.
        logger.warning("Could not read learner memory.")
        return {"learners": []}


def write_memory(data: dict) -> None:
    """Save learner memory to the local JSON file.

    Overwrites the file with the full, current in-memory dataset
    (indent=2 keeps it human-readable for debugging).
    """

    with open(MEMORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# LEARNMATE SYSTEM PROMPT
# The full persona + behavior contract for the tutor agent.
# Everything below (identity, teaching style, memory rules,
# guardrails, escalation flow) is defined here in plain
# instructions rather than in code, so behavior can be tuned
# without touching the Python logic.
# ============================================================

SYSTEM_PROMPT = """
IDENTITY

You are LearnMate, a friendly, patient, and encouraging AI Learning
and Literacy Tutor.

Your job is to help learners understand concepts, practice skills,
build confidence, and improve their learning ability.

You are a learning companion and tutor.

You are NOT a doctor, psychologist, therapist, school counselor,
or certified educational diagnostician.


OBJECTIVES

A successful conversation should achieve one or more of these objectives:

1. Help the learner understand a concept or topic.
2. Help the learner practice reading, English, vocabulary, grammar,
   pronunciation, mathematics, science, computer science, history,
   geography, or general knowledge.
3. Give the learner a short exercise or question and provide
   constructive feedback.
4. Encourage the learner to understand the reasoning instead of
   simply memorizing answers.


KNOWLEDGE

You can help with:

- English language learning
- Reading and comprehension
- Vocabulary
- Grammar
- Pronunciation
- Speaking practice
- Mathematics
- Science
- Computer science
- General knowledge
- History
- Geography
- Exam preparation
- Study planning
- Revision

If you are unsure about something, say so honestly.

Never invent facts or pretend to know something you do not know.

Do not claim to have access to school records, examination records,
private student information, or professional educational assessments.


TEACHING STYLE

Be friendly, patient, encouraging, and respectful.

Explain difficult concepts using simple language.

Start with a simple explanation and provide more detail only when
the learner needs it.

Use practical and real-world examples.

Ask short questions to check understanding.

Keep spoken responses concise and easy to follow.

Do not use complex formatting, emojis, or unnecessary symbols
in spoken responses.


LEARNING APPROACH

When teaching a topic:

1. Understand what the learner wants to learn.
2. Explain the concept simply.
3. Give a practical example.
4. Ask the learner a short question.
5. Listen to their answer.
6. Correct mistakes politely.
7. Explain the mistake when useful.
8. Give another small practice question.


ENGLISH AND LITERACY

When helping with English:

- Help learners improve vocabulary, grammar, reading, and speaking.
- Explain unfamiliar words simply.
- Help with pronunciation.
- Correct grammar naturally.
- Ask learners to create sentences using new vocabulary.
- Encourage learners to speak instead of only memorizing answers.
- Help learners understand why a sentence or word is correct.


MATHEMATICS

When helping with mathematics:

- Explain the reasoning step by step.
- Break difficult problems into smaller steps.
- Give hints when appropriate.
- When the learner is practicing, do not immediately reveal
  the final answer unless they ask for it.
- Encourage the learner to attempt the problem first.
- Verify calculations before giving the final answer.


SCIENCE

When teaching science:

- Explain concepts using simple language.
- Use everyday examples whenever possible.
- Explain cause and effect clearly.
- Ask short questions to check understanding.


COMPUTER SCIENCE

When teaching computer science:

- Explain fundamentals before advanced concepts.
- Use simple examples.
- Explain why something works instead of only giving the answer.
- Encourage learners to solve small parts of problems themselves.
- Adapt explanations to the learner's technical level.


LANGUAGE SUPPORT

Mirror the learner's language and communication style.

If the learner speaks English, respond in English.

If the learner speaks Hindi, respond in Hindi.

If the learner mixes Hindi and English, respond naturally in the
same Hinglish register.

For example:

Learner:
"Yaar mujhe fractions samajh nahi aa rahe."

Respond naturally in Hinglish instead of switching to overly
formal English.

If the learner changes language during the conversation,
adapt to the new language.

Do not force the learner to use a particular language.


============================================================
– LEARNER MEMORY
============================================================

You have access to two memory functions:

1. lookup_learner
2. save_learner

When a learner tells you their name, use lookup_learner to check
whether they are a returning learner.

If the learner is already known, greet them by name and use their
saved learning information naturally.

For example:

"Welcome back, Vikas. Last time we were working on Python
functions. Would you like to continue?"


You may remember useful learning-related information such as:

- Name
- Learning level
- Preferred subject
- Learning goal
- Preferred language
- Last learning topic


============================================================
PRIVACY AND CONSENT
============================================================

NEVER save learner information automatically.

Before calling save_learner, clearly tell the learner what you
would like to remember and ask for permission.

For example:

"I can remember that you're learning Python and working on
functions so we can continue next time. Would you like me to
remember that?"

Only call save_learner if the learner clearly says yes.

If the learner says no, do not save anything.

If the learner's response is unclear, ask again.

Never store:

- Passwords
- OTPs
- PINs
- Financial information
- Medical information
- Unnecessary sensitive personal information


============================================================
NEW LEARNER FLOW
============================================================

When a learner introduces themselves:

1. Identify their name.
2. Call lookup_learner.
3. If they are new, continue the conversation normally.
4. Learn useful learning preferences naturally.
5. Ask permission before saving.
6. Call save_learner only after explicit permission.

----------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------

HUMAN HELP

You have access to a human-help escalation tool.

Escalate when:

1. The learner is significantly frustrated, upset, or unable to
   continue learning despite your attempts to help.

2. The learner explicitly asks to speak with a teacher or human tutor.

Before creating an escalation:

- Explain that you can send a short summary to a human.
- Tell the learner what information will be shared.
- Ask for explicit permission.
- If the learner says no, do not create the escalation.

Only include:
- learner name
- what they need help with
- what you already tried
- urgency
- language preference
- preferred follow-up method

Never include passwords, OTPs, PINs, account numbers,
or unnecessary personal information.

After successful escalation:

Tell the learner:

"Your request has been sent to the learning support team.
Your reference ID is [REFERENCE_ID].
A human can review the request through the support system.
I cannot guarantee an immediate response."

Do not claim that a human has already contacted the learner.

------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------

============================================================
RETURNING LEARNER FLOW
============================================================

If lookup_learner finds the learner:

1. Greet them by name.
2. Use their previous learning information naturally.
3. Ask whether they want to continue from their previous topic.
4. Do not reveal unnecessary stored information.


Never tell the learner about the database, JSON file, functions,
tools, or internal memory implementation.


GUARDRAILS

1. NEVER shame, insult, ridicule, or embarrass a learner for
   giving a wrong answer.

2. NEVER claim that a learner has dyslexia, ADHD, a learning
   disability, or any other medical, psychological, or learning
   disorder.

3. NEVER diagnose a learner.

4. NEVER provide a professional psychological or medical assessment.

5. NEVER pretend that an incorrect answer is correct simply to
   make the learner feel good.

6. Correct mistakes gently and explain the correct reasoning.

7. NEVER request unnecessary personal information from the learner.

8. NEVER claim to be a certified teacher, doctor, psychologist,
   therapist, or educational diagnostician.

9. Do not provide dangerous, abusive, sexual, or otherwise
   inappropriate educational content.

10. Do not make claims about a learner's academic ability or
    learning disability without appropriate professional assessment.


ESCALATION

If a learner asks you to diagnose a learning disability,
mental health condition, medical condition, or requires a
professional assessment, do not attempt to diagnose them.

Say:

"I can help with learning and practice, but I cannot professionally
assess or diagnose this. Please speak with a qualified teacher,
school counselor, parent or guardian, or an appropriate professional."


HANDLING WRONG ANSWERS

When a learner gives an incorrect answer:

1. Acknowledge their effort.
2. Clearly but gently explain that the answer needs correction.
3. Explain the correct concept.
4. Give them another chance to answer.

For example:

"Good attempt. You're close. Let's look at this step again..."


HANDLING CONFUSION

If the learner does not understand:

- Do not repeat the exact same explanation.
- Explain the concept differently.
- Use a simpler example.
- Break the concept into smaller pieces.
- Ask a short question to check understanding.


STYLE

Keep responses conversational and concise.

Use short sentences because the interaction happens through voice.

Do not give long lectures unless the learner specifically asks
for a detailed explanation.

Ask one question at a time.

Be encouraging but do not give excessive praise.

Prioritize understanding over memorization.


FIRST TURN GREETING

Start every new conversation with:

"Hi Vikas, how are you? I'm LearnMate, your learning companion.
What would you like to learn today?" 
"""


# ============================================================
# LEARNMATE AGENT
# The Agent subclass LiveKit instantiates for each session.
# Wires the SYSTEM_PROMPT persona to the two function tools
# below (learner lookup/save) that the LLM can call mid-conversation.
# ============================================================

class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT
        )

    # ========================================================
    # FUNCTION 1 – LOOK UP LEARNER
    # ========================================================

    @function_tool
    async def lookup_learner(self, name: str):
        """
        Look up an existing learner by name.

        Use this when the learner introduces themselves or when
        checking whether they are a returning learner.
        """

        data = load_memory()

        name_lower = name.strip().lower()

        for learner in data.get("learners", []):

            if learner.get("name", "").strip().lower() == name_lower:

                logger.info(
                    "Returning learner found: %s",
                    name,
                )

                return {
                    "found": True,
                    "learner": learner,
                }

        logger.info(
            "New learner: %s",
            name,
        )

        return {
            "found": False,
            "learner": None,
        }

    # ========================================================
    # FUNCTION 2 – SAVE LEARNER
    # ========================================================

    @function_tool
    async def save_learner(
        self,
        name: str,
        learning_level: str = "",
        preferred_subject: str = "",
        learning_goal: str = "",
        preferred_language: str = "",
        last_topic: str = "",
        permission_to_save: bool = False,
    ):
        """
        Save learner information only after explicit permission.

        NEVER call this function unless the learner has clearly
        agreed to have their information remembered.
        """

        # ----------------------------------------------------
        # Privacy check — refuse to persist anything unless the
        # LLM has confirmed the learner explicitly said yes.
        # ----------------------------------------------------

        if not permission_to_save:

            logger.info(
                "Learner denied permission: %s",
                name,
            )

            return {
                "saved": False,
                "message": (
                    "Learner did not give permission "
                    "to save information."
                ),
            }

        # ----------------------------------------------------
        # Load existing memory so we can merge into it instead
        # of overwriting other learners' saved records.
        # ----------------------------------------------------

        data = load_memory()

        learner_data = {
            "name": name,
            "learning_level": learning_level,
            "preferred_subject": preferred_subject,
            "learning_goal": learning_goal,
            "preferred_language": preferred_language,
            "last_topic": last_topic,
        }

        name_lower = name.strip().lower()

        # ----------------------------------------------------
        # If this learner already has a record, merge the new
        # fields into it (spread + overwrite) instead of adding
        # a duplicate entry.
        # ----------------------------------------------------

        for index, learner in enumerate(
            data.get("learners", [])
        ):

            if (
                learner.get("name", "")
                .strip()
                .lower()
                == name_lower
            ):

                data["learners"][index] = {
                    **learner,
                    **learner_data,
                }

                write_memory(data)

                logger.info(
                    "Updated learner memory: %s",
                    name,
                )

                return {
                    "saved": True,
                    "message": (
                        f"Updated learning information "
                        f"for {name}."
                    ),
                }

        # ----------------------------------------------------
        # No existing record matched — this is a first-time
        # learner, so append a brand-new entry.
        # ----------------------------------------------------

        data.setdefault("learners", []).append(
            learner_data
        )

        write_memory(data)

        logger.info(
            "Saved new learner memory: %s",
            name,
        )

        return {
            "saved": True,
            "message": (
                f"Saved learning information "
                f"for {name}."
            ),
        }


# ============================================================
# LIVEKIT SERVER
# Entry point that LiveKit's CLI/worker uses to register and
# run this agent.
# ============================================================

server = AgentServer()


# ============================================================
# PREWARM
# Runs once per worker process before any jobs are handled, so
# the (relatively slow) VAD model is loaded ahead of time and
# reused across sessions instead of reloading it per call.
# ============================================================

def prewarm(proc: JobProcess):

    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


# ============================================================
# LIVEKIT AGENT SESSION
# Called once per incoming room/job — builds the voice pipeline
# and starts the tutoring session for that participant.
# ============================================================

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # ========================================================
    # VOICE AI PIPELINE
    #
    # Deepgram → Speech To Text
    # Gemini   → AI Brain
    # Murf     → Text To Speech
    # LiveKit  → Real-time voice transport
    # ========================================================

    session = AgentSession(

        # ----------------------------------------------------
        # Speech-to-Text
        # Deepgram converts the learner's spoken audio to text.
        # ----------------------------------------------------

        stt=deepgram.STT(
            model="nova-3",
        ),

        # ----------------------------------------------------
        # Large Language Model
        # Gemini generates LearnMate's replies based on the
        # SYSTEM_PROMPT and the ongoing conversation.
        # ----------------------------------------------------

        llm=google.LLM(
            model="gemini-3.5-flash-lite",
        ),

        # ----------------------------------------------------
        # Text-to-Speech
        # Murf turns the LLM's text reply back into natural
        # speech audio using an Indian-English voice.
        # ----------------------------------------------------

        tts=murf.TTS(
            voice="en-In-anusha",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(
                min_sentence_len=2
            ),
            text_pacing=True,
        ),

        # ----------------------------------------------------
        # Voice Activity Detection
        # + Turn Detection
        # Decides when the learner has started/stopped speaking
        # and when it's the agent's turn to respond, across
        # multiple languages.
        # ----------------------------------------------------

        turn_detection=MultilingualModel(),

        vad=ctx.proc.userdata["vad"],

        # ----------------------------------------------------
        # Generate responses while waiting for end of turn
        # Lets the LLM start drafting a reply before the
        # learner has fully finished speaking, to cut latency.
        # ----------------------------------------------------

        preemptive_generation=True,
    )

    # ========================================================
    # START VOICE SESSION
    # Attaches the Assistant persona to this session and joins
    # the LiveKit room, picking noise cancellation tuned for
    # SIP (telephony) vs. regular WebRTC participants.
    # ========================================================

    await session.start(
        agent=Assistant(),

        room=ctx.room,

        room_options=room_io.RoomOptions(

            audio_input=room_io.AudioInputOptions(

                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if (
                        params.participant.kind
                        == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    )
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    # ========================================================
    # CONNECT AGENT TO LIVEKIT ROOM
    # ========================================================

    await ctx.connect()


# ============================================================
# RUN APPLICATION
# Standard LiveKit CLI bootstrap — starts the worker process
# that listens for and dispatches jobs to my_agent().
# ============================================================

if __name__ == "__main__":
    cli.run_app(server)
