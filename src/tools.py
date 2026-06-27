import json
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import function_tool, RunContext
import logging
from livekit.plugins.speechmatics import TurnDetectionMode, SpeakerIdentifier
from livekit.plugins import speechmatics as speechmatics_livekit
from typing import Any

load_dotenv(".env.local")


SPEAKERS_FILE = Path(__file__).parent / "speakers.json"

def save_speakers(raw_speakers: list[Any], label_id: str, name: str) -> None:
    """ Persist speakers from GET_SPEAKERS result to disk. """
    data = []
    for speaker in raw_speakers:
        if isinstance(speaker, dict):
            label, ids = speaker.get("label", ""), speaker.get("speaker_identifiers", [])
            
        else:
            label, ids = speaker.label, speaker.speaker_identifiers

        if label and ids:
            if label == label_id:
                label = name
            data.append({"label": label, "speaker_identifiers": ids})

    if data:
        with open(SPEAKERS_FILE, "w") as f:
            json.dump(data, f, indent=2)



def load_known_speakers() -> list[SpeakerIdentifier]:
    """ Load known speakers from disk. """
    if not SPEAKERS_FILE.exists():
        return []

    with open(SPEAKERS_FILE) as f:
        data = json.load(f)

    return [
        SpeakerIdentifier(label=entry["label"], speaker_identifiers=entry["speaker_identifiers"])
        for entry in data
        if entry.get("label") and entry.get("speaker_identifiers")
    ]
            
        
known_speakers = load_known_speakers()


stt = speechmatics_livekit.STT(
    turn_detection_mode=TurnDetectionMode.SMART_TURN,
    enable_diarization=True,
    speaker_active_format="<{speaker_id}>{text}</{speaker_id}>",
    speaker_passive_format="<PASSIVE><{speaker_id}>{text}</{speaker_id}></PASSIVE>",
    known_speakers=known_speakers,

    )


@function_tool()
async def assign_name_2_speaker_ids(ctx: RunContext, label_id: str, name: str) -> str:
    """
    Assign a name to a speaker label and persist it to disk.
    
    args:
    - label_id: The speaker label to assign a name to.
    - name: The name to assign to the speaker label.
    """
    try:
        result = await stt.get_speaker_ids()
        logging.info(f"Retrieved speaker IDs: {result}")
        save_speakers(result, label_id, name)
        ctx.session.generate_reply(instructions="Greet the user with their name if known and provide helpful responses.")
        return f"Successfully assigned name '{name}' to speaker '{label_id}'."
    except Exception as e:
        logging.error(f"Error retrieving speaker IDs: {e}")
        return "Error retrieving speaker IDs."
  