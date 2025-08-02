import json
from pathlib import Path
from textwrap import dedent
from typing import Final, Mapping

_PROMPT: Final[str] = dedent(
    """
    You are a professional translator, specializing in English to Japanese translations for television dramas.

    Your task is to translate the provided game of thrones .srt file into Japanese.

    GENERAL GUIDELINES:
    1. Do not edit the timings or overall format of the file.
    2. "No" should be translated depending on what the character is responding to:
        いいえ - Answering yes/no questions ("Are you coming?" "No.")
        だめ - Rejecting actions or saying something is forbidden ("Can I eat this?" "No!")
        やめて - Stopping ongoing actions ("Stop tickling me!" "No, don't touch that!")
        違う - Correcting information ("You're from Tokyo, right?" "No, I'm from Osaka.")

    OUTPUT INSTRUCTIONS:
    Skip any preamble or explanations. Return the entire translated subtitle file enclosed in <result> tags.

    CHARACTER NAMES:
    ${character_names}
    """
)


def get_prompt() -> str:
    characters_file = Path("characters.json").read_text()
    characters_dict = json.loads(characters_file)

    character_names = []
    for entry in characters_dict["characters"]:
        character_names.append(entry["characterName"])

    return _PROMPT.replace("${character_names}", ", ".join(character_names))
