import abc
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from pprint import pp, pprint
from time import sleep, time
from typing import Final, Literal, Sequence, TypedDict

import click
import dotenv
import litellm
from common import (Location, Model, Severity, ValidationError,
                    ValidationFailure)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from system_prompt import get_prompt

log = logging.getLogger(__name__)

FOLDER_IDS = {
    5: "1p-rBulZzDqGLM3e8prtLR_L274lQMbCc",
    6: "1J0L2v3Sy13tpnvRzPK1uIkXmtNwR83DG",
}

THRESHOLD_SECONDS = 10


class Color:
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def system(msg: str) -> Message:
    return Message(role="system", content=msg)


def user(msg: str) -> Message:
    return Message(role="user", content=msg)


def assistant(msg: str) -> Message:
    return Message(role="assistant", content=msg)


def generate_credentials():
    dotenv.load_dotenv()

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    # Get credentials from environment variables
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Error: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env file"
        )

    client_config = {
        "web": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost:8080/callback"],
        }
    }

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = Flow.from_client_config(client_config, SCOPES)
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

            # Generate authorization URL
            auth_url, _ = flow.authorization_url(prompt="consent")
            print(f"Please go to this URL and authorize the application: {auth_url}")

            # Get authorization code from user
            auth_code = input("Enter the authorization code: ")
            flow.fetch_token(code=auth_code)
            creds = flow.credentials

        # Save credentials for next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    return creds


def upload_file_to_drive(srt_file_path: Path, folder_id: str) -> None:
    creds = generate_credentials()
    service = build("drive", "v3", credentials=creds)

    # Get the filename from the path
    filename = os.path.basename(srt_file_path)

    # File metadata
    file_metadata = {
        "name": filename,
        "parents": [folder_id],  # Upload to specific folder
    }

    # Media upload
    media = MediaFileUpload(
        srt_file_path,
        mimetype="text/plain",  # SRT files are plain text
        resumable=True,
    )

    # Upload the file
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    print(f"Successfully uploaded {filename} to Google Drive")
    print(f"File ID: {file.get('id')}")


class LLMFailure(RuntimeError):
    messages: Final[Sequence[Message]]

    def __init__(self, error: str, messages: Sequence[Message]) -> None:
        RuntimeError.__init__(self, error)
        self.messages = messages


class LLMTask(abc.ABC):
    model: Final[Model]
    user_prompt: Final[str]
    temperature: Final[float]

    def __init__(
        self,
        model: Model,
        user_prompt: str,
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.user_prompt = user_prompt
        self.temperature = temperature or model.get_provider().temperature()
        self.failure_count = 0

    def run(self, max_iterations: int = 5) -> str:
        try:
            messages: list[Message] = [
                system(self.system_prompt),
                user(self.user_prompt),
            ]
            for iteration in range(max_iterations):
                log.info(f"Running {iteration + 1} out of {max_iterations} iterations")
                response = litellm.completion(
                    model=self.model.value,
                    messages=messages,
                    temperature=self.temperature,
                    max_completion_tokens=50000,
                    timeout=1200,
                )

                finish_reason = response["choices"][0]["finish_reason"]

                response_content = response["choices"][0]["message"]["content"]
                messages.append(assistant(response_content))
                print(f"{finish_reason=}")

                if match := re.search(r"(?ms)<result>(.*?)</result>", response_content):
                    result = match.group(1)
                else:
                    print("XML TAGS NOT FOUND")
                    messages.append(user("No <result> tag found in the response."))
                    continue

                try:
                    self.validate(result)
                    log.info(f"Task succeeded after {iteration + 1} iterations.")
                    return result
                except ValidationFailure as e:
                    log.error(f"{Color.RED}\n{e.llm_markdown}{Color.RESET}")
                    self.failure_count += e.failure_count
                    messages.append(user(e.llm_markdown))

            raise LLMFailure(
                f"Task failed with {self.failure_count} total validation failures after {max_iterations} attempts.",
                messages=messages,
            )
        except Exception:
            import traceback

            traceback.print_exc()
            raise

    @abc.abstractmethod
    def validate(self, result: str) -> None:
        pass

    @property
    @abc.abstractmethod
    def system_prompt(self) -> str:
        pass


class TranslateSubtitles(LLMTask):
    fname: Final[str]

    def __init__(
        self,
        fname: str,
        subtitles: str,
        model: Model = Model.sonnet_37,
    ) -> None:
        self.fname = fname
        self.subtitles = subtitles

        LLMTask.__init__(self, model=model, user_prompt=subtitles)

    @property
    def system_prompt(self) -> str:
        return get_prompt()

    def validate(self, result: str) -> None:
        final_block = self.user_prompt.rstrip().lstrip().split("\n\n")[-1]
        print(final_block.split("\n"))
        final_line_number = int(final_block.split("\n")[0])

        result_final_block = result.rstrip().split("\n\n")[-1]
        result_final_line_number = int(result_final_block.split("\n")[0])
        print(f"{final_line_number=}")
        print(f"{result_final_line_number=}")
        if result_final_line_number != final_line_number:
            raise ValidationFailure(
                message="Length validation failed",
                errors=[
                    ValidationError(
                        message=f"Input has {final_line_number} blocks, output has {result_final_line_number} blocks.",
                        severity=Severity.error,
                        location=Location(line=final_line_number, column=0),
                        source=self.fname,
                    )
                ],
            )


def seconds_since_midnight(time_obj: datetime):
    return (
        time_obj.hour * 3600
        + time_obj.minute * 60
        + time_obj.second
        + time_obj.microsecond / 1_000_000
    )


def preprocess(file_contents: str) -> list[str]:
    blocks = []
    prev_timestamp = None
    current_block = []
    for block in file_contents.rstrip().split("\n\n"):
        start_time = block.split("-->")[0].rstrip().split("\n")[-1]
        timestamp_formatted = start_time.replace(",", ".")
        current_timestamp = datetime.strptime(timestamp_formatted, "%H:%M:%S.%f").time()
        if not prev_timestamp:
            prev_timestamp = current_timestamp
            current_block.append(block)
            continue

        time_diff = seconds_since_midnight(current_timestamp) - seconds_since_midnight(
            prev_timestamp
        )

        if time_diff >= THRESHOLD_SECONDS and current_block:
            blocks.append("\n\n".join(current_block))
            current_block = [block]
        else:
            current_block.append(block)

        prev_timestamp = current_timestamp

    if current_block:
        blocks.append("\n\n".join(current_block))

    return blocks


@click.group()
def cli():
    """Command line tool for LLM libraries."""
    dotenv.load_dotenv()
    logging.basicConfig(level=logging.INFO)


@cli.command("upload")
@click.argument(
    "file_path",
    type=click.Path(exists=True, path_type=Path, file_okay=True, resolve_path=True),
)
@click.argument("season", type=str)
def upload(file_path: Path, season: str):
    upload_file_to_drive(file_path, season)


@cli.command("run")
@click.argument(
    "file_path",
    type=click.Path(exists=True, path_type=Path, file_okay=True, resolve_path=True),
)
@click.argument("season", type=str)
@click.argument("episode", type=str)
def run(file_path: Path, season: str, episode: str) -> None:
    pprint(f"Available claude models: {litellm.anthropic_models}")

    contents = file_path.read_text(encoding="utf-8-sig")

    blocks = preprocess(contents)

    output_blocks = []

    for idx, block in enumerate(blocks):
        print(f"Running {idx + 1} out of {len(blocks)}")
        task = TranslateSubtitles(file_path.name, block, Model.sonnet_37)
        result = task.run()
        sleep(5)
        output_blocks.append(result)

    output = "\n\n".join(output_blocks)

    output_file = Path(
        f"Subtitles (JP)/S{season}/Game of Thrones S{season}E{episode}.jp.srt"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as file:
        file.write(output)

    folder_id = FOLDER_IDS.get(int(season))
    # upload_file_to_drive(output_file, folder_id)


if __name__ == "__main__":
    cli()
