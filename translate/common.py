import enum
import functools
import textwrap
from dataclasses import dataclass
from typing import Final, Sequence


@enum.unique
class ModelProvider(enum.Enum):
    anthropic = "anthropic"
    google = "google"
    openai = "openai"

    def temperature(self) -> float:
        match self:
            case ModelProvider.anthropic | ModelProvider.google | ModelProvider.openai:
                return 0.1
            case _:
                raise ValueError(f"Unexpected model provider: {self}")


@enum.unique
class Model(enum.Enum):
    haiku = f"{ModelProvider.anthropic.name}/claude-3-5-haiku-latest"
    opus = f"{ModelProvider.anthropic.name}/claude-3-opus-latest"
    sonnet_35 = f"{ModelProvider.anthropic.name}/claude-3-5-sonnet-latest"
    sonnet_37 = f"{ModelProvider.anthropic.name}/claude-3-7-sonnet-latest"
    sonnet_4 = f"{ModelProvider.anthropic.name}/claude-sonnet-4-20250514"

    def get_provider(self) -> ModelProvider:
        return ModelProvider(self.value.split("/")[0])


@dataclass(frozen=True)
class Location:
    line: int
    column: int

    @property
    def llm_markdown(self) -> str:
        return f"Line {self.line}, Col {self.column}"


@enum.unique
class Severity(enum.Enum):
    error = "error"
    warning = "warning"

    @classmethod
    def from_str(cls, value: str) -> "Severity":
        match value.strip().lower():
            case "error" | "e" | "err":
                return cls.error
            case "warning" | "w" | "warn":
                return cls.warning
            case _:
                raise ValueError(f"Unexpected severity: {value}")


@dataclass(frozen=True)
class ValidationError:
    message: str
    severity: Severity
    location: Location
    source: Final[str]

    @property
    def llm_markdown(self) -> str:
        return f"{self.source} {self.severity.value} - {self.location.llm_markdown} - {self.message}"


class ValidationFailure(ValueError):
    message: Final[str]
    errors: Sequence[ValidationError]

    def __init__(self, message: str, errors: Sequence[ValidationError]) -> None:
        assert errors
        self.message = message
        self.errors = errors
        ValueError.__init__(self, self.llm_markdown)

    @functools.cached_property
    def llm_markdown(self) -> str:
        errors_markdown = "\n".join(error.llm_markdown for error in self.errors)
        return textwrap.dedent(
            f"""
            {self.message}:
            ===============
            {errors_markdown}
            """
        )

    @functools.cached_property
    def failure_count(self) -> int:
        return len(self.errors)
