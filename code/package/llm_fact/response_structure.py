"""
Store OpenAI response structures for data analysis tasks.
"""

import enum

from pydantic import BaseModel
from typing import Literal
from typing_extensions import TypedDict


class OpenAIFactCheckingResponse(BaseModel):
    label: Literal[
        "True",
        "Mostly true",
        "Half true",
        "Mostly false",
        "False",
        "Pants on fire",
        "Not enough information",
    ]
    justification: str


class Label(enum.Enum):
    TRUE = "True"
    MOSTLY_TRUE = "Mostly true"
    HALF_TRUE = "Half true"
    MOSTLY_FALSE = "Mostly false"
    FALSE = "False"
    PANTS_ON_FIRE = "Pants on fire"
    NOT_ENOUGH_INFORMATION = "Not enough information"


class GoogleFactCheckingResponse(TypedDict):
    label: Label
    justification: str
