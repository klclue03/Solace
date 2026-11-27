import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from pydantic import BaseModel, Field

API_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [API_DIR, os.path.join(os.path.dirname(API_DIR), "model")]
from journal import PROMPTS, JournalError  # noqa: E402
from service import DEFAULT_DB, PRIVACY_NOTICE, WellbeingService  # noqa: E402

Confidence = Literal["low", "medium", "high"]
Source = Literal["survey", "journal"]
JournalAnswer = Annotated[str, Field(max_length=10_000)]
USER_ID_RULES = {"min_length": 1, "max_length": 100, "pattern": r"^[A-Za-z0-9_.\-]+$"}


class Question(BaseModel):
    id: str
    label: str
    question: str
    min: float
    max: float


class QuestionsOut(BaseModel):
    model_version: str
    questions: list[Question]
    journal_prompts: list[str]
    privacy_notice: str


class Support(BaseModel):
    level: Literal["crisis", "low_score", "info"]
    message: str


class ExtractedItem(BaseModel):
    value: Optional[float]
    confidence: Confidence
    evidence: str


class JournalIn(BaseModel):
    entries: list[JournalAnswer] = Field(
        min_length=1, max_length=len(PROMPTS),
        description="Answers to journal_prompts, in the same order.")


class JournalOut(BaseModel):
    items: dict[str, ExtractedItem]
    found: list[str] = Field(
        description="Items read with medium or high confidence: show for review.")
    unclear: list[str] = Field(
        description="Items to ask as follow-up questions. A low-confidence "
                    "value may be shown as a hint, never as a default.")
    safety_concern: bool
    support: Optional[Support]


class JournalExtraction(BaseModel):
    items: dict[str, ExtractedItem]
    safety_concern: bool


class ResponseIn(BaseModel):
    user_id: str = Field(**USER_ID_RULES)
    source: Source = "survey"
    answers: dict[str, float]
    journal: Optional[JournalExtraction] = Field(
        default=None,
        description="For source='journal': the extraction returned by /journal/extract.")
    save: bool = True


class Previous(BaseModel):
    id: int
    created_at: str
    source: Source
    model_version: str
    overall: float
    subscores: dict[str, float]
    rest: float


class Changes(BaseModel):
    overall: float
    subscores: dict[str, float]
    rest: float


class ResultOut(BaseModel):
    id: Optional[int]
    user_id: str
    created_at: Optional[str]
    source: Source
    model_version: str
    overall: float = Field(description="0-100: higher than this % of people in the training data.")
    subscores: dict[str, float] = Field(description="Approximate percentiles per dimension.")
    rest: float
    answers: dict[str, float]
    previous: Optional[Previous]
    changes: Optional[Changes]
    support: Optional[Support]


class HistoryOut(BaseModel):
    user_id: str
    responses: list[Previous]


@asynccontextmanager
async def lifespan(app):
    service = WellbeingService(db_path=os.environ.get("WELLBEING_DB", DEFAULT_DB))
    service.warm_up()
    app.state.service = service
    yield


app = FastAPI(
    title="Wellbeing API",
    version="0.1.0",
    description="Scores wellbeing from an 18-question survey or a short journal. "
                "Test use only: there is no authentication yet, so do not use with real personal data.",
    lifespan=lifespan,
)


def get_service(request: Request) -> WellbeingService:
    return request.app.state.service


@app.get("/health")
def health(service: WellbeingService = Depends(get_service)):
    return {"status": "ok", "model_version": service.model_version}


@app.get("/questions", response_model=QuestionsOut)
def questions(service: WellbeingService = Depends(get_service)):
    return {"model_version": service.model_version, "questions": service.questions(),
            "journal_prompts": PROMPTS, "privacy_notice": PRIVACY_NOTICE}


@app.post("/journal/extract", response_model=JournalOut)
def journal_extract(body: JournalIn, service: WellbeingService = Depends(get_service)):
    try:
        return service.extract_journal(body.entries)
    except JournalError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/responses", response_model=ResultOut, status_code=201)
def submit_response(body: ResponseIn, service: WellbeingService = Depends(get_service)):
    if body.source == "journal" and body.journal is None:
        raise HTTPException(status_code=422,
                            detail="source 'journal' requires the 'journal' extraction.")
    extraction = body.journal.model_dump() if body.journal else None
    if extraction and set(extraction["items"]) != set(service.items):
        raise HTTPException(status_code=422,
                            detail="'journal' must contain every survey item.")
    try:
        return service.submit(body.user_id, body.answers, body.source, extraction, body.save)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/users/{user_id}/responses", response_model=HistoryOut)
def user_history(user_id: str = Path(**USER_ID_RULES),
                 service: WellbeingService = Depends(get_service)):
    return {"user_id": user_id, "responses": service.history(user_id)}
