import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from summarization.summarizer import build_summarizer

summarizer = build_summarizer(
    model_path="checkpoints/finetuned-bart.pt",
    tokenizer_path="tokenizer-bart",
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SummaryRequest(BaseModel):
    text: str
    beam_size: int
    nums: int
    max_length: int


@app.get("/")
def root():
    return {
        "hi": "hihi",
    }


@app.post("/summarize")
def summarize(request: SummaryRequest) -> dict[str, Any]:
    start = time.time()

    summaries = summarizer.summarize(
        text=request.text,
        beam_size=request.beam_size,
        nums=request.nums,
        max_pred_seq_length=request.max_length,
    )

    end = time.time()

    return {
        "summaries": summaries,
        "time": end - start,
    }
