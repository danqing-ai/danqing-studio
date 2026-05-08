"""音频创作占位 — plan audios.py（全部 501）。"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/audios", tags=["audios"])


def _nope():
    raise HTTPException(501, detail="audio not implemented yet")


@router.get("/generations")
async def audio_generations_get():
    _nope()


@router.post("/generations")
async def audio_generations_post():
    _nope()


@router.post("/edits")
async def audio_edits():
    _nope()


@router.post("/dubs")
async def audio_dubs():
    _nope()
