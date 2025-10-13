from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..models import AnalysisResponse
from ..services.analyzer import AnalyzerService
from ..store import run_store

router = APIRouter()


def get_analyzer_service() -> AnalyzerService:
    return AnalyzerService()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    analyzer: AnalyzerService = Depends(get_analyzer_service),
) -> AnalysisResponse:
    try:
        await file.read()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=f"ファイル読み込みに失敗しました: {exc}") from exc

    filename = (file.filename or "").lower()
    if filename and not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="テキストファイル（.txt）のみ対応しています。")

    response = analyzer.analyze()
    run_store.save(response)
    return response


@router.get("/runs/{run_id}", response_model=AnalysisResponse)
async def get_run(run_id: str) -> AnalysisResponse:
    response = run_store.get(run_id)
    if response is None:
        raise HTTPException(status_code=404, detail="指定された run_id は存在しません。")
    return response


@router.post("/search")
async def search_unimplemented() -> dict[str, str]:
    return {"detail": "Not implemented in mock."}


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
