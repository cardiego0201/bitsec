import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .cache import AnalysisCache, CachedAnalysisResult
from .find import FileMatch, find_similar_files
from .model import VulnerabilityModel
from .schemas import AnalysisRequest, AnalysisResponse


MODEL_ID = "msc-smart-contract-auditing/deepseek-coder-6.7b-vulnerability-detection"
CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "deepseek-coder-6.7b"

app = FastAPI(
    title="BitSec Vulnerability Detection API",
    description="Detect vulnerabilities in Solidity smart contracts using the DeepSeek model.",
    version="1.0.0",
)

model = VulnerabilityModel(model_id=MODEL_ID, cache_dir=CACHE_DIR)
analysis_cache = AnalysisCache()


@app.on_event("startup")
async def load_model() -> None:
    """Load the HuggingFace model when the service starts."""
    await analysis_cache.connect()
    await model.ensure_loaded()


@app.on_event("shutdown")
async def shutdown() -> None:
    await analysis_cache.close()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze Solidity code and identify the vulnerability present."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="The provided code is empty.")
      
    vulner_code = request.code
    code_hash = analysis_cache.compute_hash(vulner_code)

    entry = await analysis_cache.get_entry(code_hash)
    if entry:
        status = entry.get("status")
        if status == "completed" and "result" in entry:
            cached_result = CachedAnalysisResult.from_dict(entry["result"])
            return AnalysisResponse(**cached_result.to_dict())
        if status == "processing":
            try:
                cached_result = await analysis_cache.wait_for_result(code_hash)
            except TimeoutError as exc:
                raise HTTPException(status_code=503, detail="Analysis is still processing.") from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            if cached_result:
                return AnalysisResponse(**cached_result.to_dict())
        if status == "failed":
            # Attempt to reprocess the request.
            pass

    started = await analysis_cache.start_processing(code_hash)
    if not started:
        try:
            cached_result = await analysis_cache.wait_for_result(code_hash)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail="Analysis is still processing.") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if cached_result:
            return AnalysisResponse(**cached_result.to_dict())
        # Fallback: entry disappeared unexpectedly.
        raise HTTPException(status_code=500, detail="Analysis cache entry is unavailable.")

    try:
        matching_file: FileMatch | None = await asyncio.to_thread(find_similar_files, vulner_code)
        clear_code = matching_file.file_content if matching_file else None

        vulnerability, confidence, rationale = await model.predict(vulner_code, clear_code)
        cached_result = CachedAnalysisResult(
            vulnerability=vulnerability,
            confidence=confidence,
            rationale=rationale,
        )
        await analysis_cache.mark_completed(
            code_hash,
            cached_result,
            clear_code,
        )
    except ValueError as exc:
        await analysis_cache.mark_failed(code_hash, str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        await analysis_cache.mark_failed(code_hash, str(exc))
        raise HTTPException(status_code=500, detail="Model inference failed.") from exc

    return AnalysisResponse(
        vulnerability=vulnerability,
        confidence=confidence,
        rationale=rationale,
    )


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}

