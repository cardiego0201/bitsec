from pathlib import Path

from fastapi import FastAPI, HTTPException

from .model import VulnerabilityModel
from .schemas import AnalysisRequest, AnalysisResponse
from .find import find_similar_files


MODEL_ID = "msc-smart-contract-auditing/deepseek-coder-6.7b-vulnerability-detection"
CACHE_DIR = Path(__file__).resolve().parent / "artifacts" / "deepseek-coder-6.7b"

app = FastAPI(
    title="BitSec Vulnerability Detection API",
    description="Detect vulnerabilities in Solidity smart contracts using the DeepSeek model.",
    version="1.0.0",
)

model = VulnerabilityModel(model_id=MODEL_ID, cache_dir=CACHE_DIR)


@app.on_event("startup")
async def load_model() -> None:
    """Load the HuggingFace model when the service starts."""
    await model.ensure_loaded()


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze Solidity code and identify the vulnerability present."""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="The provided code is empty.")
      
    vulner_code = request.code
    clear_code = find_similar_files(vulner_code) or None

    try:
        vulnerability, confidence, rationale = await model.predict(vulner_code, clear_code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
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

