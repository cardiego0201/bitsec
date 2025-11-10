from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    code: str = Field(..., description="Solidity source code to audit.")


class AnalysisResponse(BaseModel):
    vulnerability: str = Field(..., description="Predicted vulnerability.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")
    rationale: str = Field(..., description="Explanation provided by the model.")

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    code: str = Field(..., description="Solidity source code to analyze.")


class AnalysisResponse(BaseModel):
    vulnerability: str = Field(..., description="Predicted vulnerability label.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of prediction.")
    rationale: str = Field(..., description="Explanation for the prediction.")


