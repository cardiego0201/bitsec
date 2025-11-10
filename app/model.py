from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Optional, Tuple

from huggingface_hub import snapshot_download
from transformers import GenerationConfig, pipeline
from transformers.pipelines import TextGenerationPipeline


VULNERABILITIES_DIR = Path(__file__).parent / "vulnerabilities"


def _extract_vulnerability_title(content: str, fallback: Path) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
        return stripped
    return fallback.stem.replace("-", " ").title()


def _extract_vulnerability_description(content: str) -> str:
    description_lines = []
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped:
            continue
        if stripped.startswith("#"):
            continue
        stripped = stripped.lstrip("-*").strip()
        if stripped:
            description_lines.append(stripped)
    return " ".join(description_lines)


def _load_vulnerability_descriptions() -> str:
    entries = []
    for path in sorted(VULNERABILITIES_DIR.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        title = _extract_vulnerability_title(content, path)
        description = _extract_vulnerability_description(content)
        if description:
            entries.append(f"{title}: {description}")
        else:
            entries.append(title)
    return "\n".join(entries)


try:
    VULNERABILITY_DESCRIPTIONS = _load_vulnerability_descriptions()
except FileNotFoundError:
    VULNERABILITY_DESCRIPTIONS = ""


class VulnerabilityModel:
    """Wrapper around the HuggingFace DeepSeek vulnerability detection model."""

    def __init__(self, model_id: str, cache_dir: Path) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._local_dir: Optional[str] = None
        self._pipeline: Optional[TextGenerationPipeline] = None
        self._lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._generation_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9,
            do_sample=False,
        )

    async def ensure_loaded(self) -> None:
        """Ensure tokenizer and model weights are available in memory."""
        async with self._lock:
            if self._pipeline is not None:
                return

            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Download model snapshot if not yet cached.
            self._local_dir = snapshot_download(
                repo_id=self._model_id,
                local_dir=str(self._cache_dir),
                local_dir_use_symlinks=False,
            )

            self._pipeline = pipeline(
                task="text-generation",
                model=self._local_dir,
                tokenizer=self._local_dir,
                device_map="auto",
                torch_dtype="auto",
            )

    async def predict(
        self,
        vulnerable_code: str,
        original_code: Optional[str] = None,
    ) -> Tuple[str, float, str]:
        """Generate a vulnerability assessment for the given Solidity code."""
        if not vulnerable_code.strip():
            raise ValueError("Source code is empty.")

        await self.ensure_loaded()
        assert self._pipeline is not None  # for type-checkers

        prompt = self._build_prompt(vulnerable_code, original_code)
        async with self._inference_lock:
            outputs = self._pipeline(
                prompt,
                generation_config=self._generation_config,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
                return_full_text=False,
            )
        decoded = outputs[0]["generated_text"]

        vulnerability, rationale = self._parse_response(decoded)
        confidence = 1.0 if vulnerability != "unknown" else 0.0
        return vulnerability, confidence, rationale

    @staticmethod
    def _build_prompt(vulnerable_code: str, original_code: Optional[str]) -> str:
        """
        Format prompt for the model by comparing original code and vulnerability-injected code.
        `vulnerability_descriptions` is a text mapping vulnerability types to explanations.
        """
        vulnerability_descriptions = VULNERABILITY_DESCRIPTIONS.strip()

        if original_code and original_code.strip() and original_code.strip() != vulnerable_code.strip():
            instructions = (
                "You are an expert Solidity smart contract auditor. "
                "Two versions of a contract are provided:\n"
                "1. The original reference implementation (believed to be safe).\n"
                "2. A modified version that may contain an introduced vulnerability.\n\n"
                "Compare the modified contract against the original, determine which vulnerability "
                "has been introduced, and name the vulnerability using the taxonomy provided.\n"
                "Always answer using:\n"
                "Vulnerability: <name>\n"
                "Explanation: <brief explanation>\n"
            )

            original_section = f"Original Solidity Code:\n{original_code}\n\n"
        else:
            instructions = (
                "You are an expert Solidity smart contract auditor. "
                "Analyze the provided Solidity contract and determine the specific vulnerability it contains. "
                "Use the vulnerability descriptions to align your answer with the established taxonomy.\n"
                "Respond strictly with:\n"
                "Vulnerability: <name>\n"
                "Explanation: <brief explanation>\n"
            )
            original_section = ""

        descriptions_section = (
            f"Vulnerability Type Descriptions:\n{vulnerability_descriptions}\n\n"
            if vulnerability_descriptions
            else ""
        )

        return (
            f"{instructions}\n\n"
            f"{original_section}"
            f"Solidity Contract To Audit:\n{vulnerable_code}\n\n"
            f"{descriptions_section}"
        )

    @staticmethod
    def _parse_response(model_output: str) -> Tuple[str, str]:
        """Extract vulnerability and rationale from model output."""
        vuln_match = re.search(r"Vulnerability:\s*(.+)", model_output, re.IGNORECASE)
        explanation_match = re.search(r"Explanation:\s*(.+)", model_output, re.IGNORECASE)

        vulnerability = vuln_match.group(1).strip() if vuln_match else "unknown"
        rationale = explanation_match.group(1).strip() if explanation_match else model_output.strip()
        return vulnerability, rationale
