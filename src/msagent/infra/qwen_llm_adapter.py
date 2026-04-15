"""可选真实 Qwen LLM adapter 装配。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

from msagent.core.contracts.adapter_requests import (
    EvaluationAdapterRequest,
    QueryUnderstandingAdapterRequest,
)
from msagent.core.contracts.types import (
    EvaluationResult,
    EvaluationVerdict,
    FailureType,
    ImplicitnessLevel,
    QueryUnderstandingResult,
    ReferentNumber,
    TargetType,
)
from msagent.infra.adapters import LLMAdapter
from msagent.infra.runtime.shared_qwen_backbone import QwenSharedBackboneProvider


@dataclass(slots=True)
class RealQwenLLMAdapterConfig:
    """真实 Qwen LLM adapter 的最小配置。"""

    qwen_model_path: str
    provider_name: str = "service-real-llm-qwen-provider"
    backend_name: str = "qwen2.5-vl-real-llm"
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = True
    max_new_tokens: int = 256


@dataclass(slots=True)
class RealQwenLLMAdapterBundle:
    """把真实 Qwen LLM adapter 及其资源打包为受控装配单元。"""

    llm_adapter: "RealQwenLLMAdapter"
    backbone_provider: QwenSharedBackboneProvider | None = None
    owns_provider: bool = False

    def close(self) -> None:
        if self.owns_provider and self.backbone_provider is not None:
            self.backbone_provider.close()


@dataclass(slots=True, kw_only=True)
class RealQwenLLMAdapter(LLMAdapter):
    """最小真实 Qwen LLM adapter。

    当前阶段只承诺：
    - query understanding 走真实模型生成
    - evaluator 走真实模型生成
    - 若解析失败，回退到稳定启发式，避免主链直接因格式漂移崩掉
    """

    backbone_provider: QwenSharedBackboneProvider
    max_new_tokens: int = 256
    generation_override: Callable[[str], str] | None = field(default=None, repr=False)

    def run_query_understanding(
        self, request: QueryUnderstandingAdapterRequest
    ) -> QueryUnderstandingResult:
        prompt = self._build_query_understanding_prompt(request)
        try:
            payload = self._parse_json_object(self._generate_text(prompt))
            return self._query_understanding_from_payload(request, payload)
        except Exception:
            return self._fallback_query_understanding(request)

    def run_evaluation(self, request: EvaluationAdapterRequest) -> EvaluationResult:
        prompt = self._build_evaluation_prompt(request)
        try:
            payload = self._parse_json_object(self._generate_text(prompt))
            return self._evaluation_from_payload(request, payload)
        except Exception:
            return self._fallback_evaluation(request)

    def _generate_text(self, prompt: str) -> str:
        if self.generation_override is not None:
            return self.generation_override(prompt)

        model, processor = self.backbone_provider.get_loaded_components()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        chat_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[chat_prompt], return_tensors="pt")
        prepared_inputs = self._move_inputs_to_model_device(inputs, model)
        generated = model.generate(
            **prepared_inputs,
            max_new_tokens=self.max_new_tokens,
        )
        input_ids = prepared_inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[-1] :]
        decoded = processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0] if decoded else ""

    def _build_query_understanding_prompt(
        self,
        request: QueryUnderstandingAdapterRequest,
    ) -> str:
        return (
            "You are the query understanding stage of an image agent.\n"
            "Return one JSON object with keys: normalized_query, target_summary, "
            "target_type, implicitness, canonical_referent_text, referent_number, "
            "focus_terms, attribute_clues, notes.\n"
            "Allowed target_type: person, object, stuff, part, text, unknown.\n"
            "Allowed implicitness: explicit, mixed, implicit.\n"
            "Allowed referent_number: single, multiple, unknown.\n"
            f"Raw query: {request.raw_query!r}\n"
            f"User context: {(request.user_context_text or '')!r}\n"
            f"Image uri: {(request.image_uri or '')!r}\n"
            "Respond with JSON only."
        )

    def _build_evaluation_prompt(self, request: EvaluationAdapterRequest) -> str:
        return (
            "You are the evaluation stage of an image agent.\n"
            "Return one JSON object with keys: verdict, summary, failure_type, "
            "confidence, retry_hints.\n"
            "Allowed verdict: accept, reject, review.\n"
            "Allowed failure_type: localization_error, partial_mask, wrong_instance, "
            "prompt_mismatch, or null.\n"
            f"Raw query: {request.raw_query!r}\n"
            f"Prompt normalized text: {request.prompt_package.text_prompts.normalized_text!r}\n"
            f"Segmentation status: {request.segmentation.status.value!r}\n"
            f"Segmentation summary: {request.segmentation.result_summary!r}\n"
            f"Candidate count: {len(request.segmentation.candidates)}\n"
            f"Primary candidate id: {(request.segmentation.primary_candidate_id or '')!r}\n"
            "Respond with JSON only."
        )

    def _fallback_query_understanding(
        self,
        request: QueryUnderstandingAdapterRequest,
    ) -> QueryUnderstandingResult:
        normalized_query = request.raw_query.strip()
        focus_terms = _split_focus_terms(normalized_query)
        return QueryUnderstandingResult(
            understanding_id=f"{request.task_id}-understanding",
            normalized_query=normalized_query,
            target_summary=f"Target inferred from query: {normalized_query}",
            target_type=TargetType.OBJECT,
            implicitness=ImplicitnessLevel.EXPLICIT,
            canonical_referent_text=normalized_query,
            referent_number=ReferentNumber.SINGLE,
            focus_terms=focus_terms,
            attribute_clues=list(focus_terms),
            notes=["fallback generated by RealQwenLLMAdapter"],
        )

    def _fallback_evaluation(
        self,
        request: EvaluationAdapterRequest,
    ) -> EvaluationResult:
        primary_candidate = next(
            (
                candidate
                for candidate in request.segmentation.candidates
                if candidate.candidate_id == request.segmentation.primary_candidate_id
            ),
            request.segmentation.candidates[0] if request.segmentation.candidates else None,
        )
        if primary_candidate is not None:
            return EvaluationResult(
                evaluation_id=f"{request.task_id}-evaluation",
                verdict=EvaluationVerdict.ACCEPT,
                summary="Real Qwen fallback accepted the primary segmentation candidate.",
                accepted_candidate_id=primary_candidate.candidate_id,
                accepted_mask_ref=primary_candidate.mask_ref,
                confidence=0.5,
            )
        return EvaluationResult(
            evaluation_id=f"{request.task_id}-evaluation",
            verdict=EvaluationVerdict.REJECT,
            summary="Real Qwen fallback rejected because no segmentation candidate was available.",
            failure_type=FailureType.LOCALIZATION_ERROR,
            confidence=0.5,
            retry_hints=["retry_with_same_route"],
        )

    def _query_understanding_from_payload(
        self,
        request: QueryUnderstandingAdapterRequest,
        payload: dict[str, object],
    ) -> QueryUnderstandingResult:
        normalized_query = _read_payload_str(
            payload,
            "normalized_query",
            default=request.raw_query.strip(),
        )
        focus_terms = _read_payload_str_list(
            payload,
            "focus_terms",
            default=_split_focus_terms(normalized_query),
        )
        attribute_clues = _read_payload_str_list(
            payload,
            "attribute_clues",
            default=list(focus_terms),
        )
        notes = _read_payload_str_list(payload, "notes", default=[])
        return QueryUnderstandingResult(
            understanding_id=f"{request.task_id}-understanding",
            normalized_query=normalized_query,
            target_summary=_read_payload_str(
                payload,
                "target_summary",
                default=f"Target inferred from query: {normalized_query}",
            ),
            target_type=_read_target_type(payload.get("target_type")),
            implicitness=_read_implicitness(payload.get("implicitness")),
            canonical_referent_text=_read_payload_str(
                payload,
                "canonical_referent_text",
                default=normalized_query,
            ),
            referent_number=_read_referent_number(payload.get("referent_number")),
            focus_terms=focus_terms,
            attribute_clues=attribute_clues,
            notes=notes,
        )

    def _evaluation_from_payload(
        self,
        request: EvaluationAdapterRequest,
        payload: dict[str, object],
    ) -> EvaluationResult:
        verdict = _read_evaluation_verdict(payload.get("verdict"))
        primary_candidate = next(
            (
                candidate
                for candidate in request.segmentation.candidates
                if candidate.candidate_id == request.segmentation.primary_candidate_id
            ),
            request.segmentation.candidates[0] if request.segmentation.candidates else None,
        )
        accepted_candidate_id = None
        accepted_mask_ref = None
        if verdict is EvaluationVerdict.ACCEPT and primary_candidate is not None:
            accepted_candidate_id = primary_candidate.candidate_id
            accepted_mask_ref = primary_candidate.mask_ref
        return EvaluationResult(
            evaluation_id=f"{request.task_id}-evaluation",
            verdict=verdict,
            summary=_read_payload_str(
                payload,
                "summary",
                default="Real Qwen evaluation completed.",
            ),
            failure_type=_read_failure_type(payload.get("failure_type")),
            accepted_candidate_id=accepted_candidate_id,
            accepted_mask_ref=accepted_mask_ref,
            confidence=_read_optional_float(payload.get("confidence")),
            retry_hints=_read_payload_str_list(payload, "retry_hints", default=[]),
        )

    @staticmethod
    def _parse_json_object(raw_text: str) -> dict[str, object]:
        stripped = raw_text.strip()
        if not stripped:
            raise ValueError("Real Qwen adapter returned empty text")
        if stripped.startswith("```"):
            lines = [line for line in stripped.splitlines() if not line.startswith("```")]
            stripped = "\n".join(lines).strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("Real Qwen adapter did not return a JSON object")
        payload = json.loads(stripped[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Real Qwen adapter JSON payload must be an object")
        return payload

    @staticmethod
    def _move_inputs_to_model_device(inputs: object, model: object) -> dict[str, object]:
        if hasattr(inputs, "to"):
            try:
                moved = inputs.to(next(model.parameters()).device)
                if isinstance(moved, dict):
                    return dict(moved)
                if hasattr(moved, "items"):
                    return dict(moved.items())
            except Exception:
                pass

        target_device = next(model.parameters()).device
        prepared: dict[str, object] = {}
        if isinstance(inputs, dict):
            iterator = inputs.items()
        else:
            iterator = inputs.items()
        for key, value in iterator:
            if hasattr(value, "to"):
                prepared[key] = value.to(target_device)
            else:
                prepared[key] = value
        return prepared


def build_real_qwen_llm_adapter_bundle(
    config: RealQwenLLMAdapterConfig,
    *,
    shared_backbone_provider: QwenSharedBackboneProvider | None = None,
) -> RealQwenLLMAdapterBundle:
    """构造可接入 query understanding / evaluator 的真实 Qwen LLM adapter。"""
    provider = shared_backbone_provider
    owns_provider = False
    if provider is None:
        provider = QwenSharedBackboneProvider(
            provider_name=config.provider_name,
            model_path=str(Path(config.qwen_model_path).expanduser()),
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            trust_remote_code=config.trust_remote_code,
        )
        owns_provider = True
    return RealQwenLLMAdapterBundle(
        llm_adapter=RealQwenLLMAdapter(
            backend_name=config.backend_name,
            model_path=str(Path(config.qwen_model_path).expanduser()),
            backbone_provider=provider,
            max_new_tokens=config.max_new_tokens,
        ),
        backbone_provider=provider,
        owns_provider=owns_provider,
    )


def _split_focus_terms(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw_term in text.replace("_", " ").replace("-", " ").split():
        term = raw_term.strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def _read_payload_str(
    payload: dict[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return value.strip()


def _read_payload_str_list(
    payload: dict[str, object],
    key: str,
    *,
    default: list[str],
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return list(default)
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
    return result or list(default)


def _read_target_type(value: object) -> TargetType:
    if isinstance(value, str):
        try:
            return TargetType(value.strip().lower())
        except ValueError:
            pass
    return TargetType.OBJECT


def _read_implicitness(value: object) -> ImplicitnessLevel:
    if isinstance(value, str):
        try:
            return ImplicitnessLevel(value.strip().lower())
        except ValueError:
            pass
    return ImplicitnessLevel.EXPLICIT


def _read_referent_number(value: object) -> ReferentNumber:
    if isinstance(value, str):
        try:
            return ReferentNumber(value.strip().lower())
        except ValueError:
            pass
    return ReferentNumber.SINGLE


def _read_evaluation_verdict(value: object) -> EvaluationVerdict:
    if isinstance(value, str):
        try:
            return EvaluationVerdict(value.strip().lower())
        except ValueError:
            pass
    return EvaluationVerdict.REJECT


def _read_failure_type(value: object) -> FailureType | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return FailureType(value.strip().lower())
        except ValueError:
            return None
    return None


def _read_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
