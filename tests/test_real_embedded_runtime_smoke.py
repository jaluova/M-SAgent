from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.common import ImageRef, ModuleStatus
from msagent.core.contracts.types import (
    ImplicitnessLevel,
    ProposalRoute,
    ProposalStatus,
    QueryUnderstandingResult,
    ReferentNumber,
    TargetType,
)
from msagent.infra.embedded_locator import EmbeddedLocatorAdapter
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.runtime.shared_qwen_backbone import QwenSharedBackboneProvider
from msagent.infra.runtime.train_adapter_runtime import EmbeddedGridGroundTrainAdapterRuntime
from msagent.modules.proposal_engine import (
    DefaultProposalEngineModule,
    LocateProposalRouteHandler,
    ProposalEngineModuleInput,
)


def make_understanding(query_text: str) -> QueryUnderstandingResult:
    return QueryUnderstandingResult(
        understanding_id="u-real-smoke",
        normalized_query=query_text,
        target_summary=query_text,
        target_type=TargetType.OBJECT,
        implicitness=ImplicitnessLevel.EXPLICIT,
        canonical_referent_text=query_text,
        referent_number=ReferentNumber.SINGLE,
        focus_terms=query_text.split(),
    )


@unittest.skipUnless(
    os.environ.get("MSAGENT_ENABLE_REAL_SMOKE") == "1",
    "real embedded runtime smoke test is opt-in",
)
class RealEmbeddedRuntimeSmokeTests(unittest.TestCase):
    def test_shared_qwen_embedded_locator_proposal_chain(self) -> None:
        qwen_model_path = os.environ["M_SAGENT_QWEN_MODEL_PATH"]
        gridground_config_path = os.environ["GRIDGROUND_CONFIG_PATH"]
        gridground_adapter_path = os.environ["GRIDGROUND_ADAPTER_PATH"]
        image_path = Path(os.environ.get("MSAGENT_REAL_SMOKE_IMAGE", ROOT / "old/example/truck.jpg"))
        query_text = os.environ.get("MSAGENT_REAL_SMOKE_QUERY", "truck")

        if not image_path.is_file():
            self.fail(f"real smoke image does not exist: {image_path}")

        provider = QwenSharedBackboneProvider(
            provider_name="real-shared-qwen-provider",
            model_path=qwen_model_path,
            device_map=os.environ.get("MSAGENT_QWEN_DEVICE_MAP", "auto"),
            torch_dtype=os.environ.get("MSAGENT_QWEN_TORCH_DTYPE", "auto"),
            attn_implementation=os.environ.get("MSAGENT_QWEN_ATTN_IMPL"),
        )
        runtime = EmbeddedGridGroundTrainAdapterRuntime.from_files(
            runtime_name="embedded-gridground-real-smoke",
            backbone_provider=provider,
            config_path=gridground_config_path,
            adapter_path=gridground_adapter_path,
            abs_threshold=float(os.environ.get("MSAGENT_REAL_SMOKE_ABS_THRESHOLD", "0.5")),
            rel_ratio=float(os.environ.get("MSAGENT_REAL_SMOKE_REL_RATIO", "0.75")),
            min_k=int(os.environ.get("MSAGENT_REAL_SMOKE_MIN_K", "1")),
            max_k=int(os.environ.get("MSAGENT_REAL_SMOKE_MAX_K", "3")),
        )

        try:
            with TemporaryDirectory() as tmp_dir:
                store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))
                module = DefaultProposalEngineModule(
                    route_handlers={
                        ProposalRoute.LOCATE: LocateProposalRouteHandler(
                            locator_adapter=EmbeddedLocatorAdapter(
                                backend_name="embedded-locator-real-smoke",
                                runtime=runtime,
                            )
                        )
                    },
                    artifact_store=store,
                )

                output = module.run(
                    ProposalEngineModuleInput(
                        task_id="task-real-smoke",
                        attempt_index=1,
                        understanding=make_understanding(query_text),
                        image_ref=ImageRef(uri=str(image_path)),
                        preferred_route=ProposalRoute.LOCATE,
                    )
                )
        finally:
            provider.close()

        self.assertIs(output.status, ModuleStatus.SUCCESS)
        self.assertIsNotNone(output.primary_payload)
        proposal = output.primary_payload
        assert proposal is not None
        self.assertIs(proposal.status, ProposalStatus.READY)
        self.assertGreaterEqual(len(proposal.candidates), 1)
        self.assertGreaterEqual(len(proposal.candidates[0].positive_point_hints), 1)
        self.assertTrue(
            any(
                diagnostic.startswith("runtime_metadata=")
                for diagnostic in proposal.diagnostics
            )
        )
