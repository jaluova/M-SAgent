from __future__ import annotations
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.contracts.adapter_requests import SegmentAdapterRequest
from msagent.core.contracts.types import (
    BoxPrompt,
    ExecutionHints,
    PointPrompt,
    PromptMetadata,
    PromptPackage,
    PromptTextBundle,
    ProposalRoute,
    SegmentationStatus,
    SpatialPromptBundle,
)
from msagent.infra.local_artifact_store import LocalFileArtifactStore
from msagent.infra.mask_artifact import MaskArtifact
from msagent.infra.sam3_adapter import (
    RealSAM3Adapter,
    RealSAM3AdapterConfig,
    _LoadedSAM3Runtime,
    _SAM3MaskPrediction,
    _SAM3RuntimePrediction,
    build_real_sam3_adapter_bundle,
)


class FakeSAM3Runtime(_LoadedSAM3Runtime):
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, *, image_path: Path, prompt_package: PromptPackage) -> _SAM3RuntimePrediction:
        self.calls += 1
        self.last_image_path = image_path
        self.last_prompt_package = prompt_package
        return _SAM3RuntimePrediction(
            prompt_mode="spatial",
            masks=[
                _SAM3MaskPrediction(
                    mask_bitmap=[
                        [False, False, False, False],
                        [False, True, True, False],
                        [False, True, True, False],
                        [False, False, False, False],
                    ],
                    score=0.93,
                ),
                _SAM3MaskPrediction(
                    mask_bitmap=[
                        [False, False, False, False],
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, False, False],
                    ],
                    score=0.71,
                ),
            ],
            diagnostics=["sam3.runtime=fake"],
        )

    def close(self) -> None:
        return None


def make_prompt_package() -> PromptPackage:
    return PromptPackage(
        package_id="pkg-1",
        package_version="v1",
        text_prompts=PromptTextBundle(
            normalized_text="the red cup",
            raw_text="the red cup",
        ),
        spatial_prompts=SpatialPromptBundle(),
        metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
        execution_hints=ExecutionHints(
            multimask=True,
            return_top_k=2,
        ),
    )


def write_fake_external_sam_repo(
    repo_root: Path,
    *,
    marker: str,
    include_lazy_module: bool = False,
) -> tuple[Path, Path, Path]:
    package_root = repo_root / "sam3"
    model_root = package_root / "model"
    assets_root = package_root / "assets"
    model_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = repo_root / f"{marker}.pt"
    checkpoint_path.write_bytes(b"fake-checkpoint")

    custom_bpe_path = repo_root / f"{marker}-custom-bpe.txt.gz"
    custom_bpe_path.write_text("fake-bpe", encoding="utf-8")
    (assets_root / "bpe_simple_vocab_16e6.txt.gz").write_text("default-bpe", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (model_root / "__init__.py").write_text("", encoding="utf-8")
    model_builder_lines = [
        "MARKER = {!r}".format(marker),
        "",
        "class FakeModel:",
        "    def __init__(self, *, bpe_path, checkpoint_path):",
        "        self.marker = MARKER",
        "        self.bpe_path = bpe_path",
        "        self.checkpoint_path = checkpoint_path",
        "        self.cpu_calls = 0",
        "        self.last_predict_kwargs = None",
        "",
        "    def cpu(self):",
        "        self.cpu_calls += 1",
        "",
        "    def predict_inst(self, inference_state, point_coords=None, point_labels=None, box=None, multimask_output=True, normalize_coords=True):",
        "        self.last_predict_kwargs = {",
        "            'point_coords': point_coords,",
        "            'point_labels': point_labels,",
        "            'box': box,",
        "            'multimask_output': multimask_output,",
        "            'normalize_coords': normalize_coords,",
        "        }",
        "        return [[[1, 1], [0, 0]]], [0.9], None",
        "",
    ]
    if include_lazy_module:
        model_builder_lines.extend(
            [
                "def build_sam3_image_model(bpe_path=None, checkpoint_path=None, load_from_HF=True, enable_inst_interactivity=False):",
                "    from sam3 import late_module",
                "    model = FakeModel(bpe_path=bpe_path, checkpoint_path=checkpoint_path)",
                "    model.late_marker = late_module.LATE_MARKER",
                "    return model",
            ]
        )
        (package_root / "late_module.py").write_text(
            "LATE_MARKER = {!r}\n".format(marker),
            encoding="utf-8",
        )
    else:
        model_builder_lines.extend(
            [
                "def build_sam3_image_model(bpe_path=None, checkpoint_path=None, load_from_HF=True, enable_inst_interactivity=False):",
                "    return FakeModel(bpe_path=bpe_path, checkpoint_path=checkpoint_path)",
            ]
        )
    (package_root / "model_builder.py").write_text(
        "\n".join(model_builder_lines),
        encoding="utf-8",
    )
    (model_root / "sam3_image_processor.py").write_text(
        "\n".join(
            [
                "class Sam3Processor:",
                "    def __init__(self, model):",
                "        self.model = model",
                "",
                "    def set_image(self, image):",
                "        return {'image': image}",
                "",
                "    def set_text_prompt(self, state, prompt):",
                "        return {'masks': [[[1, 0], [0, 0]]], 'scores': [0.8]}",
            ]
        ),
        encoding="utf-8",
    )
    return repo_root, checkpoint_path, custom_bpe_path


def write_fake_flat_layout_sam_repo(repo_root: Path) -> tuple[Path, Path]:
    model_root = repo_root / "model"
    assets_root = repo_root / "assets"
    model_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)

    checkpoint_path = repo_root / "flat-layout.pt"
    checkpoint_path.write_bytes(b"fake-checkpoint")
    (repo_root / "model_builder.py").write_text(
        "\n".join(
            [
                "def build_sam3_image_model(**kwargs):",
                "    raise RuntimeError('should not be imported for flat layout')",
            ]
        ),
        encoding="utf-8",
    )
    (model_root / "__init__.py").write_text("", encoding="utf-8")
    (model_root / "sam3_image_processor.py").write_text(
        "\n".join(
            [
                "class Sam3Processor:",
                "    def __init__(self, model):",
                "        self.model = model",
            ]
        ),
        encoding="utf-8",
    )
    (assets_root / "bpe_simple_vocab_16e6.txt.gz").write_text("default-bpe", encoding="utf-8")
    return repo_root, checkpoint_path


class RealSAM3AdapterTests(unittest.TestCase):
    def test_real_sam3_adapter_maps_runtime_masks_to_segmentation_candidates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"not-a-real-image-needed-for-fake-runtime")
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            runtime = FakeSAM3Runtime()
            adapter = RealSAM3Adapter(
                backend_name="sam3-real",
                model_path="/models/sam3",
                checkpoint_path="/models/sam3.pt",
                artifact_store=store,
                runtime=runtime,
            )

            result = adapter.segment(
                SegmentAdapterRequest(
                    task_id="task-1",
                    image_uri=str(image_path),
                    prompt_package=make_prompt_package(),
                )
            )

            self.assertIs(result.status, SegmentationStatus.READY)
            self.assertEqual(len(result.candidates), 2)
            self.assertEqual(result.primary_candidate_id, "task-1-segmentation-candidate-1")
            self.assertEqual(adapter.segment_calls, 1)
            self.assertEqual(runtime.calls, 1)
            self.assertIn("prompt_mode=spatial", result.diagnostics)
            self.assertIn("candidate_count=2", result.diagnostics)

            first_mask = store.load_artifact(result.candidates[0].mask_ref, MaskArtifact)
            second_mask = store.load_artifact(result.candidates[1].mask_ref, MaskArtifact)

            self.assertEqual(first_mask.label, "sam3_mask")
            self.assertEqual(first_mask.backend_name, "sam3-real")
            self.assertEqual(first_mask.prompt_mode, "spatial")
            self.assertEqual(first_mask.pixel_area, 4)
            self.assertEqual(
                first_mask.mask_bitmap,
                [
                    [False, False, False, False],
                    [False, True, True, False],
                    [False, True, True, False],
                    [False, False, False, False],
                ],
            )
            self.assertAlmostEqual(first_mask.active_box.x1, 0.25)
            self.assertAlmostEqual(first_mask.active_box.x2, 0.75)
            self.assertEqual(second_mask.pixel_area, 4)
            self.assertEqual(
                result.candidates[0].notes,
                ["generated by RealSAM3Adapter", "prompt_mode=spatial"],
            )

    def test_real_sam3_adapter_returns_failed_result_for_unsupported_uri_scheme(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            adapter = RealSAM3Adapter(
                backend_name="sam3-real",
                model_path="/models/sam3",
                checkpoint_path="/models/sam3.pt",
                artifact_store=store,
                runtime=FakeSAM3Runtime(),
            )

            result = adapter.segment(
                SegmentAdapterRequest(
                    task_id="task-unsupported-uri",
                    image_uri="https://example.com/image.png",
                    prompt_package=make_prompt_package(),
                )
            )

            self.assertIs(result.status, SegmentationStatus.FAILED)
            self.assertIn("sam3.failed", result.diagnostics)
            self.assertTrue(
                any(message.startswith("reason=unsupported_image_uri_scheme") for message in result.diagnostics)
            )

    def test_real_sam3_bundle_loads_fake_external_repo_and_releases_import_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root, checkpoint_path, custom_bpe_path = write_fake_external_sam_repo(
                Path(tmp_dir) / "repo-one",
                marker="repo-one",
            )
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))

            bundle = build_real_sam3_adapter_bundle(
                RealSAM3AdapterConfig(
                    sam_model_path=str(repo_root),
                    checkpoint_path=str(checkpoint_path),
                    bpe_path=str(custom_bpe_path),
                ),
                artifact_store=store,
            )
            model = bundle.sam_adapter.runtime.model

            self.assertEqual(model.marker, "repo-one")
            self.assertEqual(model.bpe_path, str(custom_bpe_path))
            self.assertEqual(model.checkpoint_path, str(checkpoint_path))
            self.assertIsNotNone(_LoadedSAM3Runtime._active_import_state)

            bundle.close()

            self.assertEqual(model.cpu_calls, 1)
            self.assertIsNone(_LoadedSAM3Runtime._active_import_state)
            self.assertNotIn("sam3.model_builder", sys.modules)
            self.assertNotIn("sam3.model.sam3_image_processor", sys.modules)

    def test_real_sam3_bundle_rejects_second_distinct_active_repo_until_first_closes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_one_root, checkpoint_one, custom_bpe_one = write_fake_external_sam_repo(
                Path(tmp_dir) / "repo-one",
                marker="repo-one",
            )
            repo_two_root, checkpoint_two, custom_bpe_two = write_fake_external_sam_repo(
                Path(tmp_dir) / "repo-two",
                marker="repo-two",
            )
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))

            bundle_one = build_real_sam3_adapter_bundle(
                RealSAM3AdapterConfig(
                    sam_model_path=str(repo_one_root),
                    checkpoint_path=str(checkpoint_one),
                    bpe_path=str(custom_bpe_one),
                ),
                artifact_store=store,
            )
            try:
                with self.assertRaisesRegex(RuntimeError, "multiple distinct external SAM"):
                    build_real_sam3_adapter_bundle(
                        RealSAM3AdapterConfig(
                            sam_model_path=str(repo_two_root),
                            checkpoint_path=str(checkpoint_two),
                            bpe_path=str(custom_bpe_two),
                        ),
                        artifact_store=store,
                    )
            finally:
                bundle_one.close()

            bundle_two = build_real_sam3_adapter_bundle(
                RealSAM3AdapterConfig(
                    sam_model_path=str(repo_two_root),
                    checkpoint_path=str(checkpoint_two),
                    bpe_path=str(custom_bpe_two),
                ),
                artifact_store=store,
            )
            try:
                self.assertEqual(bundle_two.sam_adapter.runtime.model.marker, "repo-two")
            finally:
                bundle_two.close()

    def test_real_sam3_bundle_close_removes_lazy_imported_modules_before_next_repo_loads(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_one_root, checkpoint_one, custom_bpe_one = write_fake_external_sam_repo(
                Path(tmp_dir) / "repo-one",
                marker="repo-one",
                include_lazy_module=True,
            )
            repo_two_root, checkpoint_two, custom_bpe_two = write_fake_external_sam_repo(
                Path(tmp_dir) / "repo-two",
                marker="repo-two",
                include_lazy_module=True,
            )
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))

            bundle_one = build_real_sam3_adapter_bundle(
                RealSAM3AdapterConfig(
                    sam_model_path=str(repo_one_root),
                    checkpoint_path=str(checkpoint_one),
                    bpe_path=str(custom_bpe_one),
                ),
                artifact_store=store,
            )
            model_one = bundle_one.sam_adapter.runtime.model
            self.assertEqual(model_one.late_marker, "repo-one")
            self.assertIn("sam3.late_module", sys.modules)
            bundle_one.close()

            self.assertNotIn("sam3.late_module", sys.modules)

    def test_real_sam3_bundle_rejects_flat_layout_repo(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            repo_root, checkpoint_path = write_fake_flat_layout_sam_repo(
                Path(tmp_dir) / "flat-repo"
            )
            store = LocalFileArtifactStore(str(Path(tmp_dir) / "artifacts"))

            with self.assertRaisesRegex(
                ValueError,
                "supports only external code directories with a 'sam3/' package layout",
            ):
                build_real_sam3_adapter_bundle(
                    RealSAM3AdapterConfig(
                        sam_model_path=str(repo_root),
                        checkpoint_path=str(checkpoint_path),
                    ),
                    artifact_store=store,
                )

    def test_loaded_runtime_passes_normalized_spatial_prompts_without_re_normalizing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            repo_root, checkpoint_path, custom_bpe_path = write_fake_external_sam_repo(
                tmp_path / "repo-one",
                marker="repo-one",
            )
            store = LocalFileArtifactStore(str(tmp_path / "artifacts"))
            image_path = tmp_path / "input.png"
            image_path.write_bytes(b"fake-image")

            bundle = build_real_sam3_adapter_bundle(
                RealSAM3AdapterConfig(
                    sam_model_path=str(repo_root),
                    checkpoint_path=str(checkpoint_path),
                    bpe_path=str(custom_bpe_path),
                ),
                artifact_store=store,
            )
            try:
                prompt_package = PromptPackage(
                    package_id="pkg-spatial",
                    package_version="v1",
                    text_prompts=PromptTextBundle(
                        normalized_text="truck",
                        raw_text="truck",
                    ),
                    spatial_prompts=SpatialPromptBundle(
                        positive_points=[
                            PointPrompt(x=0.45, y=0.6),
                        ],
                        negative_points=[],
                        boxes=[
                            BoxPrompt(x1=0.2, y1=0.25, x2=0.8, y2=0.9),
                        ],
                    ),
                    metadata=PromptMetadata(produced_from_route=ProposalRoute.LOCATE),
                    execution_hints=ExecutionHints(multimask=False),
                )

                bundle.sam_adapter.runtime.predict(
                    image_path=image_path,
                    prompt_package=prompt_package,
                )

                last_predict_kwargs = bundle.sam_adapter.runtime.model.last_predict_kwargs
                self.assertIsNotNone(last_predict_kwargs)
                self.assertFalse(last_predict_kwargs["normalize_coords"])
                self.assertEqual(last_predict_kwargs["point_coords"], [[0.45, 0.6]])
                self.assertEqual(last_predict_kwargs["point_labels"], [1])
                self.assertEqual(last_predict_kwargs["box"], [0.2, 0.25, 0.8, 0.9])
            finally:
                bundle.close()


if __name__ == "__main__":
    unittest.main()
