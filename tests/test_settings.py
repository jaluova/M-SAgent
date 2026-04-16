from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from msagent.core.config.settings import MSAgentSettings


class SettingsFromEnvTests(unittest.TestCase):
    def test_from_env_reads_real_runtime_paths_and_flags(self) -> None:
        settings = MSAgentSettings.from_env(
            {
                "M_SAGENT_QWEN_MODEL_PATH": "/models/qwen",
                "GRIDGROUND_CONFIG_PATH": "/models/gridground/config.json",
                "GRIDGROUND_ADAPTER_PATH": "/models/gridground/best_model.pth",
                "M_SAGENT_SAM3_MODEL_PATH": "/models/sam3",
                "M_SAGENT_SAM3_CHECKPOINT_PATH": "/models/sam3.pt",
                "MSAGENT_ENABLE_REAL_LLM": "1",
                "MSAGENT_ENABLE_DEBUG_FEATURES": "true",
                "M_SAGENT_SERVER_HOST": "0.0.0.0",
                "M_SAGENT_SERVER_PORT": "9010",
                "MSAGENT_ARTIFACT_ROOT": "/tmp/msagent-artifacts",
                "MSAGENT_MAX_ATTEMPTS": "5",
            }
        )

        self.assertEqual(settings.model_paths.qwen_model_path, "/models/qwen")
        self.assertEqual(
            settings.model_paths.embedded_locator_config_path,
            "/models/gridground/config.json",
        )
        self.assertEqual(
            settings.model_paths.embedded_locator_adapter_path,
            "/models/gridground/best_model.pth",
        )
        self.assertEqual(settings.model_paths.sam_model_path, "/models/sam3")
        self.assertEqual(
            settings.model_paths.sam_checkpoint_path,
            "/models/sam3.pt",
        )
        self.assertTrue(settings.service.enable_real_llm)
        self.assertTrue(settings.service.enable_debug_features)
        self.assertEqual(settings.service.host, "0.0.0.0")
        self.assertEqual(settings.service.port, 9010)
        self.assertEqual(settings.runtime.artifact_root, "/tmp/msagent-artifacts")
        self.assertEqual(settings.runtime.max_attempts, 5)

    def test_from_env_treats_blank_values_as_unset(self) -> None:
        settings = MSAgentSettings.from_env(
            {
                "M_SAGENT_QWEN_MODEL_PATH": "   ",
                "MSAGENT_ENABLE_REAL_LLM": "0",
                "MSAGENT_ENABLE_DEBUG_FEATURES": "0",
            }
        )

        self.assertIsNone(settings.model_paths.qwen_model_path)
        self.assertFalse(settings.service.enable_real_llm)
        self.assertFalse(settings.service.enable_debug_features)

    def test_from_env_rejects_invalid_boolean(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported boolean value"):
            MSAgentSettings.from_env({"MSAGENT_ENABLE_REAL_LLM": "maybe"})


if __name__ == "__main__":
    unittest.main()
