from __future__ import annotations

import py_compile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SkillFileTests(unittest.TestCase):
    def test_skill_frontmatter_is_small_and_specific(self) -> None:
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("---\n"))
        _, frontmatter, _ = text.split("---", 2)
        self.assertIn("name: pytorch-training-optimizer", frontmatter)
        description = next(line for line in frontmatter.splitlines() if line.startswith("description:"))
        self.assertIn("PyTorch training", description)
        self.assertIn("not for inference", description)
        self.assertLess(len(description), 420)

    def test_confirmation_gate_present(self) -> None:
        text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Non-Negotiable Confirmation Gate", text)
        self.assertIn("Do not treat silence as approval", text)
        self.assertIn("plain language", text)
        self.assertTrue((ROOT / "references/change_confirmation.md").exists())

    def test_openai_metadata_shape(self) -> None:
        text = (ROOT / "agents/openai.yaml").read_text(encoding="utf-8")
        for needle in [
            "interface:",
            "display_name:",
            "short_description:",
            "default_prompt:",
            "policy:",
            "allow_implicit_invocation:",
        ]:
            self.assertIn(needle, text)

    def test_reference_files_exist(self) -> None:
        expected = [
            "diagnostic_matrix.md",
            "profiling_workflow.md",
            "optimization_playbook.md",
            "comparison_report.md",
            "literature_and_repos.md",
            "change_confirmation.md",
            "compiler_and_cuda_graphs.md",
            "attention_backend_matrix.md",
            "precision_policy.md",
            "distributed_memory_strategy.md",
            "dataloader_storage.md",
            "checkpoint_goodput.md",
            "large_transformer_training_checklist.md",
            "failure_modes.md",
        ]
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue((ROOT / "references" / name).exists())

    def test_scripts_compile(self) -> None:
        for path in (ROOT / "scripts").glob("*.py"):
            with self.subTest(path=path.name):
                py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    unittest.main()
