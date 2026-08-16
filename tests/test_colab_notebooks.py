"""Structural checks for the runnable Colab notebooks."""

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class MiniKandinskyNotebookTests(unittest.TestCase):
    def test_cells_are_ordered_and_code_compiles(self) -> None:
        path = REPOSITORY_ROOT / "colab" / "MetaBEARS_MiniKandinsky.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]

        self.assertEqual(cells[0]["cell_type"], "markdown")
        self.assertEqual(cells[1]["cell_type"], "markdown")
        titles = []
        for cell_index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            first_line = source.splitlines()[0]
            titles.append(first_line)
            compile(source, f"{path}:cell-{cell_index + 1}", "exec")

        self.assertEqual(
            titles,
            [
                f"#@title {number}. {name}"
                for number, name in enumerate(
                    (
                        "Configuration",
                        "Mount Google Drive",
                        "Clone or update MetaBEARS",
                        "Install Colab-safe dependencies",
                        "Runtime diagnostics",
                        "Validate and stage kand-3k.zip",
                        "Reusable job helper",
                        "One-epoch MiniKandinsky smoke test",
                        "Three-seed MiniKandinsky baseline training",
                        "Resolve the trained ensemble checkpoints",
                        "Reusable MetaBEARS evaluation helper",
                        "One-batch MetaBEARS integration check",
                        "Full MiniKandinsky MetaBEARS evaluation",
                        "Train matched v1 supervision-control ensembles",
                        "Resolve matched v1 checkpoints",
                        "MetaBEARS v1 held-out-shift evaluation helper",
                        "Run the v1 supervision-control matrix",
                        "Train entropy-regularized v2 task-only ensemble",
                        "Validate v2 sweep checkpoints",
                        "Run validation-only representation sweep",
                        "Run fixed-representation v3 scoring sweep",
                        "Run frozen-candidate v4 uncertainty ablation",
                    ),
                    start=1,
                )
            ],
        )


class BddOiaNotebookTests(unittest.TestCase):
    def test_cells_are_ordered_resumable_and_code_compiles(self) -> None:
        path = REPOSITORY_ROOT / "colab" / "MetaBEARS_BDD_OIA.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        titles = []
        sources = []
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            sources.append(source)
            titles.append(source.splitlines()[0])
            compile(source, f"{path}:cell-{cell_index + 1}", "exec")

        self.assertEqual(
            titles,
            [
                f"#@title {number}. {name}"
                for number, name in enumerate(
                    (
                        "Configuration",
                        "Mount Google Drive",
                        "Clone or update MetaBEARS",
                        "Install Colab-safe dependencies",
                        "Diagnostics and input validation",
                        "Reusable BDD job helper",
                        "Restore reusable full ResNet50 features from Drive",
                        "BDD-OIA smoke preprocessing and training",
                        "Full BDD-OIA ResNet50 preprocessing",
                        "Full multi-seed BDD-OIA baseline training",
                        "Build a compact BDD result summary",
                        "Archive repository-side outputs to Drive",
                        "Pull latest MetaBEARS code",
                        "Locate BDD-OIA MetaBEARS checkpoint ensemble",
                        "Run BDD-OIA MetaBEARS evaluation",
                    ),
                    start=1,
                )
            ],
        )
        notebook_source = "\n".join(sources)
        self.assertIn("bears_data", notebook_source)
        self.assertIn("completed_run.json", notebook_source)
        self.assertIn("'_entropy' if entropy_weight", notebook_source)
        self.assertIn("'_csup' if concept_weight", notebook_source)


if __name__ == "__main__":
    unittest.main()
