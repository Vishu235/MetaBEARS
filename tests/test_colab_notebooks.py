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
                    ),
                    start=1,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
