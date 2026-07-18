# MetaBEARS

**Meta-Cognitive Neuro-Symbolic Learning: Detecting Reasoning Shortcuts and Calibrating Confidence in Hybrid AI Systems**

MetaBEARS is the Phase II working repository for extending the BEARS neuro-symbolic baseline completed during Phase I. The central research goal is to move from aggregate uncertainty estimates to an interpretable, sample-level meta-cognitive report that distinguishes:

- symbolic confidence: how strongly the prediction satisfies the encoded knowledge;
- neural familiarity: how similar the input and representation are to the training distribution;
- shortcut risk: how likely the predicted concepts are semantically wrong despite a correct task label.

The proposed novelty combines a **concept consistency probe** with a **meta-cognitive confidence layer**. An initial framework-independent prototype is implemented; integration with trained BEARS checkpoints and benchmark validation are still in progress.

## Current status

- Phase I: complete — environment restoration, baseline reproduction, diagnostics, HalfMNIST/BEARS evaluation support, MiniKandinsky smoke testing, and a practical BDD-OIA reconstruction.
- Phase II: in progress — concept consistency, empirical neural familiarity, separated confidence reporting, serialization, and a deterministic demonstration are implemented as the first vertical slice.

Phase II will proceed from baseline equivalence to the consistency probe, the three-signal confidence report, and controlled shortcut-detection experiments.

## Repository layout

```text
MetaBEARS/
|-- XOR_MNIST/                 # Main BEARS experimental code
|   `-- metacog/               # Boundary for the Phase II extension
|-- BDD_OIA/                   # Practical BDD-OIA reconstruction
|-- colab/                     # Colab notebooks
|-- docs/
|   `-- base-paper/            # Preserved upstream citation metadata
|-- colab_runner.py            # Reproducible experiment/smoke-test entry point
`-- requirements*.txt          # Runtime, Colab, and development dependencies
```

## Quick start

The original work targeted Python 3.9. Create an isolated environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inspect the available reproducible jobs:

```powershell
python colab_runner.py --help
python colab_runner.py --job diagnostics
python colab_runner.py --job metabears_demo
```

Run the focused Phase II tests without additional test dependencies:

```powershell
python -m unittest discover -s tests -v
```

Dataset archives, generated features, checkpoints, logs, and raw outputs are deliberately excluded from Git. Keep all data-dependent artifacts outside version control and record the command, configuration, seed, and dataset version for each experiment.

## Version-control workflow

Use small branches and commits tied to one experiment or feature:

```powershell
git switch -c feature/concept-consistency-probe
git add <changed-files>
git commit -m "Implement concept consistency probe"
```

Do not commit datasets, model checkpoints, raw experiment artifacts, credentials, or virtual environments. Commit source code, configuration, result summaries, and the exact command/seed needed to reproduce each result.

## Attribution

This repository builds on the BEARS implementation by Marconato et al. The upstream citation is preserved at [docs/base-paper/CITATION_BEARS.cff](docs/base-paper/CITATION_BEARS.cff), and the inherited Apache 2.0 license remains in [LICENSE](LICENSE).
