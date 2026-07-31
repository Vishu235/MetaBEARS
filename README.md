# MetaBEARS

**Meta-Cognitive Neuro-Symbolic Learning: Detecting Reasoning Shortcuts and Calibrating Confidence in Hybrid AI Systems**

MetaBEARS is the Phase II working repository for extending the BEARS neuro-symbolic baseline completed during Phase I. The central research goal is to move from aggregate uncertainty estimates to an interpretable, sample-level meta-cognitive report that distinguishes:

- task confidence: the ensemble-mean probability of the predicted task label. This is a stand-in — a genuine symbolic rule-support/entailment signal is not yet implemented, and the field is named for what it actually measures;
- neural familiarity: an interpolated reference-relative percentile derived from member-wise nearest-neighbour distances in learned BEARS representations. The first integration uses flattened pre-softmax concept logits (`CS`) and a leave-one-out validation reference distribution;
- shortcut risk: a proxy for semantically unstable concepts despite a stable, confident task label. It is computed from concept/label (in)stability and task confidence while familiarity remains a separate evidence channel.

Review is the union of two independently triggered paths: a `shortcut_flag` (shortcut risk over a validation-selected threshold) and an `ood_flag` (familiarity at or below a validation-selected threshold). Both are reported alongside `review_flag` so a consumer can tell which evidence triggered review. The separation is structural, not a guarantee that the two flags identify distinct real-world causes.

The proposed novelty combines a **concept consistency probe** with a **meta-cognitive confidence layer**. The framework-independent prototype now includes an adapter for real BEARS ensemble outputs; execution on trained checkpoints and benchmark validation are still in progress.

## Current status

- Phase I: complete — environment restoration, baseline reproduction, diagnostics, HalfMNIST/BEARS evaluation support, MiniKandinsky smoke testing, and a practical BDD-OIA reconstruction.
- Phase II: in progress — concept consistency, member-preserving BEARS output collection, representation-distance computation, distance-to-familiarity mapping, separated confidence reporting, a familiarity-driven review path, validation-based threshold selection (`select_review_threshold`), serialization, and deterministic mechanism checks are implemented.

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
