# MetaBEARS

**Meta-Cognitive Neuro-Symbolic Learning: Detecting Reasoning Shortcuts and Calibrating Confidence in Hybrid AI Systems**

MetaBEARS is the Phase II working repository for extending the BEARS neuro-symbolic baseline completed during Phase I. The central research goal is to move from aggregate uncertainty estimates to an interpretable, sample-level meta-cognitive report that distinguishes:

- task confidence: the ensemble-mean probability of the predicted task label. This is a stand-in — a genuine symbolic rule-support/entailment signal is not yet implemented, and the field is named for what it actually measures;
- neural familiarity: an interpolated reference-relative percentile derived from member-wise nearest-neighbour distances in learned BEARS representations. The first integration uses flattened pre-softmax concept logits (`CS`) and a leave-one-out validation reference distribution;
- shortcut risk: a proxy for semantically unstable concepts despite a stable, confident task label. It is computed from concept/label (in)stability and task confidence while familiarity remains a separate evidence channel.

Review is the union of two independently triggered paths: a `shortcut_flag` (shortcut risk over a validation-selected threshold) and an `ood_flag` (familiarity at or below a validation-selected threshold). Both are reported alongside `review_flag` so a consumer can tell which evidence triggered review. The separation is structural, not a guarantee that the two flags identify distinct real-world causes.

The proposed novelty combines a **concept consistency probe** with a **meta-cognitive confidence layer**. The implementation includes member-preserving adapters for HalfMNIST and MiniKandinsky, validation-only threshold calibration, controlled interventions, and serialized per-sample diagnostics.

## Current status

- Phase I: complete — environment restoration, baseline reproduction, diagnostics, HalfMNIST/BEARS evaluation support, MiniKandinsky smoke testing, and a practical BDD-OIA reconstruction.
- Phase II HalfMNIST: frozen — the v4 leave-one-intervention-out study and v5 half-swap negative control are complete across seeds 0, 10, and 20.
- Phase II MiniKandinsky: frozen validation study — the v3 fusion candidate and v4 uncertainty ablation are immutable. Predictive entropy outperforms fusion on desaturation, so no universal OOD-superiority claim is made.
- Phase II BDD-OIA: baseline workflow ready — smoke validation, reusable ResNet50 preprocessing, resumable multi-seed training, and action/concept summary generation are available in the dedicated Colab notebook.

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

## Next-dataset Colab notebooks

Use separate notebooks because MiniKandinsky and BDD-OIA have different data
layouts, entry points, and persistence requirements:

- `colab/MetaBEARS_MiniKandinsky.ipynb` stages `kand-3k.zip`, trains or reuses
  checkpoints for seeds 0, 10, and 20, and runs the checkpoint ensemble through
  structural, semantic, and OOD MetaBEARS controls.
- `colab/MetaBEARS_BDD_OIA.ipynb` validates `lastframe.zip`, runs an isolated
  smoke workflow, optionally creates reusable ResNet50 features, and trains
  uniquely named multi-seed BDD variants without overwriting prior runs.

The MiniKandinsky adapter regroups the legacy model output into 18 categorical
concepts and selects the final binary task target. Its validation study is
frozen with an explicit negative ablation: task uncertainty detects palette
desaturation more effectively than the proposed fusion. The BDD notebook
remains a practical reconstruction based on ResNet50 ImageNet features; it is
not the exact paper setup that used the unavailable Faster-RCNN/CBM-AUC
feature archive.

For the first BDD-OIA run, keep only `RUN_SMOKE=True`. The notebook expects the
existing archive at `PES - Semester 4/bears_data/lastframe.zip`. Once smoke
preprocessing and one-epoch training pass, enable full preprocessing only if
the reusable Drive feature directory is absent. Full baseline runs persist a
completion record per condition and seed, so a disconnected runtime can reuse
finished runs.

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

Evaluate an existing MiniKandinsky ensemble:

```powershell
python colab_runner.py --job metabears_minikandinsky --minikand-checkpoints <seed-0.pt> <seed-10.pt> <seed-20.pt> --minikand-intervention figure_permute --minikand-ood-transform palette_desaturate
```

Run the MiniKandinsky v1 held-out-shift protocol:

```powershell
python colab_runner.py --job metabears_minikandinsky --minikand-checkpoints <seed-0.pt> <seed-10.pt> <seed-20.pt> --minikand-intervention figure_permute --minikand-representation-key pCS --minikand-representation-normalization zscore_l2 --minikand-ood-validation-transform palette_desaturate --minikand-ood-transform palette_pastel --minikand-shortcut-max-false-review-rate 0.05 --minikand-familiarity-max-false-review-rate 0.05
```

The Colab notebook trains a matched v1 pair under the corrected differentiable
MiniKandinsky task loss: `c_sup=1`, `w_c=10` and `c_sup=0`, `w_c=0`. All six
checkpoints use a `v1-task-loss` filename tag and separate Drive directories,
so they cannot overwrite the original v0 supervised baseline.

The next exploratory stage trains a separately tagged `c_sup=0` ensemble with
concept-balance entropy regularization. It then compares `CS` and `pCS` under
four validation-fitted normalizations using ID validation and desaturated OOD
validation only. A candidate is accepted only if it meets predeclared AUROC,
average-precision, recall, and false-review criteria. This sweep never iterates
the test loader; a new held-out OOD transform is frozen only after selection.

The v3 follow-up keeps the strongest supervised representation fixed at
`CS + zscore_l2`. It uses five-fold cross-fitting to compare nearest-reference,
shrinkage Mahalanobis, predicted-class-conditional Mahalanobis, and an
equal-weight class-conditional/disagreement fusion. The same validation gate
is retained, and no held-out test data is used during scorer selection.

After v3 acceptance, the candidate configuration and source-artifact hashes
are recorded in `minikandinsky_results_freeze_v3.json`. The v4 validation-only
ablation leaves that candidate unchanged and compares it with label
disagreement, predictive entropy, and confidence deficit as standard
uncertainty-only controls. Its negative result and artifact hashes are frozen
in `minikandinsky_results_freeze_v4.json`.

Run MetaBEARS with existing HalfMNIST ensemble checkpoints:

```powershell
python colab_runner.py --job metabears_halfmnist --metabears-checkpoints <member-1.pt> <member-2.pt> <member-3.pt>
```

Add the controlled commutative half-swap intervention without retraining:

```powershell
python colab_runner.py --job metabears_halfmnist --metabears-checkpoints <member-1.pt> <member-2.pt> <member-3.pt> --metabears-intervention half_swap --ece-bins 15
```

The intervention exchanges the two HalfMNIST digits, preserves the addition
label, and aligns the predicted concept positions back before measuring
semantic consistency. Because the shared encoder and symbolic addition rule
are permutation invariant, this is retained as a negative control.

Train a controlled shortcut ensemble with a task-correlated canonical-pair
patch, then contradict only that patch at evaluation time:

```powershell
python colab_runner.py --job metabears_halfmnist --metabears-train-ensemble --metabears-shortcut-patch-training --metabears-intervention patch_conflict --halfmnist-preset paper --n-ensembles 5 --seed 0
```

The patch encodes two pseudo-digits whose symbolic sum equals the task label.
The conflicting intervention encodes an incorrect sum while leaving the
visible digits and ground-truth label unchanged. Checkpoint names include a
`shortcut-patch-True` suffix so controlled models cannot overwrite ordinary
BEARS members. Behavioral proxies remain explicitly separated from claims
about the model's internal causal mechanism.

The same controlled checkpoints support two additional ablations without
retraining. `patch_removed` restores every reserved cue cell to background
zero, while `patch_shuffled` selects the cyclic batch rotation with the fewest
label matches. The latter preserves the empirical one-hot patch distribution
while breaking as much sample-level alignment as that batch permits.

Frozen Phase-II experiments use `experiment_protocol.json`. The matrix runner
rejects configuration drift, records Git/environment and SHA-256 provenance,
trains each ensemble once, reuses its checkpoints across controls, and writes
directly to a durable result root:

```powershell
python metabears_matrix.py --output-root <results-root> --seeds 10
```

To migrate the already-trained seed-0 members into the frozen layout without
retraining, provide their checkpoint directory:

```powershell
python metabears_matrix.py --output-root <results-root> --seeds 0 --checkpoint-source <seed-0-checkpoints>
```

After seeds 0, 10, and 20 are complete, validate and aggregate the matrix:

```powershell
python aggregate_metabears_results.py --results-root <results-root>
```

The aggregator writes per-run and mean/standard-deviation CSV files with
seed-level 95% Student-t confidence intervals, `aggregate_results.json`, and a
percentage-scaled `aggregate_metrics.png` with individual seed points when
Matplotlib is available. It also reports the accuracy drop normalized by the
effective shuffled-patch mismatch rate. OOD metrics are additionally written
to `model_results.csv` and `model_aggregate_results.csv`, deduplicated by
checkpoint fingerprint so controls that reuse one ensemble are not counted as
independent models. Incomplete or protocol-inconsistent matrices are rejected
by default.

When the saved prediction artifacts are present, the same command also runs a
held-out post-hoc comparison of task uncertainty, task entropy, task-label
disagreement, concept disagreement, intervention-only instability, and the
full MetaBEARS score. These are threshold-free evaluations; no threshold is
tuned on test labels. The additional outputs are:

- `paired_control_results.csv` and `paired_control_aggregate_results.csv` for
  within-seed primary-versus-control differences;
- `detector_results.csv` and `detector_aggregate_results.csv` for AUROC,
  average precision, area under the risk-coverage curve, risk at 80% automatic
  coverage, and review rate required for 95% failure recall;
- `detector_precision_recall_curves.csv` and
  `detector_risk_coverage_curves.csv` for the complete per-seed curves;
- `paired_control_effects.png`, `detector_average_precision.png`,
  `detector_curves.png`, and `ood_metrics.png` for seed-level figures; and
- `reporting_provenance.json`, which records the reporting Git commit,
  command, environment, and SHA-256 hashes of the analysis source files.

The validation-fitted fusion is separately frozen in
`analysis_protocol_v2.json`. For each ensemble seed, it fits non-negative
weights on the `patch_shuffled` validation intervention using stratified
out-of-fold average precision, freezes empirical percentile references and a
95%-validation-recall threshold, and then applies that unchanged model to both
ID-test controls. It adds task-distribution Jensen-Shannon change as a direct
invariance signal and never uses ID-test labels for fitting. Its model weights,
held-out threshold metrics, and paired baseline comparisons are written to
`fusion_v2_models.csv`, `fusion_v2_threshold_results.csv`,
`detector_paired_results.csv`, and
`detector_paired_aggregate_results.csv`.

The next analysis layer is frozen separately in `analysis_protocol_v3.json`.
It preserves the v2 weights and 95%-recall threshold learned only from the
labelled `patch_shuffled` validation split. For each known intervention, it
then rebuilds the empirical percentile references from that intervention's
validation predictions without loading task or concept targets. This tests
whether unlabeled score-scale calibration improves threshold transfer while
leaving every ID-test example and label untouched. It is a known-probe
calibration protocol, not a claim of generalization to an intervention for
which no validation predictions are available.

The default aggregator now reports v1, v2, and v3 together. It writes
`fusion_v3_models.csv`, `fusion_v3_reference_calibrations.csv`,
`fusion_v3_threshold_results.csv`, and
`fusion_threshold_aggregate_results.csv`. Paired detector tables include
v2-minus-v1, v3-minus-v1, and v3-minus-v2 comparisons automatically. Existing
v2 outputs and the frozen training protocol remain unchanged; rerunning model
training is not required.

Protocol v4 is defined in `analysis_protocol_v4.json`. It adds the frozen
`patch_conflict` and `patch_neutral` probes and performs strict
leave-one-intervention-out evaluation. For each seed and evaluated probe, the
fusion weights, percentile references, and 95%-recall threshold are fitted
only from the other three interventions' validation artifacts. The held-out
intervention's validation predictions and labels are never loaded, and only
its untouched ID-test artifacts are evaluated. Weight selection uses
intervention-blocked folds rather than mixing examples from the same probe
across random folds.

Existing checkpoints can generate the supplementary artifacts without model
training:

```powershell
python metabears_matrix.py --output-root <results-root> --seeds 0 10 20 --interventions patch_conflict patch_neutral --reuse-checkpoints --skip-completed
```

The runner preserves existing matrix-manifest entries when these controls are
added. The default aggregator then expects all four patch interventions and
writes `fusion_v4_models.csv` and `fusion_v4_threshold_results.csv` alongside
the unchanged v1-v3 outputs and paired comparisons.

The completed Phase-II evidence is frozen by `results_freeze_v4.json`. The
result archive remains outside Git, while its exact filename, byte size,
SHA-256 digest, protocol identities, run matrix, source hashes, and integrity
checks are versioned. Any later experimental change requires a new protocol
and result freeze; report-only tables and figures must trace back to the frozen
archive digest.

Protocol v5 is a post-freeze negative-control extension defined in
`analysis_protocol_v5.json`. It does not modify or replace the v4 evidence.
The already implemented `half_swap` transformation exchanges the two
HalfMNIST digits and then realigns their concept axes. Addition is commutative,
so this is a label-preserving structural control. Generate only the new control
artifacts from the existing checkpoints:

```powershell
python metabears_matrix.py --output-root <results-root> --seeds 0 10 20 --interventions half_swap --reuse-checkpoints --skip-completed
```

Then run the v5 analysis over the four patch interventions plus the control:

```powershell
python aggregate_metabears_results.py --results-root <results-root> --analysis-protocol analysis_protocol_v5.json --output-dir <results-root>/aggregate_v5
```

V5 fits its weights, empirical references, and threshold only on validation
artifacts from the four patch interventions. It never loads `half_swap`
validation data and evaluates the fixed detector only on the control's ID-test
artifacts. The added outputs are `fusion_v5_models.csv` and
`fusion_v5_threshold_results.csv`; the expected control behavior is low task
and aligned semantic failure prevalence, with the observed review rate reported
without post-hoc adjustment.

The completed negative-control extension is frozen by
`results_freeze_v5.json` and tagged `metabears-results-v5`. Its external result
archive is `aggregate_half_swap_v5_seed42.zip` with SHA-256
`8106b416fcb0266e8e759c01e0c9929be192f3ec644313c6ba44c5b4c7583bf8`.
Across three seeds and 1,260 paired ID-test sample evaluations, `half_swap` produced
zero task failures, zero aligned semantic-instability events, zero accuracy
drop, and zero v5 review flags. Because the control has no positive failures,
its AUROC and average precision are undefined rather than zero. This v5 freeze
is specificity evidence and does not replace the primary v4 performance
archive.

If a fresh runtime reserializes the HalfMNIST source artifact, its file-level
SHA-256 can differ from the frozen runtime. V5 does not blindly bypass that
check. It accepts the control only when labels, concepts, and batch boundaries
match exactly and the saved base concept probabilities, task probabilities,
and representations match the frozen `patch_shuffled` run within the fixed v5
tolerances on validation, ID-test, and OOD-test splits. The equivalence decision
and maximum observed difference are written to `aggregate_results.json`.

For a deliberately summary-only aggregation that does not have the `.npz`
prediction artifacts, add `--skip-detector-analysis`.

Or train a fresh BEARS ensemble before evaluation:

```powershell
python colab_runner.py --job metabears_halfmnist --metabears-train-ensemble --halfmnist-preset repo-best
```

The runner calibrates only on the ID validation split, then writes separate
validation, ID-test and OOD-test JSON/CSV reports, compressed member-level
predictions, held-out shortcut-proxy metrics, task/concept F1 and ECE,
selective-risk metrics, optional paired-intervention artifacts, and
`run_summary.json` under `colab_outputs/metabears_halfmnist/`.

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
