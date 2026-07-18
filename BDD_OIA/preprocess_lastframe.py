"""Prepare BDD-OIA files from the official lastframe.zip archive.

The BDD runner in this repository expects split pkl files plus per-sample
`.pt` tensors under data/bdd2048. This script converts the official
action/reason JSON labels into that layout and can optionally create 2048-d
ResNet-50 features for smoke experiments.
"""

import argparse
import json
import pickle
import shutil
import zipfile
from pathlib import Path

import torch
from PIL import Image


SPLITS = ("train", "val", "test")
ACTION_DIM = 5
CONCEPT_DIM = 21


def _read_json(archive, name):
    return json.loads(archive.read(name).decode("utf-8"))


def _action_records(actions):
    images = actions["images"]
    annotations = actions["annotations"]

    if len(images) != len(annotations):
        raise ValueError(
            "Expected the same number of images and action annotations, "
            f"got {len(images)} and {len(annotations)}."
        )

    records = {}
    for image, annotation in zip(images, annotations):
        file_name = image["file_name"]
        category = annotation["category"]
        if len(category) == ACTION_DIM - 1:
            category = category + [0]
        if len(category) != ACTION_DIM:
            raise ValueError(
                f"{file_name} has {len(category)} action labels; "
                f"expected {ACTION_DIM}."
            )
        records[file_name] = [float(value) for value in category]

    return records


def _reason_records(reasons):
    records = {}
    for item in reasons:
        file_name = item["file_name"]
        reason = item["reason"]
        if len(reason) != CONCEPT_DIM:
            raise ValueError(
                f"{file_name} has {len(reason)} reason labels; "
                f"expected {CONCEPT_DIM}."
            )
        records[file_name] = [float(value) for value in reason]

    return records


def _prepare_output_dir(path, force):
    if path.exists():
        has_content = any(path.iterdir())
        if has_content and not force:
            raise FileExistsError(
                f"{path} already exists and is not empty. Pass --force "
                "or choose a different --output directory."
            )
        if force:
            shutil.rmtree(path)

    for split in SPLITS:
        for subdir in ("inputs", "labels", "concepts"):
            (path / split / subdir).mkdir(parents=True, exist_ok=True)


def _build_resnet50_extractor(weights_name, device):
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import ResNet50_Weights, resnet50

    if weights_name == "imagenet":
        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
    elif weights_name == "none":
        weights = None
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        raise ValueError(
            "--feature-weights must be either 'imagenet' or 'none'"
        )

    model = resnet50(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model, transform


def _extract_features(archive, image_names, feature_dir, weights_name, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = _build_resnet50_extractor(weights_name, device)

    batch = []
    stems = []
    with torch.no_grad():
        for image_name in image_names:
            with archive.open(f"data/{image_name}") as image_file:
                image = Image.open(image_file).convert("RGB")
                batch.append(transform(image))
                stems.append(Path(image_name).stem)

            if len(batch) == batch_size:
                _save_feature_batch(model, batch, stems, feature_dir, device)
                batch, stems = [], []

        if batch:
            _save_feature_batch(model, batch, stems, feature_dir, device)


def _save_feature_batch(model, batch, stems, feature_dir, device):
    images = torch.stack(batch).to(device)
    features = model(images).flatten(1).cpu()
    for stem, feature in zip(stems, features):
        torch.save(feature.unsqueeze(0), feature_dir / f"{stem}.pt")


def _write_split(
    archive,
    split,
    output_dir,
    materialize_images,
    feature_mode,
    feature_weights,
    feature_batch_size,
    limit,
):
    actions = _read_json(archive, f"{split}_25k_images_actions.json")
    reasons = _read_json(archive, f"{split}_25k_images_reasons.json")

    actions_by_file = _action_records(actions)
    reasons_by_file = _reason_records(reasons)

    matched_image_names = [
        image["file_name"]
        for image in actions["images"]
        if image["file_name"] in reasons_by_file
    ]
    image_names = matched_image_names
    if limit is not None:
        image_names = image_names[:limit]

    split_dir = output_dir / split
    records = []
    for image_name in image_names:
        stem = Path(image_name).stem
        action = actions_by_file[image_name]
        reason = reasons_by_file[image_name]

        records.append(
            {
                "img_path": image_name,
                "class_label": action,
                "attribute_label": reason,
                "uncertain_attribute_label": reason,
            }
        )

        torch.save(
            torch.tensor(action, dtype=torch.float32).unsqueeze(0),
            split_dir / "labels" / f"{stem}.pt",
        )
        torch.save(
            torch.tensor(reason, dtype=torch.float32).unsqueeze(0),
            split_dir / "concepts" / f"{stem}.pt",
        )

        if materialize_images:
            with archive.open(f"data/{image_name}") as source:
                with open(split_dir / image_name, "wb") as target:
                    shutil.copyfileobj(source, target)

    with open(output_dir / f"{split}_BDD_OIA.pkl", "wb") as pkl_file:
        pickle.dump(records, pkl_file)

    if feature_mode == "resnet50":
        _extract_features(
            archive,
            image_names,
            split_dir / "inputs",
            feature_weights,
            feature_batch_size,
        )

    return {
        "split": split,
        "samples": len(records),
        "actions_json": len(actions_by_file),
        "reasons_json": len(reasons_by_file),
        "missing_reasons": len(actions_by_file)
        - len(matched_image_names),
    }


def _validate_output(output_dir, require_inputs):
    for split in SPLITS:
        pkl_path = output_dir / f"{split}_BDD_OIA.pkl"
        with open(pkl_path, "rb") as pkl_file:
            records = pickle.load(pkl_file)

        if not records:
            raise ValueError(f"{pkl_path} has no records.")

        first = records[0]
        stem = Path(first["img_path"]).stem
        label = torch.load(output_dir / split / "labels" / f"{stem}.pt")
        concept = torch.load(output_dir / split / "concepts" / f"{stem}.pt")

        if tuple(label.shape) != (1, ACTION_DIM):
            raise ValueError(f"Bad label shape for {split}: {label.shape}")
        if tuple(concept.shape) != (1, CONCEPT_DIM):
            raise ValueError(
                f"Bad concept shape for {split}: {concept.shape}"
            )

        input_path = output_dir / split / "inputs" / f"{stem}.pt"
        if require_inputs:
            feature = torch.load(input_path)
            if tuple(feature.shape) != (1, 2048):
                raise ValueError(
                    f"Bad input feature shape for {split}: {feature.shape}"
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert official BDD-OIA lastframe.zip to BEARS layout."
    )
    parser.add_argument(
        "--zip",
        default="../lastframe.zip",
        help="Path to official lastframe.zip.",
    )
    parser.add_argument(
        "--output",
        default="data/bdd2048",
        help="Output directory relative to BDD_OIA.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Optional sample limit per split for smoke tests.",
    )
    parser.add_argument(
        "--materialize-images",
        action="store_true",
        help="Also copy raw JPGs into each split directory.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["none", "resnet50"],
        default="none",
        help="How to create input .pt tensors.",
    )
    parser.add_argument(
        "--feature-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Weights for --feature-mode resnet50.",
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    zip_path = Path(args.zip)
    output_dir = Path(args.output)

    _prepare_output_dir(output_dir, args.force)

    summaries = []
    with zipfile.ZipFile(zip_path) as archive:
        for split in SPLITS:
            summary = _write_split(
                archive=archive,
                split=split,
                output_dir=output_dir,
                materialize_images=args.materialize_images,
                feature_mode=args.feature_mode,
                feature_weights=args.feature_weights,
                feature_batch_size=args.feature_batch_size,
                limit=args.limit_per_split,
            )
            summaries.append(summary)
            print(
                "{split}: {samples} samples "
                "({missing_reasons} missing reasons skipped)".format(
                    **summary
                )
            )

    _validate_output(output_dir, require_inputs=args.feature_mode != "none")
    with open(output_dir / "preprocess_summary.json", "w") as summary_file:
        json.dump(summaries, summary_file, indent=2)

    if args.feature_mode == "none":
        print(
            "Done. Labels/concepts/pkl files are ready, but inputs/*.pt "
            "were not created. main_bdd.py needs 2048-d inputs."
        )
    else:
        print("Done. Output is ready for main_bdd.py.")


if __name__ == "__main__":
    main()
