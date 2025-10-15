from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torchvision.transforms import v2
from PIL import Image
import h5py
import numpy as np
from collections import defaultdict

import os
from pathlib import Path
from typing import Union

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_processor = AutoImageProcessor.from_pretrained(
    "microsoft/resnet-18", use_fast=True
)
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
del model.classifier[1]  # remove the classification head
model.eval()
model = model.to(device)

full_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
full_model.eval()
full_model = full_model.to(device)


def extract_features(image_path, apply_transform=False):
    if not isinstance(image_path, list):
        image_path = [image_path]

    transforms = v2.Compose(
        [
            v2.RandomHorizontalFlip(),
            v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            v2.RandomRotation(10),
        ]
    )

    output_logits = []
    with torch.no_grad():
        for path in [
            image_path[: len(image_path) // 2],
            image_path[len(image_path) // 2 :],
        ]:
            images = [Image.open(img).convert("RGB") for img in path]

            if apply_transform:
                images = [transforms(img) for img in images]

            inputs = image_processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)

            output_logits.append(outputs.logits.cpu().numpy())
    return np.concatenate(output_logits, axis=0)


def gen_all_features(
    output="_animals_features.h5",
    ignore_split=False,  # Mix train and test and val splits
    equalize_classes: Union[int, None] = 1000,  # noqa # Crop or Data-augment classes to this size
    generate_validation_from_test=True,
):
    with h5py.File(output, "w") as features_file:
        if not ignore_split:
            if generate_validation_from_test:
                split_file = dict(
                    train=features_file.create_group("train"),
                    test=features_file.create_group("test"),
                    validation=features_file.create_group("validation"),
                )
            else:
                split_file = dict(
                    train=features_file.create_group("train"),
                    test=features_file.create_group("test"),
                )

        image_dir = "raw_bears_only"
        walk = os.walk(image_dir)

        class_ids = []
        class_labels = []
        split_processed = {}
        for folder, _, content in walk:
            if len(content) == 0:
                continue

            folder = Path(folder)

            class_label = folder.stem[2:]
            class_id = (
                len(class_ids)
                if class_label not in class_labels
                else class_labels.index(class_label)
            )
            split = folder.parts[1].lower()
            apply_equalize_classes = equalize_classes is not None and split == "train"
            print(f"[{split.upper()}] Processing {class_label} with id {class_id}")

            if apply_equalize_classes:
                if len(content) > equalize_classes:
                    content = content[:equalize_classes]
                else:
                    while len(content) < equalize_classes:
                        diff = equalize_classes - len(content)
                        content.extend(np.random.permutation(content)[:diff].tolist())

            paths = [folder / img for img in content]
            features = extract_features(
                paths,
                apply_transform=apply_equalize_classes,
            )

            if ignore_split:
                # Resize dataset if class_id already exists, but this split has not been processed yet
                if str(class_id) in features_file:
                    if (
                        class_id not in split_processed
                        or split not in split_processed[class_id]
                    ):
                        dset = features_file[str(class_id)]
                        dset.resize(
                            (dset.shape[0] + features.shape[0], features.shape[1])
                        )

                        if class_id not in split_processed:
                            split_processed[class_id] = [split]
                        else:
                            split_processed[class_id].append(split)
                else:
                    dset = features_file.create_dataset(
                        str(class_id),
                        features.shape,
                        compression="gzip",
                        maxshape=(None, features.shape[1]),
                    )

            else:
                dset = split_file[split].create_dataset(
                    str(class_id), features.shape, compression="gzip"
                )

            dset.attrs["label"] = class_label
            dset.attrs["paths"] = content
            dset[-len(features) :, :] = features

            if class_label not in class_labels:
                class_ids.append(class_id)
                class_labels.append(class_label)

        features_file.attrs["class_labels"] = class_labels
        features_file.attrs["class_ids"] = class_ids


def evaluate_model(image_dir="raw_bears_only", N=512):
    splits = ["Test", "Validation"]
    correct = defaultdict(int)
    total = defaultdict(int)

    def match_class_to_imagenet(custom_label, label2id):
        custom_label = custom_label.lower()
        for full_label in label2id:
            synonyms = [s.strip().lower() for s in full_label.split(",")]
            if custom_label in synonyms:
                return label2id[full_label]  # Return the class index
        return None

    def zero_out_weights_and_biases(model, N):
        classifier = model.classifier[-1]
        with torch.no_grad():
            classifier.weight[:, N:] = 0.0
            classifier.bias[:] = 0.0
        print(f"Zeroed out classifier weights from feature {N} onward and all biases.")

    zero_out_weights_and_biases(full_model, N)

    for split in splits:
        print(f"\nEvaluating on {split.upper()} set")
        for folder in Path(image_dir, split).iterdir():
            if not folder.is_dir():
                continue
            class_label = folder.stem[2:]
            image_paths = list(folder.glob("*"))

            images = [Image.open(p).convert("RGB") for p in image_paths]
            inputs = image_processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = full_model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

            # Get ground truth class index
            class_index = match_class_to_imagenet(class_label, model.config.label2id)
            if class_index is None:
                print(f"Label '{class_label}' not found in ImageNet, skipping.")
                continue

            # Count correct predictions
            num_correct = (preds.cpu().numpy() == class_index).sum()
            correct[split] += num_correct
            total[split] += len(images)
            print(f" - {class_label}: {num_correct}/{len(images)} correct")

        acc = correct[split] / total[split] if total[split] > 0 else 0
        print(f"\n{split.upper()} Accuracy: {acc:.4f}")


if __name__ == "__main__":
    gen_all_features()

    # Run evaluation
    evaluate_model(N=512)
    evaluate_model(N=256)
    evaluate_model(N=128)
    evaluate_model(N=64)
    evaluate_model(N=30)
    evaluate_model(N=16)
