import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from openemotion.config import CLASS_NAMES, IMG_SIZE
from openemotion.model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        f"{args.data}/test",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    model = load_model(
        args.model,
        device,
        len(CLASS_NAMES),
    )

    labels = []
    predictions = []

    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(device))
            batch_predictions = outputs.argmax(1).cpu()

            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.tolist())

    print(classification_report(
        labels,
        predictions,
        target_names=dataset.classes,
        digits=4,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions))


if __name__ == "__main__":
    main()
