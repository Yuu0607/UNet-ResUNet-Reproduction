"""2026task2: six-class facial expression classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification import Recipe, main

GOOGLE = "google/vit-base-patch16-224-in21k"
HARDLY = "HardlyHumans/Facial-expression-detection"
RECIPES = {
    "vit-imagenet": Recipe("hf_vit", 224, 8, lr=2e-5, wd=.01, crop=1.,
                           head_lr=1e-4, smoothing=.05, source=GOOGLE, batch=16),
    "vit-torchvision": Recipe("vit_b_16", 224, 8, lr=2e-5, crop=1., batch=16),
    "resnet18-scratch": Recipe("resnet18", 48, 40, lr=8e-4, crop=1.,
                               pretrained=False, schedule="onecycle", batch=128),
    "resnet18-pretrained": Recipe("resnet18", 224, 12, lr=1e-4, crop=1., batch=64),
    "efficientnet-b0": Recipe("efficientnet_b0", 224, 10, crop=1.),
    "vit-trpakov": Recipe("hf_vit", 224, 6, lr=2e-5, crop=1.,
                          source="trpakov/vit-face-expression", batch=16),
    "vit-hardlyhumans": Recipe("hf_vit", 224, 6, lr=2e-5, crop=1., source=HARDLY, batch=16),
    "vit-weighted": Recipe("hf_vit", 224, 5, lr=5e-6, wd=.01, crop=1.,
                           source=HARDLY, weighted=True, head_lr=2e-5, patience=2),
    "vit-weighted-low-lr": Recipe("hf_vit", 224, 4, lr=1e-6, wd=.01, crop=1.,
                                  source=HARDLY, weighted=True, head_lr=5e-6, patience=2),
}

if __name__ == "__main__":
    main(RECIPES, "expressions", (.15, "horizontal", "ID", "Emotion"))
