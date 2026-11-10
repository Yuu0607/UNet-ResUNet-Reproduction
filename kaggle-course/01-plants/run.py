"""2026task1: five-class plant seedling classification."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification import Recipe, main

RECIPES = {
    "b0-224": Recipe("efficientnet_b0", 224, 20, crop=.75),
    "b0-224-light": Recipe("efficientnet_b0", 224, 20, crop=.90),
    "b0-224-no-crop": Recipe("efficientnet_b0", 224, 20, crop=1.),
    "b0-224-smoothing": Recipe("efficientnet_b0", 224, 20, crop=.90, smoothing=.05),
    "b0-288-light": Recipe("efficientnet_b0", 288, 20, crop=.90),
    "b0-320-light": Recipe("efficientnet_b0", 320, 20, crop=.90),
    "b2-224-light": Recipe("efficientnet_b2", 224, 20, crop=.90),
    "b2-320-light": Recipe("efficientnet_b2", 320, 20, crop=.90),
    "b3-320-light": Recipe("efficientnet_b3", 320, 20, crop=.90),
    "convnext-tiny": Recipe("convnext_tiny", 224, 20, crop=.90),
    "efficientnetv2-s": Recipe("efficientnet_v2_s", 224, 20, crop=.90),
}
for bound in (.75, .85, .90, .95):
    RECIPES[f"crop-{bound:.2f}"] = Recipe("efficientnet_b0", 320, 20, crop=bound)

if __name__ == "__main__":
    main(RECIPES, "plants", (.20, "four", "ID", "Category"))
