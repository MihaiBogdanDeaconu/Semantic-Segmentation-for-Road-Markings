import os
from PIL import Image
import random
from pathlib import Path

IMAGES_DIR = r"C:\Users\SUR3CLJ\Desktop\train_rgb\images"
TARGETS_DIR = FILTERED_IMAGES_DIR = (
    r"C:\Users\SUR3CLJ\Desktop\train_rgb\labels\background_restricted_crosswalk"
)

BACKGROUND_ONLY_PATH = r"C:\Users\SUR3CLJ\Desktop\train_rgb\labels\background_only"


def get_stage():
    x = random.random()
    if x < 0.75:
        return "train"
    else:
        return "eval"


def main():
    if not os.path.isdir(IMAGES_DIR) or not os.path.isdir(TARGETS_DIR):
        raise RuntimeError(
            "Dataset not found or incomplete. Please make sure all required folders for the"
            ' specified "split" and "mode" are inside the "root" directory'
        )

    sorted_names = sorted(os.listdir(FILTERED_IMAGES_DIR))

    images = [
        os.path.join(IMAGES_DIR, image_name)
        for image_name in sorted_names
        if image_name != "Thumbs.db"
    ]

    targets = [
        os.path.join(TARGETS_DIR, target_name)
        for target_name in sorted_names
        if target_name != "Thumbs.db"
    ]

    print(len(images))
    print(len(targets))

    for i in range(len(images)):
        stage = get_stage()
        image = Image.open(images[i])
        target = Image.open(targets[i])
        image_name = Path(images[i]).relative_to(IMAGES_DIR)

        if image.size == (1664, 640):
            image.save(
                f"C:\\Users\\SUR3CLJ\\Desktop\\painted_symbols_dataset\\images\\{stage}\\1664x640\\{image_name}"
            )
            target.save(
                f"C:\\Users\\SUR3CLJ\\Desktop\\painted_symbols_dataset\\labels\\{stage}\\1664x640\\{image_name}"
            )
        elif Image.open(images[i]).size == (1664, 512):
            image.save(
                f"C:\\Users\\SUR3CLJ\\Desktop\\painted_symbols_dataset\\images\\{stage}\\1664x512\\{image_name}"
            )
            target.save(
                f"C:\\Users\\SUR3CLJ\\Desktop\\painted_symbols_dataset\\labels\\{stage}\\1664x512\\{image_name}"
            )


if __name__ == "__main__":
    main()
