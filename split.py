import os
import shutil
from sklearn.model_selection import train_test_split


def create_symlinks(src_dir, dst_dir, files):
    for file in files:
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)
        os.symlink(src_path, dst_path)


def split_dataset(root_dir, save_dir, train_size=0.8):
    classes = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        images = os.listdir(cls_dir)

        train_imgs, val_imgs = train_test_split(
            images, train_size=train_size, random_state=42
        )

        train_cls_dir = os.path.join(save_dir, "train", cls)
        val_cls_dir = os.path.join(save_dir, "val", cls)

        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)

        create_symlinks(cls_dir, train_cls_dir, train_imgs)
        create_symlinks(cls_dir, val_cls_dir, val_imgs)


if __name__ == "__main__":
    dataset_root = "/media/hdd3/neo/topviews_1k"  # Replace with your dataset path
    save_directory = (
        "/media/hdd3/neo/topviews_1k_split"  # Replace with your save directory path
    )

    os.makedirs(save_directory, exist_ok=True)
    split_dataset(dataset_root, save_directory)
