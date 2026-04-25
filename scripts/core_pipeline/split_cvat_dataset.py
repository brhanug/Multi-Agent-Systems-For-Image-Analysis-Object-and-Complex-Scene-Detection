import argparse
import random
import shutil
import zipfile
from pathlib import Path


def split_and_prepare_datasets(
    project_root: Path,
    expected_images: int = 2000,
    parts: int = 2,
    unlabelled_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    base_dir = project_root / "cvat_upload_temp" / "obj_train_data"
    output_base = project_root / "human_baseline_gold_v2"
    names_src = project_root / "cvat_upload_temp" / "obj.names"
    data_src = project_root / "cvat_upload_temp" / "obj.data"

    if not base_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {base_dir}")
    if not names_src.exists() or not data_src.exists():
        raise FileNotFoundError("Missing obj.names or obj.data in cvat_upload_temp")

    images = sorted(base_dir.glob("*.jpg"))
    if len(images) != expected_images:
        raise ValueError(
            f"Expected {expected_images} images, found {len(images)} in {base_dir}. "
            "Use --expected-images to override if intentional."
        )
    if parts < 1:
        raise ValueError("--parts must be >= 1")
    if not 0 <= unlabelled_ratio <= 1:
        raise ValueError("--unlabelled-ratio must be between 0 and 1")

    random.seed(seed)
    random.shuffle(images)

    images_per_part = len(images) // parts
    if images_per_part == 0:
        raise ValueError("Too many parts requested for number of images.")
    unlabelled_count = int(images_per_part * unlabelled_ratio)

    def prepare_part(part_images: list[Path], part_name: str) -> None:
        part_dir = output_base / f"cvat_temp_{part_name}"
        obj_train_dir = part_dir / "obj_train_data"
        obj_train_dir.mkdir(parents=True, exist_ok=True)

        unlabelled_images = set(random.sample(part_images, unlabelled_count))
        train_txt_lines: list[str] = []

        for img_path in part_images:
            base_name = img_path.name
            txt_name = f"{img_path.stem}.txt"
            orig_txt_path = base_dir / txt_name

            dest_img_path = obj_train_dir / base_name
            dest_txt_path = obj_train_dir / txt_name

            shutil.copy2(img_path, dest_img_path)
            train_txt_lines.append(f"obj_train_data/{base_name}")

            if img_path in unlabelled_images:
                dest_txt_path.write_text("", encoding="utf-8")
            else:
                if not orig_txt_path.exists():
                    raise FileNotFoundError(f"Missing label for {base_name}: {orig_txt_path}")
                shutil.copy2(orig_txt_path, dest_txt_path)

        shutil.copy2(names_src, part_dir / "obj.names")
        shutil.copy2(data_src, part_dir / "obj.data")
        (part_dir / "train.txt").write_text("\n".join(train_txt_lines) + "\n", encoding="utf-8")

        zip_path = output_base / f"cvat_human_baseline_{part_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in part_dir.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(part_dir))

        print(f"Created {zip_path} with {len(part_images)} images ({unlabelled_count} unlabelled)")

    for idx in range(parts):
        start = idx * images_per_part
        end = (idx + 1) * images_per_part if idx < parts - 1 else len(images)
        prepare_part(images[start:end], f"part{idx + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split CVAT dataset into baseline packages.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root containing cvat_upload_temp and human_baseline_gold_v2",
    )
    parser.add_argument("--expected-images", type=int, default=2000)
    parser.add_argument("--parts", type=int, default=2)
    parser.add_argument("--unlabelled-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_and_prepare_datasets(
        project_root=Path(args.project_root).resolve(),
        expected_images=args.expected_images,
        parts=args.parts,
        unlabelled_ratio=args.unlabelled_ratio,
        seed=args.seed,
    )

