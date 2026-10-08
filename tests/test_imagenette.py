"""Imagenette loading (data/imagenette.py): the 90/10 stratified split and the three splits."""

import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from edgeinf.data.imagenette import (
    CLASS_NAMES,
    DIRNAME,
    EVAL_TRANSFORM,
    INPUT_SIZE,
    RESIZE,
    WNIDS,
    export_frames,
    load_frames,
    load_splits,
    normalise,
    stratified_split,
)

EXPORT_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "export_frames.py"
SEED = 0
OTHER_SEED = 1
VAL_FRACTION = 0.1
TRAIN_PER_CLASS = 10
TEST_PER_CLASS = 2
IMAGE_SIDE = 320  # px, like the 320 px version of the dataset


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A tiny copy of the dataset's folder layout: <root>/<DIRNAME>/{train,val}/<wnid>/*.JPEG."""
    root = tmp_path_factory.mktemp("data")
    generator = torch.Generator().manual_seed(SEED)
    for split, per_class in (("train", TRAIN_PER_CLASS), ("val", TEST_PER_CLASS)):
        for wnid in WNIDS:
            folder = root / DIRNAME / split / wnid
            folder.mkdir(parents=True)
            for i in range(per_class):
                pixels = torch.randint(0, 256, (IMAGE_SIDE, IMAGE_SIDE, 3), generator=generator)
                Image.fromarray(pixels.to(torch.uint8).numpy()).save(folder / f"{wnid}_{i}.JPEG")
    return root


def test_split_takes_the_same_fraction_of_every_class():
    labels = [c for c in range(10) for _ in range(50 + 10 * c)]  # classes of different sizes
    train, val = stratified_split(labels, VAL_FRACTION, SEED)
    assert sorted(train + val) == list(range(len(labels)))
    assert not set(train) & set(val)
    per_class = Counter(labels)
    val_per_class = Counter(labels[i] for i in val)
    assert val_per_class == {c: round(VAL_FRACTION * n) for c, n in per_class.items()}


def test_split_is_fixed_by_the_seed():
    labels = [c for c in range(10) for _ in range(100)]
    assert stratified_split(labels, VAL_FRACTION, SEED) == stratified_split(
        labels, VAL_FRACTION, SEED
    )
    assert stratified_split(labels, VAL_FRACTION, SEED) != stratified_split(
        labels, VAL_FRACTION, OTHER_SEED
    )


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1])
def test_split_rejects_fractions_that_leave_a_split_empty(val_fraction):
    with pytest.raises(ValueError):
        stratified_split([0, 0, 1, 1], val_fraction, SEED)


def test_ten_classes_in_folder_order():
    assert len(CLASS_NAMES) == len(WNIDS) == 10
    assert list(WNIDS) == sorted(WNIDS)


def test_splits_have_the_expected_sizes_and_do_not_overlap(root):
    splits = load_splits(root, SEED, VAL_FRACTION)
    n_train_dir = TRAIN_PER_CLASS * len(WNIDS)
    assert len(splits.val) == round(VAL_FRACTION * TRAIN_PER_CLASS) * len(WNIDS)
    assert len(splits.train) == n_train_dir - len(splits.val)
    assert len(splits.test) == TEST_PER_CLASS * len(WNIDS)

    train, val, test = (set(splits.files(name)) for name in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert all(Path(f).parts[-3] == "train" for f in train | val)
    assert all(Path(f).parts[-3] == "val" for f in test)


def test_val_and_test_are_stratified_and_labels_follow_the_folders(root):
    splits = load_splits(root, SEED, VAL_FRACTION)
    labels = Counter(label for _, label in (splits.val[i] for i in range(len(splits.val))))
    assert set(labels.values()) == {round(VAL_FRACTION * TRAIN_PER_CLASS)}
    for index in (0, len(splits.test) - 1):
        _, label = splits.test[index]
        assert Path(splits.files("test")[index]).parent.name == WNIDS[label]


def test_images_come_out_as_normalised_224_tensors(root):
    splits = load_splits(root, SEED, VAL_FRACTION)
    for dataset in (splits.train, splits.val, splits.test):
        image, label = dataset[0]
        assert image.shape == (3, INPUT_SIZE, INPUT_SIZE)
        assert image.dtype == torch.float32
        assert 0 <= label < len(WNIDS)


def test_only_training_images_are_augmented(root):
    splits = load_splits(root, SEED, VAL_FRACTION)
    assert torch.equal(splits.val[0][0], splits.val[0][0])
    assert torch.equal(splits.test[0][0], splits.test[0][0])
    torch.manual_seed(SEED)
    first = splits.train[0][0]
    torch.manual_seed(OTHER_SEED)
    assert not torch.equal(first, splits.train[0][0])


def test_normalising_an_8_bit_frame_reproduces_the_dataset_transform(root):
    # A device holds the resized and cropped frame as 8-bit pixels and normalises it itself.
    # That must give exactly what the models saw in training and evaluation.
    image = Image.open(load_splits(root, SEED, VAL_FRACTION).files("test")[0]).convert("RGB")
    crop = transforms.Compose([transforms.Resize(RESIZE), transforms.CenterCrop(INPUT_SIZE)])
    frame = transforms.functional.pil_to_tensor(crop(image)).unsqueeze(0)
    assert frame.dtype == torch.uint8
    assert torch.allclose(normalise(frame)[0], EVAL_TRANSFORM(image), atol=1e-6)


def test_missing_dataset_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        load_splits(tmp_path, SEED, VAL_FRACTION)


def test_labels_are_listed_in_dataset_order(root):
    splits = load_splits(root, SEED, VAL_FRACTION)
    for name, dataset in (("val", splits.val), ("test", splits.test)):
        assert splits.labels(name) == [dataset[i][1] for i in range(len(dataset))]
        assert len(splits.labels(name)) == len(splits.files(name))
    assert len(splits.labels("train")) == len(splits.train)


@pytest.mark.parametrize("split", ["val", "test"])
def test_exported_frames_are_the_dataset_images_before_normalisation(root, tmp_path, split):
    splits = load_splits(root, SEED, VAL_FRACTION)
    dataset = getattr(splits, split)

    export_frames(splits, split, tmp_path / "frames")
    frames, labels = load_frames(tmp_path / "frames", split)

    assert frames.shape == (len(dataset), 3, INPUT_SIZE, INPUT_SIZE)
    assert frames.dtype == np.uint8
    assert isinstance(frames, np.memmap)  # the test split does not fit in a device's memory
    assert labels.tolist() == splits.labels(split)
    for index in (0, len(dataset) // 2, len(dataset) - 1):
        frame = torch.from_numpy(frames[index : index + 1].copy())
        image, label = dataset[index]
        assert torch.allclose(normalise(frame)[0], image, atol=1e-6)
        assert labels[index] == label


def test_training_images_are_not_exported(root, tmp_path):
    # They are cropped at random on every read, so there is no fixed frame to write.
    with pytest.raises(ValueError, match="train"):
        export_frames(load_splits(root, SEED, VAL_FRACTION), "train", tmp_path)


def test_missing_frames_give_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_frames"):
        load_frames(tmp_path, "test")


@pytest.mark.parametrize(("arguments", "written"), [([], ["test", "val"]), (["val"], ["val"])])
def test_the_export_script_writes_the_splits_of_a_training_config(
    root, tmp_path, monkeypatch, arguments, written
):
    spec = importlib.util.spec_from_file_location("export_frames", EXPORT_SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    # The dataset is copied in name only: the frames must land next to it, under frames/.
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / DIRNAME).symlink_to(root / DIRNAME)
    config = tmp_path / "train.yaml"
    config.write_text(f"data_root: {data_root}\nseed: {SEED}\nval_fraction: {VAL_FRACTION}\n")
    monkeypatch.setattr("sys.argv", ["export_frames.py", str(config), *arguments])

    script.main()

    files = sorted(path.name for path in (data_root / "frames").iterdir())
    assert files == sorted(
        f"{split}_{kind}.npy" for split in written for kind in ("frames", "labels")
    )
    splits = load_splits(root, SEED, VAL_FRACTION)
    for split in written:
        frames, labels = load_frames(data_root / "frames", split)
        assert len(frames) == len(splits.files(split)) and labels.tolist() == splits.labels(split)
