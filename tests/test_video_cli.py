import cv2
import numpy as np
import pandas as pd
import pytest

from nvf import cli, pipeline, synthetic, video


def write_video(path, n_frames=50, fps=10.0, size=128):
    """Grey frames whose brightness encodes the frame index (index * 5)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.full((size, size, 3), i * 5, np.uint8)
        frame[size // 4 : size // 2, size // 4 : size // 2] = 255 - i * 5
        writer.write(frame)
    writer.release()


@pytest.fixture
def video_file(tmp_path):
    path = tmp_path / "clip.avi"
    write_video(path)
    return path


def test_sample_frames_every_n_seconds(video_file):
    frames = list(video.sample_frames(video_file, every_s=1.0))
    assert [i for i, _, _ in frames] == [0, 10, 20, 30, 40]
    assert [t for _, t, _ in frames] == pytest.approx([0, 1, 2, 3, 4])
    # Brightness of the background identifies the frame (MJPG is lossy).
    for i, _, rgb in frames:
        assert abs(int(rgb[2, 2, 0]) - i * 5) <= 3


def test_fractional_interval(video_file):
    frames = list(video.sample_frames(video_file, every_s=0.25))
    assert [i for i, _, _ in frames][:5] == [0, 2, 5, 8, 10]


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        list(video.sample_frames(tmp_path / "missing.avi", 1.0))
    with pytest.raises(ValueError):
        list(video.sample_frames(tmp_path / "missing.avi", 0))


def test_video_features_and_summary(video_file):
    frames = video.video_features(video_file, every_s=2.0)
    assert len(frames) == 3
    summary = video.summarise(frames).set_index("feature")
    assert "frame_index" not in summary.index and "fd_otsu" in summary.index
    assert summary.loc["mean_value", "mean"] == pytest.approx(frames["mean_value"].mean())
    assert summary.loc["mean_value", "sd"] == pytest.approx(frames["mean_value"].std(ddof=1))
    assert summary.loc["mean_value", "n_frames"] == 3


def test_extract_features_downscales_only():
    img = (synthetic.power_law_noise(256, 1.0) * 255).astype(np.uint8)
    feats = pipeline.extract_features(img, max_side=128)
    assert feats["width"] == 128 and feats["scale"] == 0.5
    feats = pipeline.extract_features(img, max_side=1024)
    assert feats["width"] == 256 and feats["scale"] == 1.0


def test_extract_features_with_mocked_segmenter():
    class Sky:
        def predict(self, rgb):
            return np.full(rgb.shape[:2], 2)

    feats = pipeline.extract_features(np.zeros((64, 64, 3), np.uint8), segmenter=Sky())
    assert feats["sky_view_share"] == 1.0


def test_cli_images(tmp_path):
    img_dir = tmp_path / "imgs"
    (img_dir / "sub").mkdir(parents=True)
    tri = synthetic.sierpinski_triangle(256).astype(np.uint8) * 255
    cv2.imwrite(str(img_dir / "tri.png"), tri)
    cv2.imwrite(str(img_dir / "sub" / "noise.png"),
                (synthetic.power_law_noise(256, 1.5) * 255).astype(np.uint8))
    (img_dir / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "features.csv"
    assert cli.main(["images", str(img_dir), "--out", str(out)]) == 0
    df = pd.read_csv(out).set_index("image")
    assert list(df.index) == ["sub/noise.png", "tri.png"]
    assert abs(df.loc["tri.png", "fd_otsu"] - synthetic.KNOWN_D["sierpinski_triangle"]) < 0.02


def test_cli_video(tmp_path, video_file):
    out = tmp_path / "frames.csv"
    assert cli.main(["video", str(video_file), "--every", "1", "--out", str(out)]) == 0
    assert len(pd.read_csv(out)) == 5
    assert (tmp_path / "frames_summary.csv").exists()


def test_cli_empty_directory(tmp_path):
    assert cli.main(["images", str(tmp_path), "--out", str(tmp_path / "f.csv")]) == 1
