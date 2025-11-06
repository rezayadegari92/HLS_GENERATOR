from pathlib import Path
from hls.bitrate import choose_bitrate_ladder


def test_bitrate_preset_and_per_title(tmp_path):
    dummy = tmp_path / "x.mkv"
    dummy.write_bytes(b"\x00")
    preset = choose_bitrate_ladder(dummy, [1080, 720, 480], mode="preset")
    assert 1080 in preset and 720 in preset and 480 in preset
    pt = choose_bitrate_ladder(dummy, [1080, 720], mode="per-title")
    assert set(pt.keys()) == {1080, 720}
