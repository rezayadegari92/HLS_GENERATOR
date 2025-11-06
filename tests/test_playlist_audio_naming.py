from pathlib import Path
from typing import List

from hls_generator_v2 import extract_audio_tracks_to_hls, write_master_playlist, AudioTrack


class DummyTrack:
    def __init__(self, idx: int, lang: str, title: str, codec: str, channels: int, sample_rate: int):
        self.index = idx
        self.language = lang
        self.title = title
        self.codec = codec
        self.channels = channels
        self.sample_rate = sample_rate


def test_audio_naming_monkeypatch(tmp_path, monkeypatch):
    # Monkeypatch get_audio_tracks to provide deterministic tracks
    def fake_get_audio_tracks(_):
        return [
            DummyTrack(0, "en", "English", "aac", 2, 48000),
            DummyTrack(1, "fa", "Persian", "aac", 2, 48000),
        ]

    from hls_generator_v2 import get_audio_tracks
    monkeypatch.setattr("hls_generator_v2.get_audio_tracks", fake_get_audio_tracks)

    film_dir = tmp_path / "FilmX"
    film_dir.mkdir(parents=True, exist_ok=True)

    # Call extract to generate paths (skip real ffmpeg by not executing)
    # We assert naming scheme only by inspecting returned metadata
    outputs = extract_audio_tracks_to_hls(Path("/dev/null"), film_dir, 6, overwrite=False)

    # Expect deterministic folders like audio/en-00/index.m3u8 and audio/fa-01/index.m3u8
    rels = [str(p) for (_, _, p) in outputs]
    assert any("audio/en-00" in s for s in rels)
    assert any("audio/fa-01" in s for s in rels)

    # Master playlist should include EXT-X-MEDIA with LANGUAGE codes
    master = write_master_playlist(film_dir, [(1080, Path("1080/index.m3u8"))], audio_tracks=outputs)
    txt = master.read_text()
    assert "TYPE=AUDIO" in txt
    assert "LANGUAGE=\"en\"" in txt
    assert "LANGUAGE=\"fa\"" in txt
