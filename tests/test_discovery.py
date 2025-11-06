import os
from pathlib import Path
import shutil
import tempfile

from hls_generator_v2 import discover_films, parse_resolution_list


def test_discover_films_simple_structure():
    tmp = tempfile.mkdtemp()
    try:
        movies = Path(tmp) / "movies" / "Film1" / "1080"
        movies.mkdir(parents=True, exist_ok=True)
        sample = movies / "Film1.mkv"
        sample.write_bytes(b"\x00")
        film_map = discover_films(Path(tmp) / "movies", [1080, 720, 480])
        assert "Film1" in film_map
        assert 1080 in film_map["Film1"]
        assert film_map["Film1"][1080] == sample
    finally:
        shutil.rmtree(tmp)
