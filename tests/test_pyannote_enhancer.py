from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyannote_enhancer import dominant_speaker, gate_speaker


class PyannoteEnhancerTest(unittest.TestCase):
    def test_dominant_speaker_uses_exclusive_activity_duration(self) -> None:
        rows = [
            {"start": 0.0, "end": 0.5, "speaker": "SECONDARY"},
            {"start": 0.5, "end": 2.0, "speaker": "MAIN"},
            {"start": 2.0, "end": 2.4, "speaker": "MAIN"},
        ]
        selected, totals = dominant_speaker(rows)
        self.assertEqual("MAIN", selected)
        self.assertAlmostEqual(1.9, totals["MAIN"])

    def test_gating_keeps_timestamps_and_removes_other_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            output = root / "output.wav"
            samples = np.full(2000, 8192, dtype="<i2")
            with wave.open(str(source), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(1000)
                stream.writeframes(samples.tobytes())
            rows = [
                {"start": 0.0, "end": 0.8, "speaker": "OTHER"},
                {"start": 0.8, "end": 2.0, "speaker": "MAIN"},
            ]
            gate_speaker(source, output, rows, "MAIN", pad_ms=0.0, fade_ms=0.0)
            with wave.open(str(output), "rb") as stream:
                rendered = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
                self.assertEqual(2000, len(rendered))
            self.assertTrue(np.all(rendered[:800] == 0))
            self.assertTrue(np.all(rendered[800:] != 0))


if __name__ == "__main__":
    unittest.main()
