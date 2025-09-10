# app/core/fetcher.py
"""Dataset fetcher using Hugging Face Datasets (fast slice/stream) with on-disk cache."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Track:
    title: str
    artist: str
    genre: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    decade: Optional[str] = None
    language: Optional[str] = "English"
    tempo: Optional[str] = None  # "slow" | "mid" | "fast"
    features: Dict[str, float] = field(default_factory=dict)  # energy, valence, etc.


class SpotifyDatasetFetcher:
    """
    Loads a small sample from a public HF dataset and normalizes rows.

    Fast path order:
      1) Split slicing (train[:N]) — downloads only the first N rows
      2) Streaming head via islice — no local cache but minimal memory/IO
      3) Tiny offline fallback — guarantees UI still works without internet
    Also writes a compact JSON cache to disk for instant subsequent loads.
    """

    _REPO = "maharshipandya/spotify-tracks-dataset"
    _SPLIT = "train"

    # --------------------------
    # Public API
    # --------------------------
    def sample_tracks(self, n_samples: int = 120, seed: int = 42) -> list[Track]:
        cache_dir = Path(os.getenv("APP_CACHE_DIR", "data/cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"tracks_{n_samples}.json"

        # 0) Load from small JSON cache if present
        if cache_file.exists():
            try:
                raw_rows = json.loads(cache_file.read_text(encoding="utf-8"))
                return [self._normalize(r) for r in raw_rows]
            except Exception:
                # corrupt cache: ignore and rebuild
                pass

        raw_rows: List[Dict] = []

        # 1) Split slicing: train[:N]
        try:
            from datasets import load_dataset  # import here to avoid hard dep at import time

            ds = load_dataset(self._REPO, split=f"{self._SPLIT}[:{n_samples}]")
            raw_rows = list(ds)  # already small
        except Exception:
            # 2) Streaming head
            try:
                from datasets import load_dataset  # type: ignore

                ds_stream = load_dataset(self._REPO, split=self._SPLIT, streaming=True)
                raw_rows = list(islice(ds_stream, n_samples))
            except Exception:
                # 3) Tiny offline fallback
                raw_rows = self._fallback_raw()

        # Write cache (raw HF rows) for fast future loads
        try:
            cache_file.write_text(json.dumps(raw_rows), encoding="utf-8")
        except Exception:
            pass

        return [self._normalize(r) for r in raw_rows]

    # --------------------------
    # Normalization helpers
    # --------------------------
    @staticmethod
    def _tempo_bucket(bpm: Optional[float]) -> str:
        if bpm is None:
            return "mid"
        try:
            bpm_f = float(bpm)
        except Exception:
            return "mid"
        if bpm_f < 90:
            return "slow"
        if bpm_f <= 130:
            return "mid"
        return "fast"

    @staticmethod
    def _decade_from_date(iso_date: Optional[str]) -> Optional[str]:
        if not iso_date or len(iso_date) < 4 or not iso_date[:4].isdigit():
            return None
        year = int(iso_date[:4])
        return f"{(year // 10) * 10}s"

    @staticmethod
    def _tags_from_features(
        energy: Optional[float], danceability: Optional[float], valence: Optional[float]
    ) -> List[str]:
        tags: List[str] = []
        try:
            if energy is not None:
                e = float(energy)
                tags.append("energetic" if e >= 0.65 else ("mellow" if e <= 0.35 else "balanced"))
        except Exception:
            pass
        try:
            if danceability is not None:
                d = float(danceability)
                tags.append("danceable" if d >= 0.6 else "laidback")
        except Exception:
            pass
        try:
            if valence is not None:
                v = float(valence)
                tags.append("happy" if v >= 0.6 else ("somber" if v <= 0.35 else "neutral"))
        except Exception:
            pass
        return tags

    def _normalize(self, row: Dict) -> Track:
        title = (row.get("track_name") or "").strip() or "Unknown Title"
        artist = (row.get("artists") or "").strip() or "Unknown Artist"

        tempo = row.get("tempo")
        energy = row.get("energy")
        valence = row.get("valence")
        dance = row.get("danceability")

        # Build features dict with safe defaults
        features: Dict[str, float] = {
            "energy": float(energy) if self._is_num(energy) else 0.5,
            "valence": float(valence) if self._is_num(valence) else 0.5,
            "danceability": float(dance) if self._is_num(dance) else 0.5,
            "tempo_bpm": float(tempo) if self._is_num(tempo) else 110.0,
        }

        return Track(
            title=title,
            artist=artist,
            genre="unknown",  # dataset has no explicit genre column
            tags=self._tags_from_features(energy, dance, valence),
            decade=self._decade_from_date(row.get("release_date")),
            language="English",  # unknown in dataset
            tempo=self._tempo_bucket(tempo),
            features=features,
        )

    @staticmethod
    def _is_num(x: object) -> bool:
        try:
            float(x)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    # --------------------------
    # Offline fallback
    # --------------------------
    @staticmethod
    def _fallback_raw() -> List[Dict]:
        """Tiny built-in sample to keep the UI usable offline."""
        return [
            {
                "track_name": "Blinding Lights",
                "artists": "The Weeknd",
                "tempo": 171.0,
                "energy": 0.73,
                "valence": 0.33,
                "danceability": 0.51,
                "release_date": "2019-11-29",
            },
            {
                "track_name": "Yellow",
                "artists": "Coldplay",
                "tempo": 174.0,
                "energy": 0.57,
                "valence": 0.55,
                "danceability": 0.35,
                "release_date": "2000-06-26",
            },
            {
                "track_name": "Shape of You",
                "artists": "Ed Sheeran",
                "tempo": 96.0,
                "energy": 0.65,
                "valence": 0.60,
                "danceability": 0.82,
                "release_date": "2017-01-06",
            },
            {
                "track_name": "Hotel California",
                "artists": "Eagles",
                "tempo": 75.0,
                "energy": 0.55,
                "valence": 0.45,
                "danceability": 0.55,
                "release_date": "1976-12-08",
            },
            {
                "track_name": "Nandemonaiya",
                "artists": "Radwimps",
                "tempo": 140.0,
                "energy": 0.55,
                "valence": 0.48,
                "danceability": 0.45,
                "release_date": "2016-08-24",
            },
        ]
