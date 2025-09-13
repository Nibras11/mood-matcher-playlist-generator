"""Fetch and prepare Spotify dataset tracks"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import random
import pandas as pd
from datasets import load_dataset


@dataclass
class Track:
    """Music track with metadata and features"""
    title: str
    artist: str
    genre: Optional[str] = None
    decade: Optional[str] = None
    tempo: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


class SpotifyDatasetFetcher:
    """Fetch and process Spotify tracks dataset from Hugging Face"""

    def __init__(self, dataset_name: str = "maharshipandya/spotify-tracks-dataset"):
        self.dataset_name = dataset_name
        self._dataset = None

    def load_dataset(self):
        """Load dataset from Hugging Face Hub"""
        if self._dataset is None:
            try:
                self._dataset = load_dataset(self.dataset_name, split="train")
            except Exception as e:
                raise RuntimeError(f"Failed to fetch dataset: {e}")
        return self._dataset

    def sample_tracks(self, n_samples: int = 100, seed: int = 42) -> List[Track]:
        """Sample N random tracks from the dataset"""
        dataset = self.load_dataset()
        df = dataset.to_pandas()

        # Shuffle and sample
        df = df.sample(n=min(n_samples, len(df)), random_state=seed)

        tracks: List[Track] = []
        for _, row in df.iterrows():
            # Try to normalize columns (dataset may vary slightly)
            title = str(row.get("track_name", "")).strip()
            artist = str(row.get("track_artist", "")).strip()
            genre = str(row.get("playlist_genre", "")).strip() or None
            subgenre = str(row.get("playlist_subgenre", "")).strip() or None

            decade = None
            try:
                year = int(row.get("track_album_release_date", "0")[:4])
                decade = f"{(year // 10) * 10}s" if year > 0 else None
            except Exception:
                pass

            # Tags from playlist and genre info
            tags = []
            if genre:
                tags.append(genre)
            if subgenre:
                tags.append(subgenre)

            # Features (normalize floats)
            features = {
                "energy": float(row.get("energy", 0.5)),
                "valence": float(row.get("valence", 0.5)),
                "danceability": float(row.get("danceability", 0.5)),
                "tempo_val": float(row.get("tempo", 120.0)),
            }

            # Categorize tempo (slow/medium/fast)
            tempo_val = features["tempo_val"]
            tempo_cat = (
                "slow" if tempo_val < 90
                else "medium" if tempo_val < 140
                else "fast"
            )

            tracks.append(
                Track(
                    title=title or "Unknown Title",
                    artist=artist or "Unknown Artist",
                    genre=genre or subgenre or "unknown",
                    decade=decade,
                    tempo=tempo_cat,
                    tags=tags,
                    features=features,
                )
            )

        return tracks
