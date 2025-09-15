"""Spotify dataset loader (used mainly for index building)"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datasets import load_dataset


@dataclass
class Track:
    title: str
    artist: str
    genre: Optional[str] = None
    decade: Optional[str] = None
    tempo: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)


class SpotifyDatasetFetcher:
    """Fetch dataset from Hugging Face Hub"""

    def __init__(self, dataset_name: str = "maharshipandya/spotify-tracks-dataset"):
        self.dataset_name = dataset_name
        self._dataset = None

    def load_tracks(self, n_samples: Optional[int] = None, seed: int = 42) -> List[Track]:
        if self._dataset is None:
            self._dataset = load_dataset(self.dataset_name, split="train")

        df = self._dataset.to_pandas()
        if n_samples:
            df = df.sample(n=n_samples, random_state=seed)

        tracks: List[Track] = []
        for _, row in df.iterrows():
            # Basic cleanup
            title = str(row.get("track_name", "")).strip() or "Unknown"
            artist = str(row.get("track_artist", "")).strip() or "Unknown"
            genre = str(row.get("playlist_genre", "")).strip() or None
            subgenre = str(row.get("playlist_subgenre", "")).strip() or None

            # decade from release year
            decade = None
            try:
                year = int(str(row.get("track_album_release_date", ""))[:4])
                if year > 0:
                    decade = f"{(year // 10) * 10}s"
            except Exception:
                pass

            features = {
                "energy": float(row.get("energy", 0.5)),
                "valence": float(row.get("valence", 0.5)),
                "danceability": float(row.get("danceability", 0.5)),
                "tempo_val": float(row.get("tempo", 120.0)),
            }

            tempo = (
                "slow" if features["tempo_val"] < 90 else
                "medium" if features["tempo_val"] < 140 else
                "fast"
            )

            tracks.append(
                Track(
                    title=title,
                    artist=artist,
                    genre=genre or subgenre or "unknown",
                    decade=decade,
                    tempo=tempo,
                    tags=[g for g in [genre, subgenre] if g],
                    features=features,
                )
            )

        return tracks
