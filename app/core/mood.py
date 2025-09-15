"""Mood parsing and analysis for playlist matching"""

import re
from dataclasses import dataclass
from typing import List, Set


@dataclass
class MoodProfile:
    energy: str      # low / medium / high
    valence: str     # negative / neutral / positive
    tempo: str       # slow / medium / fast
    genres: List[str]
    keywords: List[str]
    raw_text: str


class MoodParser:
    """Parse user mood text into structured attributes"""

    ENERGY_MAP = {
        "high": ["energetic", "excited", "upbeat", "powerful"],
        "medium": ["calm", "balanced", "relaxed"],
        "low": ["tired", "sleepy", "chill", "mellow"]
    }

    VALENCE_MAP = {
        "positive": ["happy", "joyful", "bright", "uplifting"],
        "negative": ["sad", "dark", "gloomy", "angry"],
        "neutral": ["nostalgic", "bittersweet", "reflective"]
    }

    TEMPO_MAP = {
        "fast": ["fast", "quick", "danceable"],
        "medium": ["steady", "moderate"],
        "slow": ["slow", "ballad", "peaceful"]
    }

    GENRE_KEYWORDS = {
        "rock", "pop", "hip-hop", "rap", "jazz", "classical",
        "electronic", "country", "folk", "blues", "reggae",
        "metal", "punk", "indie", "r&b", "soul"
    }

    def parse(self, mood_text: str) -> MoodProfile:
        if not mood_text.strip():
            return MoodProfile("medium", "neutral", "medium", [], [], "")

        text = mood_text.lower()
        words = set(re.findall(r"\b[a-z]+\b", text))

        return MoodProfile(
            energy=self._detect(words, self.ENERGY_MAP, "medium"),
            valence=self._detect(words, self.VALENCE_MAP, "neutral"),
            tempo=self._detect(words, self.TEMPO_MAP, "medium"),
            genres=[g for g in words if g in self.GENRE_KEYWORDS],
            keywords=list(words),
            raw_text=mood_text.strip()
        )

    def _detect(self, words: Set[str], mapping, default: str) -> str:
        for k, kws in mapping.items():
            if any(w in words for w in kws):
                return k
        return default
