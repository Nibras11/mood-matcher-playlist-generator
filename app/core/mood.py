"""Mood parsing and analysis for playlist matching"""

import re
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class MoodProfile:
    """Parsed mood profile with extracted attributes"""
    energy: str  # low, medium, high
    valence: str  # negative, neutral, positive
    tempo: str  # slow, medium, fast
    genres: List[str]
    keywords: List[str]
    raw_text: str


class MoodParser:
    """Parse mood text into structured attributes"""

    # Energy level keywords
    ENERGY_MAP = {
        "high": ["energetic", "pumped", "excited", "hyper", "intense", "powerful", "upbeat"],
        "medium": ["moderate", "balanced", "steady", "calm", "relaxed"],
        "low": ["tired", "sleepy", "mellow", "peaceful", "quiet", "soft", "gentle"]
    }

    # Emotional valence keywords
    VALENCE_MAP = {
        "positive": ["happy", "joyful", "cheerful", "uplifting", "bright", "optimistic", "fun"],
        "negative": ["sad", "depressed", "melancholy", "dark", "gloomy", "angry", "frustrated"],
        "neutral": ["contemplative", "thoughtful", "reflective", "nostalgic", "bittersweet"]
    }

    # Tempo keywords
    TEMPO_MAP = {
        "fast": ["fast", "quick", "rapid", "danceable", "upbeat", "energetic"],
        "medium": ["moderate", "steady", "walking", "medium"],
        "slow": ["slow", "ballad", "chill", "ambient", "peaceful", "relaxed"]
    }

    # Common genre keywords
    GENRE_KEYWORDS = {
        "rock", "pop", "hip-hop", "rap", "jazz", "classical", "electronic",
        "country", "folk", "blues", "reggae", "metal", "punk", "indie",
        "alternative", "r&b", "soul", "funk", "disco", "house", "techno"
    }

    def parse(self, mood_text: str) -> MoodProfile:
        """Parse mood text into structured profile"""
        if not mood_text or not mood_text.strip():
            return MoodProfile(
                energy="medium",
                valence="neutral",
                tempo="medium",
                genres=[],
                keywords=[],
                raw_text=""
            )

        text_lower = mood_text.lower()
        words = self._extract_words(text_lower)

        return MoodProfile(
            energy=self._detect_energy(words),
            valence=self._detect_valence(words),
            tempo=self._detect_tempo(words),
            genres=self._detect_genres(words),
            keywords=list(words),
            raw_text=mood_text.strip()
        )

    def _extract_words(self, text: str) -> Set[str]:
        """Extract meaningful words from text"""
        # Remove punctuation and split
        words = re.findall(r'\b[a-z]+\b', text)
        # Filter out common stop words
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "through", "during", "before", "after", "above",
            "below", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "want", "like", "feel", "feeling"
        }
        return {word for word in words if len(word) > 2 and word not in stop_words}

    def _detect_energy(self, words: Set[str]) -> str:
        """Detect energy level from words"""
        return self._find_best_match(words, self.ENERGY_MAP, default="medium")

    def _detect_valence(self, words: Set[str]) -> str:
        """Detect emotional valence from words"""
        return self._find_best_match(words, self.VALENCE_MAP, default="neutral")

    def _detect_tempo(self, words: Set[str]) -> str:
        """Detect tempo preference from words"""
        return self._find_best_match(words, self.TEMPO_MAP, default="medium")

    def _detect_genres(self, words: Set[str]) -> List[str]:
        """Detect mentioned genres"""
        return list(words.intersection(self.GENRE_KEYWORDS))

    def _find_best_match(self, words: Set[str], keyword_map: Dict[str, List[str]],
                         default: str) -> str:
        """Find best matching category based on keyword overlap"""
        scores = {}
        for category, keywords in keyword_map.items():
            scores[category] = len(words.intersection(set(keywords)))

        if not any(scores.values()):
            return default

        return max(scores, key=scores.get)