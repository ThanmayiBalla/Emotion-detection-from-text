# keyword_rules.py
import re
from typing import Dict, List
from processing import EMOTIONS

KEYWORD_RULES: Dict[str, List[str]] = {
    "Confused": ["don't understand", "confused", "lost", "difficult", "unclear", "puzzled", "stuck"],
    "Frustrated": ["hate", "impossible", "annoying", "tired", "giving up", "infuriated", "broken", "useless"],
    "Curious": ["interested", "explore", "why", "how", "excited", "curious", "wonder", "fascinated"],
    "Confident": ["solved", "easy", "understand", "completed", "got it", "simple", "clear now", "perfect"],
    "Bored": ["boring", "sleepy", "repetitive", "not interested", "dry", "bored", "zoning out"]
}

def calculate_keyword_boost(text: str) -> Dict[str, float]:
    """
    Calculates absolute additive keyword score offsets for each emotion.
    Returns scores matching EMOTIONS index positions.
    """
    text_lower = text.lower()
    boosts = {emotion: 0.0 for emotion in EMOTIONS}
    
    for emotion, keywords in KEYWORD_RULES.items():
        match_count = 0
        for word in keywords:
            # Use word boundaries to prevent substring overlaps (e.g., "how" in "show")
            pattern = r'\b' + re.escape(word) + r'\b'
            matches = re.findall(pattern, text_lower)
            match_count += len(matches)
        
        if match_count > 0:
            # Apply weighted scaling: +0.15 boost per matched term
            boosts[emotion] = min(0.40, match_count * 0.15)
            
    return boosts

def apply_hybrid_correction(probs: Dict[str, float], text: str) -> Dict[str, float]:
    """Merges raw ML output probabilities with rule-based keyword boosts."""
    boosts = calculate_keyword_boost(text)
    
    # Add boosts
    corrected_probs = {}
    for emotion in EMOTIONS:
        corrected_probs[emotion] = probs.get(emotion, 0.0) + boosts.get(emotion, 0.0)
        
    # Re-normalize back to a valid probability distribution (sums to 1.0)
    total = sum(corrected_probs.values())
    if total > 0:
        corrected_probs = {k: v / total for k, v in corrected_probs.items()}
    else:
        corrected_probs = {k: 1.0 / len(EMOTIONS) for k in EMOTIONS}
        
    return corrected_probs