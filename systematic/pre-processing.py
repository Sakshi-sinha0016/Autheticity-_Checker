import re
import nltk
from nltk.tokenize import word_tokenize
import textstat
from typing import Dict, Any

def preprocess_profile_text(profile_data: Dict[str, str], regex_blacklist: list) -> Dict[str, Any]:
    cleaned_data = {}
    buzzword_hits = {}
    readability_scores = {}

    for field, text in profile_data.items():
        if not text:
            continue

        # Step 1: Clean text
        text_clean = text.lower().strip()
        text_clean = re.sub(r'\s+', ' ', text_clean)
        cleaned_data[field] = text_clean

        # Step 2: Tokenize
        tokens = word_tokenize(text_clean)

        # Step 3: Apply regex to detect buzzwords
        matched_buzzwords = []
        for pattern in regex_blacklist:
            if re.search(pattern, text_clean):
                matched_buzzwords.append(pattern)

        buzzword_hits[field] = matched_buzzwords

        # Step 4: Compute readability score
        readability_scores[field] = textstat.flesch_reading_ease(text_clean)

    return {
        "cleaned_data": cleaned_data,
        "buzzword_hits": buzzword_hits,
        "readability_scores": readability_scores
    }
