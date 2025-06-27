import joblib
import re
import textstat
import json
import numpy as np
import scipy.sparse as sp

# Load model and vectorizer
model = joblib.load('models/model2.pkl')
vectorizer = joblib.load('models/vectorizer2.pkl')

# Load config
with open('config.json') as f:
    config = json.load(f)

regex_blacklist = [re.compile(pat, re.IGNORECASE) for pat in config["regex_blacklist"]]
thresholds = config["score_thresholds"]  # contains "authentic" and "borderline"

# Clean text helper
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Count regex matches helper
def regex_flag_count(text):
    return sum(bool(pattern.search(text)) for pattern in regex_blacklist)

# Main prediction function
def predict_profile(headline, bio):
    combined_text = headline + " " + bio
    clean = clean_text(combined_text)
    vec = vectorizer.transform([clean])
    readability = textstat.flesch_reading_ease(combined_text)

    # Count buzzword hits separately in headline and bio
    def count_hits(text):
        return sum(bool(pattern.search(text)) for pattern in regex_blacklist)

    headline_hits = count_hits(headline)
    bio_hits = count_hits(bio)

    total_hits = headline_hits + bio_hits

    numeric_feats = np.array([[readability, total_hits]])
    final_input = sp.hstack([vec, numeric_feats])
    
    prob = model.predict_proba(final_input)[0][1]  # probability of "authentic"

    # Penalize the score by buzzword hits (tweak penalty multiplier as needed)
    penalty = 0.05 * total_hits
    adjusted_prob = max(0, prob - penalty)

    # Determine verdict based on thresholds and adjusted score
    if adjusted_prob >= thresholds["authentic"]:
        verdict = "authentic"
    elif adjusted_prob >= thresholds["borderline"]:
        verdict = "borderline"
    else:
        verdict = "likely_fabricated"

    # Build reason string dynamically
    reasons = []
    if total_hits > 0:
        reasons.append(f"{total_hits} buzzword(s) detected")
    if readability < 40:
        reasons.append("low readability score")
    if not reasons:
        reasons.append("No suspicious patterns detected")

    reason = ", ".join(reasons)

    flagged = []

    # Flag empty fields
    if not headline.strip():
        flagged.append("headline")
    if not bio.strip():
        flagged.append("bio")

    # Flag fields that contain buzzwords
    if headline_hits > 0:
        flagged.append("headline")
    if bio_hits > 0:
        flagged.append("bio")

    # Flag bio for low readability
    if readability < 40:
        flagged.append("bio")

    return {
        "authenticity_score": round(adjusted_prob, 2),
        "verdict": verdict,
        "reason": reason,
        "flagged_fields": list(set(flagged))
    }
