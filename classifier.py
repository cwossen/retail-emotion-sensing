# classifier.py

from collections import defaultdict

SIGNAL_WEIGHTS = {
    "passive":       0,
    "attention":     1,
    "curiosity":     2,
    "interest":      3,
    "re_engagement": 4,
    "disengaged":   -1
}


EMOTION_CALIBRATION = {
    "sad":  0.3,   # deflate by 70%
    "fear": 0.3,   # deflate by 70%
    "neutral": 1.5, # inflate
    "happy": 1.0,
    "angry": 0.7,
    "surprise": 1.0,
    "disgust": 0.7
}

def map_emotion_to_signal(emotion, duration, revisit):
    # Apply calibration weight to bias correction
    calibration = EMOTION_CALIBRATION.get(emotion, 1.0)
    
    if emotion == "happy":
        return "interest"
    elif emotion == "surprise":
        return "curiosity"
    elif emotion in ("sad", "fear") and calibration < 0.5:
        return "passive"  # treat inflated negatives as passive
    elif emotion == "angry":
        return "attention"
    elif emotion == "disgust":
        return "disengaged"
    elif revisit:
        return "re_engagement"
    elif duration > 3:
        return "attention"
    return "passive"

def process_detection(emotion, duration, category, signal_counts, category_scores, revisit=False):
    signal = map_emotion_to_signal(emotion, duration, revisit)
    signal_counts[signal] += 1
    category_scores[category] += SIGNAL_WEIGHTS[signal]  # ✅ correct dict

def normalize_signals(signal_counts):
    total = sum(signal_counts.values())
    return {k: v / total for k, v in signal_counts.items()} if total > 0 else {}

def generate_recommendation(signal_counts, category_scores, category_dwell=None):
    signal_scores = normalize_signals(signal_counts)

    if not category_scores:
        return "No data yet."

    top_category = max(category_scores, key=category_scores.get)
    worst_category = min(category_scores, key=category_scores.get)

    # Dwell-based insight
    if category_dwell:
        top_dwell = max(category_dwell, key=category_dwell.get)
        if top_dwell != top_category:
            dwell_note = f" Note: {top_dwell} has the highest avg dwell time despite lower foot traffic."
        else:
            dwell_note = ""
    else:
        dwell_note = ""

    if signal_scores.get("interest", 0) > 0.05:
        return f"Strong interest detected in {top_category} titles.{dwell_note}"
    elif signal_scores.get("curiosity", 0) > 0.02:
        return f"Customers are curious about {top_category}. Recommend highlighting these.{dwell_note}"
    elif signal_scores.get("disengaged", 0) > 0.1:
        return f"High disengagement detected. {worst_category} may need repositioning.{dwell_note}"
    elif signal_scores.get("attention", 0) > 0.15:
        return f"Browsing activity detected. {top_category} is drawing the most engagement.{dwell_note}"

    return f"No strong trends detected.{dwell_note}"