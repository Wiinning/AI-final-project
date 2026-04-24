import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SKIN_TYPE_MAP = {
    "Oily": [1, 0, 0, 0],
    "Dry": [0, 1, 0, 0],
    "Combination": [0, 0, 1, 0],
    "Sensitive": [0, 0, 0, 1],
    "Normal": [0, 0, 1, 0]
}

CONCERN_KEYS = ["acne", "aging", "brightening", "dryness", "sensitivity"]
SENSITIVITY_MAP = {"Mild": 0.3, "Moderate": 0.6, "Severe": 1.0}
CLIMATE_MAP = {"Humid": 1.0, "Dry": 0.0, "Temperate": 0.5, "Cold": 0.3}

CATEGORY_TO_STEP = {
    "humectant": ("Hydrating Serum", "both"),
    "emollient": ("Moisturizer", "both"),
    "barrier_repair": ("Moisturizer", "both"),
    "soothing": ("Essence/Toner", "both"),
    "brightening": ("Brightening Serum", "both"),
    "antioxidant": ("Antioxidant Serum", "AM"),
    "anti-aging": ("Anti-Aging Treatment", "PM"),
    "exfoliant_BHA": ("BHA Exfoliant", "PM"),
    "exfoliant_AHA": ("AHA Exfoliant", "PM"),
    "antibacterial": ("Spot Treatment", "PM"),
    "repair": ("Repair Essence", "both"),
    "sunscreen": ("Sunscreen", "AM"),
}

MASTER_CONFLICTS = [
    ("Retinol", "Vitamin C"), ("Retinol", "Salicylic Acid"),
    ("Retinol", "Glycolic Acid"), ("Retinol", "Lactic Acid"),
    ("Retinol", "Mandelic Acid"), ("Retinol", "Benzoyl Peroxide"),
    ("Retinaldehyde", "Vitamin C"), ("Retinaldehyde", "Salicylic Acid"),
    ("Retinaldehyde", "Glycolic Acid"), ("Retinaldehyde", "Lactic Acid"),
    ("Retinaldehyde", "Benzoyl Peroxide"),
    ("Vitamin C", "Niacinamide"), ("Vitamin C", "Benzoyl Peroxide"),
    ("Vitamin C", "Copper Peptide"),
    ("Glycolic Acid", "Salicylic Acid"), ("Glycolic Acid", "Lactic Acid"),
    ("Glycolic Acid", "Mandelic Acid"), ("Glycolic Acid", "Benzoyl Peroxide"),
    ("Lactic Acid", "Salicylic Acid"), ("Lactic Acid", "Mandelic Acid"),
    ("Lactic Acid", "Benzoyl Peroxide"), ("Mandelic Acid", "Salicylic Acid"),
    ("Mandelic Acid", "Benzoyl Peroxide"), ("Salicylic Acid", "Benzoyl Peroxide"),
    ("Benzoyl Peroxide", "Sulfur"),
    ("Niacinamide", "Vitamin C (high conc)"),
    ("Niacinamide (high conc)", "Vitamin C"),
    ("Copper Peptide", "Retinol"), ("Copper Peptide", "Glycolic Acid"),
    ("Copper Peptide", "Salicylic Acid"), ("Copper Peptide", "Vitamin C"),
    ("Ferulic Acid", "Benzoyl Peroxide"), ("Resveratrol", "Benzoyl Peroxide"),
]

NULL_AVOID_TOKENS = {"", "none", "no", "n/a", "na", "nil"}

def parse_avoid_input(avoid_input: str):
    if not avoid_input:
        return []
    cleaned = []
    for x in avoid_input.split(","):
        token = x.strip().lower()
        if token and token not in NULL_AVOID_TOKENS:
            cleaned.append(x.strip())
    return list(dict.fromkeys(cleaned))

def build_conflict_pairs(kb):
    conflict_pairs = set()
    for _, row in kb.iterrows():
        if pd.notna(row["conflict_with"]) and str(row["conflict_with"]).strip():
            for conflict in str(row["conflict_with"]).split(","):
                conflict = conflict.strip()
                if conflict:
                    conflict_pairs.add(tuple(sorted([row["ingredient_name"], conflict])))
    for a, b in MASTER_CONFLICTS:
        conflict_pairs.add(tuple(sorted([a, b])))
    return conflict_pairs

def vectorize_profile(profile: dict) -> np.ndarray:
    skin_vec = SKIN_TYPE_MAP.get(profile.get("skin_type", "Combination"), [0, 0, 1, 0])
    concern_vec = [1 if c in profile.get("concerns", []) else 0 for c in CONCERN_KEYS]
    sensitivity_val = SENSITIVITY_MAP.get(profile.get("sensitivity", "Mild"), 0.3)
    climate_val = CLIMATE_MAP.get(profile.get("climate", "Humid"), 1.0)
    return np.array(skin_vec + concern_vec + [sensitivity_val, climate_val], dtype=float)

def vectorize_user_row(row) -> np.ndarray:
    skin_vec = SKIN_TYPE_MAP.get(str(row["Skin_Type"]).strip(), [0, 0, 1, 0])
    acne_concern = 1 if float(row.get("Acne_Severity", 0)) > 3 else 0
    aging_concern = 1 if float(row.get("Aging_Severity", 0)) > 3 else 0
    bright_concern = 1 if float(row.get("Pigmentation_Severity", 0)) > 3 else 0
    dryness_concern = 1 if float(row.get("Dryness_Severity", 0)) > 3 else 0
    sensitivity_concern = 1 if float(row.get("Sensitivity_Severity", 0)) > 3 else 0
    concern_vec = [acne_concern, aging_concern, bright_concern, dryness_concern, sensitivity_concern]
    sensitivity_val = min(float(row.get("Sensitivity_Severity", 0)) / 10.0, 1.0)
    climate_val = CLIMATE_MAP.get(str(row.get("Climate", "Humid")).strip(), 1.0)
    return np.array(skin_vec + concern_vec + [sensitivity_val, climate_val], dtype=float)

def score_ingredient(ing_name, profile, kb, sensitivity_weight=1.0):
    row = kb[kb["ingredient_name"] == ing_name]
    if row.empty or ing_name in profile.get("avoid", []):
        return 0.0

    row = row.iloc[0]
    skin = profile.get("skin_type", "Combination")
    skin_map = {
        "Oily": "oily",
        "Dry": "dry",
        "Combination": "combination",
        "Sensitive": "sensitive",
        "Normal": "combination"
    }
    skin_key = f"skin_{skin_map.get(skin, skin.lower())}"

    score = float(row.get(skin_key, 0)) * 35

    for concern in profile.get("concerns", []):
        col = f"concern_{concern}"
        if col in kb.columns:
            score += float(row.get(col, 0)) * 25

    score -= float(row.get("irritation_level", 0)) * 20 * sensitivity_weight

    climate = profile.get("climate", "Humid")
    cat = str(row.get("category", ""))
    if climate == "Humid" and cat in ["exfoliant_BHA", "exfoliant_AHA", "antibacterial"]:
        score += 5
    elif climate in ["Dry", "Cold"] and cat in ["humectant", "emollient", "barrier_repair"]:
        score += 5
    elif climate == "Cold" and cat in ["barrier_repair", "emollient"]:
        score += 3

    return round(score, 2)

def remove_conflicts(recommended_pairs, conflict_pairs):
    clean, removal_log, names_so_far = [], [], []
    for ing, score in recommended_pairs:
        conflicting_with = [
            ex for ex in names_so_far
            if tuple(sorted([ing, ex])) in conflict_pairs
        ]
        if conflicting_with:
            removal_log.append(
                f"Removed '{ing}' — conflicts with: {', '.join(conflicting_with)}"
            )
        else:
            clean.append((ing, score))
            names_so_far.append(ing)
    return clean, removal_log

def check_conflicts(ingredient_list, conflict_pairs):
    out = []
    for i in range(len(ingredient_list)):
        for j in range(i + 1, len(ingredient_list)):
            pair = tuple(sorted([ingredient_list[i], ingredient_list[j]]))
            if pair in conflict_pairs:
                out.append(f"{ingredient_list[i]} + {ingredient_list[j]}")
    return out

def build_routine(top_ingredients, kb):
    am, pm = {}, {}
    for ing in top_ingredients:
        row = kb[kb["ingredient_name"] == ing]
        if row.empty:
            continue
        cat = row.iloc[0]["category"]
        step, timing = CATEGORY_TO_STEP.get(cat, ("Treatment", "both"))
        if timing in ("AM", "both") and step not in am:
            am[step] = ing
        if timing in ("PM", "both") and step not in pm:
            pm[step] = ing
    return {"AM": am, "PM": pm}

def compute_confidence(scores, profile):
    if not scores:
        return 0.0
    num_concerns = len(profile.get("concerns", []))
    max_possible = 35 + (25 * max(num_concerns, 1))
    return round((np.mean(scores) / max_possible) * 100, 1)

def explain_similarity(user_profile, similar_row):
    same_parts = []
    diff_parts = []

    if str(similar_row["Skin_Type"]).strip() == user_profile["skin_type"]:
        same_parts.append("skin type")
    else:
        diff_parts.append(f"skin type differs ({similar_row['Skin_Type']})")

    if str(similar_row.get("Climate", "")).strip() == user_profile["climate"]:
        same_parts.append("climate")
    else:
        diff_parts.append(f"climate differs ({similar_row['Climate']})")

    return same_parts, diff_parts

def find_similar_profiles(user_vec, user_vectors, users_df, profile, top_n=3):
    sims = cosine_similarity([user_vec], user_vectors)[0]
    ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
    results = []

    for idx, sim in ranked:
        u = users_df.iloc[idx]
        same_parts, diff_parts = explain_similarity(profile, u)

        reason = []
        if same_parts:
            reason.append("matched on " + ", ".join(same_parts))
        if diff_parts:
            reason.append("but " + "; ".join(diff_parts))

        results.append({
            "label": f"{u['Skin_Type']} | {u['Climate']} | Age {u['Age']}",
            "similarity": round(float(sim), 4),
            "reason": " ".join(reason) if reason else "closest overall feature match"
        })
    return results

def recommend(profile, kb, user_vectors, users_df, conflict_pairs, top_n=8):
    s_weight = {"Mild": 0.5, "Moderate": 1.0, "Severe": 1.5}.get(
        profile.get("sensitivity", "Mild"), 0.5
    )

    all_scores = {
        ing: score_ingredient(ing, profile, kb, s_weight)
        for ing in kb["ingredient_name"]
    }

    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    candidates = [
        (ing, sc) for ing, sc in ranked
        if sc > 20 and ing not in profile.get("avoid", [])
    ]

    recommended, removal_log = remove_conflicts(candidates, conflict_pairs)
    recommended = recommended[:top_n]

    avoid_list = list(dict.fromkeys(
        profile.get("avoid", []) +
        [ing for ing, sc in ranked if sc <= 15]
    ))[:8]

    rec_names = [ing for ing, _ in recommended]
    conflicts = check_conflicts(rec_names, conflict_pairs)
    routine = build_routine(rec_names[:6], kb)
    confidence = compute_confidence([sc for _, sc in recommended], profile)
    user_vec = vectorize_profile(profile)
    similar = find_similar_profiles(user_vec, user_vectors, users_df, profile)

    return {
        "recommended": recommended,
        "avoid": avoid_list,
        "conflicts": conflicts,
        "routine": routine,
        "confidence": confidence,
        "similar": similar,
        "removal_log": removal_log
    }