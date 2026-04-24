# ============================================================
# AI SKINCARE INGREDIENT RECOMMENDATION TOOL — STREAMLIT UI
# ITM-360: Artificial Intelligence
# American University of Phnom Penh
# Team: Kry Winning, Brak Sreytoch, Ek Sithiroth
# Advisor: Kuntha PIN
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from skincare_engine import (
    parse_avoid_input,
    build_conflict_pairs,
    vectorize_user_row,
    recommend,
)

warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Skincare Advisor",
    page_icon="🌿",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fdf6f0; }

    .stButton>button {
        background-color: #8B5E3C;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-size: 16px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #6b4226;
        color: white;
    }

    .result-box {
        background-color: #fff8f3;
        border-left: 5px solid #8B5E3C;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        color: #2b2b2b !important;
    }

    .result-box * {
        color: #2b2b2b !important;
    }

    .conflict-box {
        background-color: #fff0f0;
        border-left: 5px solid #e74c3c;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        color: #2b2b2b !important;
    }

    .confidence-box {
        background-color: #f0fff4;
        border-left: 5px solid #2ecc71;
        padding: 1em;
        border-radius: 8px;
        margin-bottom: 1em;
        color: #2b2b2b !important;
    }

    .muted-note {
        font-size: 0.95rem;
        opacity: 0.9;
        margin-top: 0.35rem;
    }

    h1 { color: #5a3e2b; }
    h2, h3 { color: #7a5230; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    kb    = pd.read_csv('ingredient_kb.csv')       # root level
    users = pd.read_csv('data/users.csv')
    return kb, users

@st.cache_resource
def build_knn(user_vectors, user_labels):
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(user_vectors, user_labels)
    return knn


# ── INITIALIZE ────────────────────────────────────────────────────────────────
kb, users_df   = load_data()
conflict_pairs = build_conflict_pairs(kb)
user_vectors   = np.array([vectorize_user_row(row) for _, row in users_df.iterrows()])
knn_model      = build_knn(user_vectors, users_df['Skin_Type'].values)

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("AI Skincare Ingredient Recommendation Tool")
st.markdown("**ITM-360 Artificial Intelligence · Group 4 · AUPP**")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📋 Your Skin Profile")

    skin_type = st.selectbox(
        "Skin Type",
        ["Oily", "Dry", "Combination", "Sensitive", "Normal"]
    )
    concerns = st.multiselect(
        "Skin Concerns (select all that apply)",
        ["acne", "aging", "brightening", "dryness", "sensitivity"],
        default=["acne"]
    )
    sensitivity = st.selectbox(
        "Sensitivity Level",
        ["Mild", "Moderate", "Severe"]
    )
    climate = st.selectbox(
        "Your Climate",
        ["Humid", "Dry", "Temperate", "Cold"]
    )
    avoid_input = st.text_input(
        "Ingredients to Avoid (comma-separated, optional)",
        placeholder="e.g. Retinol, Benzoyl Peroxide"
    )
    run = st.button("✨ Get My Recommendations")

with col2:
    if run:
        if not concerns:
            st.warning("Please select at least one skin concern.")
        else:
            avoid_list = parse_avoid_input(avoid_input)
            profile = {
                "skin_type":   skin_type,
                "concerns":    concerns,
                "sensitivity": sensitivity,
                "climate":     climate,
                "avoid":       avoid_list
            }

            with st.spinner("Analyzing your skin profile..."):
                result = recommend(profile, kb, user_vectors, users_df, conflict_pairs)

            # ── CONFIDENCE ───────────────────────────────────────────────────
            conf = result['confidence']
            conf_color = "#2ecc71" if conf >= 70 else "#e67e22" if conf >= 50 else "#e74c3c"
            st.markdown(f"""
            <div class="confidence-box">
                <h3 style="margin:0">Confidence Score:
                <span style="color:{conf_color}">{conf}%</span></h3>
                <p style="margin:0;">
                    This result shows the overall recommendation strength based on profile matching and ingredient suitability.
                </p>
                <p class="muted-note">
                    AI can make mistakes. Users may use the reccomendations as seen fit.
                </p>
            </div>
                """, unsafe_allow_html=True)

            # ── SIMILAR PROFILES ─────────────────────────────────────────────
            st.subheader("Most Similar User Profiles (k-NN)")
            st.caption("These are the closest overall feature matches, not exact copies of your profile.")

            for item in result["similar"]:
                st.progress(item["similarity"], text=f"{item['label']} — similarity: {item['similarity']:.3f}")
                st.caption(item["reason"])

            # ── RECOMMENDED INGREDIENTS ──────────────────────────────────────
            st.subheader("Recommended Ingredients")
            for ing, score in result['recommended'][:6]:
                rows = kb[kb['ingredient_name'] == ing]
                if rows.empty: continue
                row = rows.iloc[0]
                st.markdown(f"""
                <div class="result-box">
                    <b>{ing}</b> &nbsp;·&nbsp; Match Score: <b>{score}</b>
                    &nbsp;·&nbsp; Concentration:
                    <b>{row['concentration_min']}–{row['concentration_max']}%</b>
                    &nbsp;·&nbsp; Category: <i>{row['category']}</i>
                </div>
                """, unsafe_allow_html=True)

            # ── ROUTINE ──────────────────────────────────────────────────────
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.subheader("☀️ AM Routine")
                if result['routine']['AM']:
                    for step, ing in result['routine']['AM'].items():
                        st.markdown(f"- **{step}:** {ing}")
                else:
                    st.info("No AM steps generated.")
            with rcol2:
                st.subheader("🌙 PM Routine")
                if result['routine']['PM']:
                    for step, ing in result['routine']['PM'].items():
                        st.markdown(f"- **{step}:** {ing}")
                else:
                    st.info("No PM steps generated.")

            # ── AVOID LIST ───────────────────────────────────────────────────
            st.subheader("❌ Ingredients to Avoid")
            st.markdown(
                ", ".join(result['avoid'][:8]) if result['avoid'] else "None"
            )

            # ── CONFLICTS ────────────────────────────────────────────────────
            if result['conflicts']:
                st.subheader("Conflict Warnings")
                for c in result['conflicts']:
                    st.markdown(f"""
                    <div class="conflict-box"> {c}</div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No ingredient conflicts detected!")

            # ── REMOVAL LOG ──────────────────────────────────────────────────
            if result['removal_log']:
                with st.expander("Removed Ingredients and Reasons"):
                    for entry in result['removal_log']:
                        st.markdown(f"- {entry}")

    else:
        st.info("Fill in your skin profile on the left and click **Get My Recommendations**.")
        st.markdown("""
        **How it works:**
        - k-NN matches your profile against 15,000 real user profiles
        - Rule-based scoring ranks **60 ingredients** for your skin type & concerns
        - ⚠️ Conflict detection removes unsafe ingredient combinations (33 pairs)
        - Climate bonus adjusts scores based on your environment
        - Generates your personalized AM/PM skincare routine
        """)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999'>AI Skincare Tool · ITM-360 · "
    "Group 4: Kry Winning, Brak Sreytoch, Ek Sithiroth · "
    "Advisor: Kuntha PIN · AUPP</p>",
    unsafe_allow_html=True
)