import streamlit as st
import pandas as pd
import joblib
import json

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("outputs/rf_model_baseline.joblib")

# Load champion metadata
with open("data/champion_info_2.json", "r", encoding="utf-8") as f:
    champ_data = json.load(f)["data"]

# Build champion ID -> name map
champ_map = {int(v["id"]): v["name"] for _, v in champ_data.items()}
champ_ids = sorted(champ_map.keys())
champ_names = [champ_map[cid] for cid in champ_ids]
champ_name_to_id = {v: k for k, v in champ_map.items()}

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("League of Legends Win Predictor 🎮")
st.markdown("Pick 5 champions for **Team 1** and **Team 2**. Optionally add first objectives, then predict the probability that **Team 1 wins**.")

# Pick champions
st.subheader("Champion Draft")
cols = st.columns(2)
t1_picks = [cols[0].selectbox(f"Team 1 Champion {i+1}", champ_names, index=i) for i in range(5)]
t2_picks = [cols[1].selectbox(f"Team 2 Champion {i+1}", champ_names, index=i) for i in range(5)]

# Optional objectives
st.subheader("First Objectives (optional)")
objectives = ["firstBlood","firstTower","firstDragon","firstBaron","firstInhibitor","firstRiftHerald"]
obj_inputs = {}
for obj in objectives:
    obj_inputs[obj] = st.selectbox(
        f"Who got {obj}?",
        ["None","Team1","Team2"],
        index=0
    )

# ----------------------------
# Build feature vector
# ----------------------------
# New: read only column names from a text file
with open("data/feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f.readlines()]

# Start with zeros
x = pd.Series(0, index=feature_cols, dtype=int)

# Add champion picks
for champ in t1_picks:
    cid = champ_name_to_id[champ]
    col = f"t1_champ_{cid}"
    if col in x.index:
        x[col] = 1

for champ in t2_picks:
    cid = champ_name_to_id[champ]
    col = f"t2_champ_{cid}"
    if col in x.index:
        x[col] = 1

# Add objectives
for obj, val in obj_inputs.items():
    if val == "Team1":
        col = f"{obj}_team1"
        if col in x.index: x[col] = 1
    elif val == "Team2":
        col = f"{obj}_team2"
        if col in x.index: x[col] = 1

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Win Probability"):
    prob = model.predict_proba([x])[0][1]  # P(Team1 wins)
    st.success(f"**Predicted probability Team 1 wins: {prob:.2%}**")
