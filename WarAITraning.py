import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def train():

    # ================================
    # LOAD ALL WAR FILES
    # ================================
    files = sorted(glob.glob("data/*.xlsx"))
    wars = []

    if len(files) == 0:
        print("❌ No war files found in /data/")
        return

    for f in files:
        df = pd.read_excel(f)

        # ================================
        # AUTO FIX COLUMN NAMES
        # ================================
        df.columns = (
            df.columns
            .str.strip()
            .str.upper()
            .str.replace("  ", " ")   # remove double spaces
            .str.replace("_", " ")    # unify underscores
        )

        # ================================
        # VALIDATE REQUIRED COLUMNS
        # ================================
        required = [
            "NAME",
            "TAG",
            "ATTACKER TH",
            "DEFENDER TH",
            "DESTRUCTION",
            "DEFENSE STAR",
            "DEFENSE DESTRUCTION",
            "TRUE STARS"
        ]

        for col in required:
            if col not in df.columns:
                print(f"❌ Missing column in {f}: {col}")
                return

        df = df.dropna(how="all").fillna(0)

        # ================================
        # GROUP BOTH ATTACKS FOR EACH WAR
        # ================================
        grouped = df.groupby("TAG").agg({
            "NAME": "first",
            "ATTACKER TH": "first",
            "DEFENDER TH": "mean",
            "DESTRUCTION": "mean",
            "DEFENSE STAR": "mean",
            "DEFENSE DESTRUCTION": "mean",
            "TRUE STARS": "sum"
        }).reset_index()

        wars.append(grouped)

    # ================================
    # WEIGHT RECENT WARS MORE (80/20)
    # ================================
    recent_5 = wars[-5:]
    older = wars[:-5]

    recent_df = pd.concat(recent_5, ignore_index=True)
    recent_df["WEIGHT"] = 0.80

    if len(older) > 0:
        older_df = pd.concat(older, ignore_index=True)
        older_df["WEIGHT"] = 0.20
        full_df = pd.concat([recent_df, older_df], ignore_index=True)
    else:
        full_df = recent_df

    # ================================
    # TRAIN ML MODEL
    # ================================
    X = full_df[[
        "ATTACKER TH",
        "DEFENDER TH",
        "DESTRUCTION",
        "DEFENSE STAR",
        "DEFENSE DESTRUCTION"
    ]]

    y = full_df["TRUE STARS"]
    w = full_df["WEIGHT"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y, sample_weight=w)

    # ================================
    # SAVE MODEL
    # ================================
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/war_model.pkl")

    print("🔥 AI Model Trained Successfully! Saved to models/war_model.pkl")
