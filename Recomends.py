import pandas as pd
import glob
import joblib


# ==========================================================
# INTERNAL FUNCTION: Load all wars + prepare final dataframe
# ==========================================================
def _load_final_df():

    model = joblib.load("models/war_model.pkl")

    files = sorted(glob.glob("data/*.xlsx"))
    wars = []

    for f in files:
        df = pd.read_excel(f)

        # Fix column names
        df.columns = (
            df.columns
            .str.strip()
            .str.upper()
            .str.replace("  ", " ")
            .str.replace("_", " ")
        )

        df = df.dropna(how="all").fillna(0)

        # Group both attacks
        grouped = df.groupby("TAG").agg({
            "NAME": "first",
            "ATTACKER TH": "first",
            "DEFENDER TH": "mean",
            "DESTRUCTION": "mean",
            "DEFENSE STAR": "mean",
            "DEFENSE DESTRUCTION": "mean",
            "TRUE STARS": "sum" if "TRUE STARS" in df.columns else "mean"
        }).reset_index()

        wars.append(grouped)

    # Merge all wars
    merged = pd.concat(wars, ignore_index=True)

    # Combine same TAG players across wars
    final = merged.groupby("TAG").agg({
        "NAME": "first",
        "ATTACKER TH": "first",
        "DEFENDER TH": "mean",
        "DESTRUCTION": "mean",
        "DEFENSE STAR": "mean",
        "DEFENSE DESTRUCTION": "mean"
    }).reset_index()

    # Prediction
    X = final[[
        "ATTACKER TH",
        "DEFENDER TH",
        "DESTRUCTION",
        "DEFENSE STAR",
        "DEFENSE DESTRUCTION"
    ]]

    final["PREDICTED_STARS"] = model.predict(X)

    return final


# ==========================================================
# API FUNCTION FOR BOT: return dataframe to other commands
# ==========================================================
def get_dataframe():
    return _load_final_df()


# ==========================================================
# OLD FUNCTION USED BY /predict_ai (text output)
# ==========================================================
def run():

    final = _load_final_df()

    # Sort best attackers
    final = final.sort_values(by="PREDICTED_STARS", ascending=False)

    msg = "⭐ **AI Next-War Predictions** ⭐\n\n"

    for _, row in final.iterrows():
        msg += f"**{row['NAME']}** → ⭐ **{round(row['PREDICTED_STARS'], 2)}**\n"

    return msg
