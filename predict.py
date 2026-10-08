import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import softmax

def create_mock_dataset():
    np.random.seed(42)
    drivers = [
        "Andrea Kimi Antonelli", "Max Verstappen", "Lando Norris", 
        "George Russell", "Charles Leclerc", "Lewis Hamilton", 
        "Oscar Piastri", "Carlos Sainz", "Fernando Alonso", "Pierre Gasly"
    ]
    
    records = []
    for race_id in range(30):
        grid = np.random.permutation(np.arange(1, len(drivers) + 1))
        for idx, driver in enumerate(drivers):
            qualifying_pos = grid[idx]
            street_win_rate = np.random.uniform(0.0, 0.4)
            rolling_pace_ewma = np.random.normal(loc=90.0, scale=0.5) - (0.05 * (10 - qualifying_pos))
            low_speed_traction_score = np.random.uniform(7.0, 10.0)
            safety_car_incident_risk = np.random.uniform(0.01, 0.15)
            
            latent_perf = (
                - 1.8 * qualifying_pos 
                + 4.0 * street_win_rate 
                - 2.5 * (rolling_pace_ewma - 89.5) 
                + 0.8 * low_speed_traction_score 
                + np.random.normal(0, 1.2)
            )
            
            records.append({
                "race_id": race_id,
                "driver": driver,
                "grid_pos": qualifying_pos,
                "street_win_rate": street_win_rate,
                "rolling_pace_ewma": rolling_pace_ewma,
                "low_speed_traction": low_speed_traction_score,
                "incident_risk": safety_car_incident_risk,
                "latent_perf": latent_perf
            })
            
    df = pd.DataFrame(records)
    df["finish_pos"] = df.groupby("race_id")["latent_perf"].rank(ascending=False, method="first").astype(int)
    df["relevance"] = 11 - df["finish_pos"]
    return df

def train_and_predict():
    df = create_mock_dataset()
    features = ["grid_pos", "street_win_rate", "rolling_pace_ewma", "low_speed_traction", "incident_risk"]
    
    df = df.sort_values("race_id").reset_index(drop=True)
    X = df[features]
    y = df["relevance"]
    groups = df.groupby("race_id").size().to_numpy()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=100,
        learning_rate=0.05,
        importance_type="gain",
        random_state=42
    )
    ranker.fit(X, y, group=groups)

    singapore_grid = pd.DataFrame([
        {"driver": "Andrea Kimi Antonelli", "grid_pos": 1, "street_win_rate": 0.35, "rolling_pace_ewma": 89.4, "low_speed_traction": 9.6, "incident_risk": 0.04},
        {"driver": "Max Verstappen",        "grid_pos": 2, "street_win_rate": 0.38, "rolling_pace_ewma": 89.5, "low_speed_traction": 9.4, "incident_risk": 0.03},
        {"driver": "Lando Norris",          "grid_pos": 3, "street_win_rate": 0.28, "rolling_pace_ewma": 89.5, "low_speed_traction": 9.5, "incident_risk": 0.05},
        {"driver": "George Russell",        "grid_pos": 4, "street_win_rate": 0.22, "rolling_pace_ewma": 89.7, "low_speed_traction": 9.1, "incident_risk": 0.04},
        {"driver": "Charles Leclerc",       "grid_pos": 5, "street_win_rate": 0.25, "rolling_pace_ewma": 89.8, "low_speed_traction": 9.3, "incident_risk": 0.06},
        {"driver": "Lewis Hamilton",        "grid_pos": 6, "street_win_rate": 0.30, "rolling_pace_ewma": 90.0, "low_speed_traction": 8.9, "incident_risk": 0.04},
    ])

    raw_scores = ranker.predict(singapore_grid[features])
    
    temperature = 1.2
    probabilities = softmax(raw_scores / temperature)
    singapore_grid["win_probability"] = np.round(probabilities * 100, 2)
    
    results = singapore_grid[["driver", "grid_pos", "win_probability"]].sort_values(
        by="win_probability", ascending=False
    ).reset_index(drop=True)

    print("\n--- SINGAPORE GP WIN PROBABILITY (LambdaMART) ---")
    for _, row in results.iterrows():
        bar = "█" * int(row["win_probability"] // 2)
        print(f"{row['driver']:<24} (P{row['grid_pos']}) : {row['win_probability']:5.1f}% | {bar}")

if __name__ == "__main__":
    train_and_predict()
