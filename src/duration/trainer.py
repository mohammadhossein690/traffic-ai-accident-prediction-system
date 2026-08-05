import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Constants
DATA_PATH = r"C:\Users\NoteBook\Desktop\traffic_ai_system\dataset\US_Accidents_500k.csv"
MODEL_DIR = r"C:\Users\NoteBook\Desktop\traffic_ai_system\models\duration"
MODEL_PATH = os.path.join(MODEL_DIR, "duration_pipeline.pkl")

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning, parsing, and extracts features."""
    df = df.copy()
    
    # Convert time columns to datetime
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    
    # Calculate duration in minutes and clean
    df["Duration"] = (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    df = df.dropna(subset=["Duration"])
    df = df[(df["Duration"] > 0) & (df["Duration"] <= 776.0)] # 99th percentile limit
    
    # Extract time-based features
    df["Hour"] = df["Start_Time"].dt.hour
    df["Month"] = df["Start_Time"].dt.month
    df["Weekday"] = df["Start_Time"].dt.dayofweek
    df["Year"] = df["Start_Time"].dt.year
    df["Day"] = df["Start_Time"].dt.day
    df["Is_Weekend"] = (df["Weekday"] >= 5).astype(int)
    
    # Fill missing values
    numeric_to_fill = ["Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)", "Wind_Speed(mph)", "Pressure(in)"]
    for col in numeric_to_fill:
        df[col] = df[col].fillna(df[col].median())
        
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")
    df["Sunrise_Sunset"] = df["Sunrise_Sunset"].fillna("Unknown")
    df["Precipitation(in)"] = df["Precipitation(in)"].fillna(0.0)
    
    return df

def train():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Preparing and cleaning data...")
    df = prepare_data(df)
    
    # Define features
    numeric_features = [
        "Distance(mi)", "Temperature(F)", "Humidity(%)", "Severity", "Visibility(mi)",
        "Wind_Speed(mph)", "Hour", "Month", "Weekday", "Year", "Day", "Pressure(in)", "Precipitation(in)"
    ]
    binary_features = [
        "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
        "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Is_Weekend"
    ]
    categorical_features = ["Weather_Condition", "State", "Sunrise_Sunset"]
    
    features = numeric_features + binary_features + categorical_features
    
    X = df[features]
    y = df["Duration"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Column Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="passthrough"
    )
    
    # XGBoost Regressor
    xgb_regressor = xgb.XGBRegressor(
        subsample=0.8,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )
    
    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb_regressor)
    ])
    
    print("Training XGBoost Pipeline...")
    pipeline.fit(X_train, y_train)
    print("Training Complete!")
    
    # Evaluation
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nEVALUATION RESULTS")
    print("=" * 40)
    print(f"MAE  : {mae:.2f} minutes")
    print(f"RMSE : {rmse:.2f} minutes")
    print(f"R²   : {r2:.4f}")
    
    # Save Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nSaved trained pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    train()
