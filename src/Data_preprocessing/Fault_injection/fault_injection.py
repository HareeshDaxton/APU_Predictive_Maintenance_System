import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import os

# --- Configuration ---
INPUT_FILE = r"e:\APU_Predictive_Maintenance\Artifacts\training_datasets\train_FD002.csv"
OUTPUT_FILE = r"e:\APU_Predictive_Maintenance\Artifacts\Fault_injected_data\Fault_injection_dataset.csv"
RANDOM_STATE = 42
N_ESTIMATORS = 100
CONTAMINATION = 'auto'

# Feature columns
OP_SETTINGS = [f'op_setting_{i}' for i in range(1, 4)]
SENSORS = [f'sensor_{i}' for i in range(1, 22)]
FEATURES = OP_SETTINGS + SENSORS

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def train_unsupervised_model(df):
    print("Training Isolation Forest on HEALTHY data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    
    iso_forest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )
    iso_forest.fit(X_scaled)
    
    print("Model trained.")
    return scaler, iso_forest

def compute_anomaly_scores(df, scaler, model):
    # Reuse the same scaler and model - NO REFIT
    X_scaled = scaler.transform(df[FEATURES])
    scores = -model.decision_function(X_scaled)
    return scores

def inject_gradual_drift(df, engine_id):
    # Fault 1: Gradual Drift on sensor_2, sensor_7
    targets = ['sensor_2', 'sensor_7']
    df_faulty = df.copy()
    
    # Per engine logic
    start_cycle_index = np.random.randint(0, len(df_faulty))
    drift_rate = np.random.uniform(0.001, 0.003)
    
    for sensor in targets:
        mean_val = df_faulty[sensor].mean()
        # Apply strict relative step drift
        for i in range(start_cycle_index, len(df_faulty)):
            local_step = i - start_cycle_index
            drift_factor = 1 + (drift_rate * local_step)
            
            # Cap drift at +/- 10%
            max_val = mean_val * 1.10
            min_val = mean_val * 0.90
            
            new_val = df_faulty.iloc[i][sensor] * drift_factor
            df_faulty.at[df_faulty.index[i], sensor] = np.clip(new_val, min_val, max_val)
            
    return df_faulty, "sensor_2|sensor_7"

def inject_bias_shift(df, engine_id):
    # Fault 2: Bias Shift on sensor_4, sensor_8
    targets = ['sensor_4', 'sensor_8']
    df_faulty = df.copy()
    
    start_cycle_index = np.random.randint(0, len(df_faulty))
    offset_pct = np.random.uniform(0.01, 0.02)
    
    for sensor in targets:
        mean_val = df_faulty[sensor].mean()
        offset = mean_val * offset_pct
        # Apply constant offset from start to end
        df_faulty.iloc[start_cycle_index:, df_faulty.columns.get_loc(sensor)] += offset
        
    return df_faulty, "sensor_4|sensor_8"

def inject_noise_increase(df, engine_id):
    # Fault 3: Noise Increase on sensor_3
    target = 'sensor_3'
    df_faulty = df.copy()
    
    # Calculate original stats
    values = df_faulty[target].values
    original_std = np.std(values)
    
    # Linear increase factor from 1x to 3x std
    n_rows = len(values)
    noise_scale = np.linspace(original_std, 3 * original_std, n_rows)
    
    # Generate Zero-Mean Noise
    noise = np.random.normal(0, 1, n_rows) * noise_scale
    vals_centered = noise - np.mean(noise) # Strict integrity check
    
    # Add noise
    df_faulty[target] = values + vals_centered
    
    return df_faulty, "sensor_3"

def inject_intermittent_spike(df, engine_id):
    # Fault 4: Spikes on sensor_9
    target = 'sensor_9'
    df_faulty = df.copy()
    
    values = df_faulty[target].values
    original_std = np.std(values)
    n_rows = len(values)
    
    # Randomly selecting spike centers with explicit gap
    spike_candidates = []
    last_spike = -100
    
    # Simple logic to ensuring spacing
    # We will try to place a few spikes
    num_potential_spikes = max(1, int(n_rows * 0.05)) # 5% density max
    indices = np.random.choice(n_rows, num_potential_spikes, replace=False)
    indices.sort()
    
    for idx in indices:
        if idx - last_spike >= 10: # Minimum 10 cycle gap
            duration = np.random.randint(1, 4) # 1-3 cycles
            magnitude = np.random.uniform(3, 5) * original_std
            
            # Apply spike
            end_idx = min(idx + duration, n_rows)
            df_faulty.iloc[idx:end_idx, df_faulty.columns.get_loc(target)] += magnitude
            
            last_spike = end_idx
            
    return df_faulty, "sensor_9"

def inject_coupled_fault(df, engine_id):
    # Fault 5: Coupled drift on sensor_11, sensor_12
    targets = ['sensor_11', 'sensor_12']
    df_faulty = df.copy()
    
    start_cycle_index = np.random.randint(0, len(df_faulty))
    drift_rate = np.random.uniform(0.001, 0.0025)
    
    for sensor in targets:
        mean_val = df_faulty[sensor].mean()
        # Strictly coupled: same start, same rate logic
        for i in range(start_cycle_index, len(df_faulty)):
            local_step = i - start_cycle_index
            drift_factor = 1 + (drift_rate * local_step)
             # Cap drift at +/- 10%
            max_val = mean_val * 1.10
            min_val = mean_val * 0.90
            
            new_val = df_faulty.iloc[i][sensor] * drift_factor
            df_faulty.at[df_faulty.index[i], sensor] = np.clip(new_val, min_val, max_val)

    return df_faulty, "sensor_11|sensor_12"


def generate_fault_dataset(df_original, fault_func, fault_label, fault_type):
    print(f"Generating Fault {fault_label}: {fault_type}...")
    df_faulty_list = []
    
    grouped = df_original.groupby('engine_id')
    
    for engine_id, group in grouped:
        # Pass a copy to avoid mutating original within the loop (though we reset anyway)
        group_copy = group.copy()
        
        # Inject fault
        df_injected, target_str = fault_func(group_copy, engine_id)
        
        # Add metadata
        df_injected['fault_label'] = fault_label
        df_injected['fault_type'] = fault_type
        df_injected['fault_target'] = target_str
        
        df_faulty_list.append(df_injected)
        
    return pd.concat(df_faulty_list)

def main():
    np.random.seed(RANDOM_STATE)
    
    # 1. Load Data
    df = load_data(INPUT_FILE)
    
    # IMPORTANT: Keep a clean copy for integrity verification
    df_clean_integrity_ref = df.copy()
    
    # 2. Train Model on HEALTHY data
    scaler, model = train_unsupervised_model(df)
    
    # 3. Score HEALTHY data
    print("Scoring healthy data...")
    anomaly_scores_healthy = compute_anomaly_scores(df, scaler, model)
    df_healthy = df.copy()
    df_healthy['anomaly_score'] = anomaly_scores_healthy
    df_healthy['fault_label'] = 0
    df_healthy['fault_type'] = 'healthy'
    df_healthy['fault_target'] = 'none'
    
    # 4. Generate Faults
    # Each function returns a full DataFrame with the fault injected per engine
    
    df_fault_1 = generate_fault_dataset(df, inject_gradual_drift, 1, 'gradual_drift')
    df_fault_2 = generate_fault_dataset(df, inject_bias_shift, 2, 'bias_shift')
    df_fault_3 = generate_fault_dataset(df, inject_noise_increase, 3, 'noise_increase')
    df_fault_4 = generate_fault_dataset(df, inject_intermittent_spike, 4, 'spike')
    df_fault_5 = generate_fault_dataset(df, inject_coupled_fault, 5, 'coupled_fault')
    
    # 5. Score Faulty Datasets (Reuse Scaler & Model)
    print("Scoring faulty datasets...")
    datasets = [df_fault_1, df_fault_2, df_fault_3, df_fault_4, df_fault_5]
    
    score_list = []
    for d in datasets:
        d['anomaly_score'] = compute_anomaly_scores(d, scaler, model)
        score_list.append(d)
        
    # 6. Combine
    final_df = pd.concat([df_healthy] + score_list, ignore_index=True)
    
    # 7. Verification
    print("\n--- Verifying Dataset Integrity ---")
    
    # A. Check Rows
    total_rows = len(final_df)
    print(f"Total Rows: {total_rows}")
    counts = final_df['fault_label'].value_counts()
    print("Class Balance:\n", counts)
    assert counts.nunique() == 1, "Classes are not balanced!"
    
    # B. NaN Check
    assert not final_df.isnull().values.any(), "Dataset contains NaNs!"
    
    # C. Healthy Integrity (Byte-level check of original columns)
    # Extract healthy portion from final df
    final_healthy_portion = final_df[final_df['fault_label'] == 0][df_clean_integrity_ref.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(df_clean_integrity_ref, final_healthy_portion)
    print("✅ Healthy data integrity verified (byte-identical).")
    
    # D. Anomaly Score Sanity
    median_healthy = final_df[final_df['fault_label'] == 0]['anomaly_score'].median()
    median_faulty = final_df[final_df['fault_label'] > 0]['anomaly_score'].median()
    print(f"Median Healthy Score: {median_healthy:.4f}")
    print(f"Median Faulty Score : {median_faulty:.4f}")
    
    if median_faulty > median_healthy:
        print("✅ Anomaly score assumption holds (Faulty > Healthy).")
    else:
        print("⚠️ Warning: Faulty scores are not significantly higher. Check injection magnitude.")

    # 8. Save
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
