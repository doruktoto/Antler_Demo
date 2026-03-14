"""
WWTP Forecasting MVP v2 — Temperature-Aware
=============================================
Adds Braunschweig hourly temperature as exogenous feature.
Compares v1 (no temp) vs v2 (with temp) performance.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND MERGE
# =============================================================================

def load_and_prepare(wwtp_path, temp_path):
    # Load WWTP data
    df = pd.read_csv(wwtp_path)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d.%m.%Y %H:%M:%S')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    rename_map = {
        'Aeration_rate:_total_of_5EE_if_all_14_aerators_per_tank_are_blowing_EnergieeinheitenBB1_ival': 'aeration_ee',
        'DO_AKBBBB1QO2_M01U_ival': 'DO_1',
        'DO_AKBBBB1QO2_M02U_ival': 'DO_2',
        'Ammonia_AKBVBW1QNH4H01_ival': 'NH4',
        'Nitric_oxide_AKBVBW1QNO_H01_ival': 'NO',
        'phosphate_AKBVBW1QPO4H01_ival': 'PO4',
        'Redox_Potential_AKBVBW1QROXH01_ival': 'redox',
        'pH_in_tank_AKBVBW1QPH_H01_ival': 'pH',
        'Schlammalter_ival': 'sludge_age',
        'Aerobic_sludge_age_Aerobes_Schlammalter_ival': 'aerobic_sludge_age',
        'TS_AKBVBW1QTS_H01_ival': 'TS_1',
        'AKBVBW1QTS_H02_ival': 'TS_2',
        'AKBNKB1ML02H01RW01_ival': 'flow_1',
        'AKBNKB2ML02H01RW01_ival': 'flow_2',
        'AKBNKB3ML02H01RW01_ival': 'flow_3',
        'AKBNKB4ML02H01RW01_ival': 'flow_4',
    }
    df = df.rename(columns=rename_map)
    df['total_flow'] = df[['flow_1', 'flow_2', 'flow_3', 'flow_4']].sum(axis=1)
    df['est_blowers'] = (df['aeration_ee'] / 5.0) * 14.0
    
    # Load temperature
    temp = pd.read_csv(temp_path)
    temp['datetime'] = pd.to_datetime(temp['datetime'])
    temp = temp.sort_values('datetime').reset_index(drop=True)
    temp = temp.rename(columns={'temperature_2m': 'temp_c'})
    
    # Merge on datetime
    df = df.merge(temp[['datetime', 'temp_c']], on='datetime', how='left')
    
    matched = df['temp_c'].notna().sum()
    print(f"Temperature merge: {matched}/{len(df)} rows matched ({matched/len(df)*100:.1f}%)")
    
    # Forward fill small gaps in temperature (if any)
    df['temp_c'] = df['temp_c'].ffill().bfill()
    
    keep_cols = ['datetime', 'NH4', 'NO', 'PO4', 'aeration_ee', 'est_blowers',
                 'DO_1', 'DO_2', 'redox', 'pH', 'sludge_age', 'aerobic_sludge_age',
                 'TS_1', 'TS_2', 'total_flow', 'flow_1', 'flow_2', 'flow_3', 'flow_4',
                 'temp_c']
    df = df[keep_cols].copy()
    
    print(f"Temperature range: {df['temp_c'].min():.1f}°C to {df['temp_c'].max():.1f}°C")
    
    return df


# =============================================================================
# 2. FEATURE ENGINEERING (with temperature)
# =============================================================================

def add_time_features(df):
    dt = df['datetime']
    df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df['doy_sin'] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    df['is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
    df['hour'] = dt.dt.hour
    df['dayofweek'] = dt.dt.dayofweek
    return df


def build_features(df, target_col, horizon, use_temp=True):
    df = df.copy()
    df = add_time_features(df)
    
    # --- Target lags ---
    target_lags = [horizon + i for i in [0, 1, 2, 3, 4, 6, 12, 24, 48, 168]]
    for lag in target_lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # --- Rolling stats on target ---
    for w in [4, 12, 24, 48, 168]:
        df[f'{target_col}_rmean_{w}_h{horizon}'] = df[target_col].shift(horizon).rolling(window=w, min_periods=1).mean()
        df[f'{target_col}_rstd_{w}_h{horizon}'] = df[target_col].shift(horizon).rolling(window=w, min_periods=1).std()
    
    # --- Rate of change on target ---
    for p in [1, 4, 24]:
        df[f'{target_col}_diff_{p}_h{horizon}'] = df[target_col].shift(horizon).diff(p)
    
    # --- Cross-variable ---
    other = 'NO' if target_col == 'NH4' else 'NH4'
    for lag in [horizon, horizon+1, horizon+4, horizon+12, horizon+24]:
        df[f'{other}_lag_{lag}'] = df[other].shift(lag)
    df[f'{other}_rmean_24_h{horizon}'] = df[other].shift(horizon).rolling(24, min_periods=1).mean()
    
    # --- Aeration / blower features ---
    for lag in [horizon, horizon+1, horizon+4, horizon+12, horizon+24]:
        df[f'aeration_lag_{lag}'] = df['aeration_ee'].shift(lag)
        df[f'blowers_lag_{lag}'] = df['est_blowers'].shift(lag)
    df[f'aeration_rmean_12_h{horizon}'] = df['aeration_ee'].shift(horizon).rolling(12, min_periods=1).mean()
    df[f'aeration_rmean_24_h{horizon}'] = df['aeration_ee'].shift(horizon).rolling(24, min_periods=1).mean()
    
    # --- Flow features ---
    for lag in [horizon, horizon+4, horizon+12, horizon+24]:
        df[f'total_flow_lag_{lag}'] = df['total_flow'].shift(lag)
    df[f'total_flow_rmean_24_h{horizon}'] = df['total_flow'].shift(horizon).rolling(24, min_periods=1).mean()
    df[f'total_flow_rstd_24_h{horizon}'] = df['total_flow'].shift(horizon).rolling(24, min_periods=1).std()
    
    # --- Process variables ---
    for var in ['redox', 'pH', 'PO4', 'sludge_age', 'TS_1']:
        for lag in [horizon, horizon+12, horizon+24]:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)
    
    # --- DO if available ---
    if df['DO_1'].notna().sum() > 100:
        for lag in [horizon, horizon+4, horizon+12]:
            df[f'DO_1_lag_{lag}'] = df['DO_1'].shift(lag)
    
    # === TEMPERATURE FEATURES (the key addition) ===
    if use_temp and 'temp_c' in df.columns:
        # Current temperature (lagged by horizon — what we know at prediction time)
        for lag in [horizon, horizon+4, horizon+12, horizon+24]:
            df[f'temp_lag_{lag}'] = df['temp_c'].shift(lag)
        
        # Rolling temperature stats
        df[f'temp_rmean_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean()
        df[f'temp_rmean_72_h{horizon}'] = df['temp_c'].shift(horizon).rolling(72, min_periods=1).mean()
        df[f'temp_rmean_168_h{horizon}'] = df['temp_c'].shift(horizon).rolling(168, min_periods=1).mean()
        df[f'temp_rstd_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).std()
        
        # Temperature trend (is it getting colder or warmer?)
        df[f'temp_diff_24_h{horizon}'] = df['temp_c'].shift(horizon).diff(24)
        df[f'temp_diff_72_h{horizon}'] = df['temp_c'].shift(horizon).diff(72)
        df[f'temp_diff_168_h{horizon}'] = df['temp_c'].shift(horizon).diff(168)
        
        # Min/max in last 24h (captures overnight lows, daytime highs)
        df[f'temp_min_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).min()
        df[f'temp_max_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).max()
        
        # Temperature × aeration interaction (cold + low aeration = nitrification collapse)
        df[f'temp_x_aeration_h{horizon}'] = (
            df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean() *
            df['aeration_ee'].shift(horizon).rolling(24, min_periods=1).mean()
        )
        
        # Below-zero flag (biological activity near-halt)
        df[f'temp_below_zero_h{horizon}'] = (df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean() < 0).astype(int)
        
        # Cold stress duration: hours below 5°C in last 72h
        df[f'cold_hours_72_h{horizon}'] = (
            (df['temp_c'].shift(horizon) < 5).rolling(72, min_periods=1).sum()
        )
    
    # --- Target ---
    df['target'] = df[target_col].shift(-horizon)
    
    return df


# =============================================================================
# 3. TRAIN / EVALUATE
# =============================================================================

def train_and_evaluate(df, target_col, horizon, use_temp=True, test_frac=0.15):
    label = "WITH TEMP" if use_temp else "NO TEMP"
    print(f"\n{'='*60}")
    print(f"TARGET: {target_col} | HORIZON: {horizon}h | {label}")
    print(f"{'='*60}")
    
    feat_df = build_features(df, target_col, horizon, use_temp=use_temp)
    
    drop_cols = ['datetime', 'NH4', 'NO', 'PO4', 'aeration_ee', 'est_blowers',
                 'DO_1', 'DO_2', 'redox', 'pH', 'sludge_age', 'aerobic_sludge_age',
                 'TS_1', 'TS_2', 'total_flow', 'flow_1', 'flow_2', 'flow_3', 'flow_4',
                 'temp_c']
    
    feature_cols = [c for c in feat_df.columns if c not in drop_cols + ['target', 'datetime']]
    
    mask = feat_df['target'].notna()
    feat_df = feat_df[mask].copy()
    
    valid_features = [c for c in feature_cols if feat_df[c].notna().sum() > len(feat_df) * 0.3]
    feat_df = feat_df.dropna(subset=[f'{target_col}_lag_{horizon}'])
    
    X = feat_df[valid_features]
    y = feat_df['target']
    datetimes = feat_df['datetime']
    
    print(f"Features: {len(valid_features)}")
    print(f"Samples: {len(X)}")
    
    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dt_test = datetimes.iloc[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': 42,
    }
    
    model = lgb.train(
        params, train_data, num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    y_naive = X_test[f'{target_col}_lag_{horizon}'].values
    naive_mask = ~np.isnan(y_naive)
    naive_mae = mean_absolute_error(y_test[naive_mask], y_naive[naive_mask])
    
    print(f"\n--- Results ---")
    print(f"MAE:  {mae:.3f} mg/l")
    print(f"RMSE: {rmse:.3f} mg/l")
    print(f"R²:   {r2:.3f}")
    print(f"Naive MAE: {naive_mae:.3f} mg/l")
    print(f"Skill vs naive: {(1 - mae/naive_mae)*100:.1f}%")
    
    importance = pd.DataFrame({
        'feature': valid_features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    # Show top 15 features with temp features highlighted
    print(f"\nTop 20 features:")
    for _, row in importance.head(20).iterrows():
        marker = " *** TEMP" if 'temp' in row['feature'] or 'cold' in row['feature'] else ""
        print(f"  {row['feature']:45s} {row['importance']:10.0f}{marker}")
    
    return {
        'model': model,
        'target': target_col,
        'horizon': horizon,
        'use_temp': use_temp,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'naive_mae': naive_mae,
        'skill': (1 - mae/naive_mae),
        'feature_importance': importance,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'dt_test': dt_test.values,
    }


# =============================================================================
# 4. RUN COMPARISON
# =============================================================================

if __name__ == '__main__':
    print("Loading and merging data...")
    df = load_and_prepare(
        'input_data/Lir_Labs_6_Monate_KW_20260312_converted_renamed.csv',
        'input_data/braunschweig_temperature.csv'
    )
    
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Run all 8 models: 2 targets × 2 horizons × 2 variants (with/without temp)
    results = {}
    for target in ['NH4', 'NO']:
        for horizon in [4, 10]:
            for use_temp in [False, True]:
                suffix = 'temp' if use_temp else 'notemp'
                key = f'{target}_h{horizon}_{suffix}'
                results[key] = train_and_evaluate(df, target, horizon, use_temp=use_temp)
    
    # =============================================================================
    # 5. COMPARISON TABLE
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"COMPARISON: WITHOUT TEMPERATURE vs WITH TEMPERATURE")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'No Temp MAE':>12} {'+ Temp MAE':>12} {'Δ MAE':>10} {'No Temp R²':>11} {'+ Temp R²':>11} {'Δ R²':>8}")
    print(f"{'-'*76}")
    
    for target in ['NH4', 'NO']:
        for horizon in [4, 10]:
            k1 = f'{target}_h{horizon}_notemp'
            k2 = f'{target}_h{horizon}_temp'
            r1, r2_res = results[k1], results[k2]
            delta_mae = r2_res['mae'] - r1['mae']
            delta_r2 = r2_res['r2'] - r1['r2']
            better = "✓" if delta_mae < 0 else "✗"
            print(f"{target}_h{horizon:<6} {r1['mae']:>12.3f} {r2_res['mae']:>12.3f} {delta_mae:>+10.3f} {better} {r1['r2']:>10.3f} {r2_res['r2']:>11.3f} {delta_r2:>+8.3f}")
    
    # =============================================================================
    # 6. VISUALIZATION
    # =============================================================================
    fig, axes = plt.subplots(4, 1, figsize=(18, 16))
    fig.suptitle('WWTP Forecasting — Temperature-Aware Model Comparison', fontsize=14, fontweight='bold')
    
    configs = [
        ('NH4', 4, 'NH4 — 4h Ahead'),
        ('NH4', 10, 'NH4 — 10h Ahead'),
        ('NO', 4, 'NO — 4h Ahead'),
        ('NO', 10, 'NO — 10h Ahead'),
    ]
    
    for ax, (target, horizon, title) in zip(axes, configs):
        k_nt = f'{target}_h{horizon}_notemp'
        k_t = f'{target}_h{horizon}_temp'
        r_nt = results[k_nt]
        r_t = results[k_t]
        
        dt = pd.to_datetime(r_t['dt_test'])
        
        ax.plot(dt, r_t['y_test'], color='#333', alpha=0.7, linewidth=0.8, label='Actual')
        ax.plot(dt, r_nt['y_pred'], color='#aaa', alpha=0.6, linewidth=0.8, linestyle='--',
                label=f'No Temp (MAE={r_nt["mae"]:.2f})')
        color = '#e74c3c' if 'NO' in target else '#3498db'
        ax.plot(dt, r_t['y_pred'], color=color, alpha=0.85, linewidth=0.9,
                label=f'+ Temp (MAE={r_t["mae"]:.2f})')
        
        delta = r_t['mae'] - r_nt['mae']
        sign = "+" if delta > 0 else ""
        ax.set_title(f'{title}  |  ΔMAE = {sign}{delta:.3f} mg/l  |  R² {r_nt["r2"]:.3f} → {r_t["r2"]:.3f}', fontsize=11)
        ax.set_ylabel('mg/l')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    plt.tight_layout()
    plt.savefig('forecast_temp_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved forecast_temp_comparison.png")
    
    # Save temperature feature importance ranking
    for target in ['NH4', 'NO']:
        for horizon in [4, 10]:
            k = f'{target}_h{horizon}_temp'
            imp = results[k]['feature_importance']
            temp_feats = imp[imp['feature'].str.contains('temp|cold')]
            print(f"\n{target} h{horizon} — Temperature feature ranks:")
            for _, row in temp_feats.iterrows():
                rank = imp.index.get_loc(row.name) + 1
                print(f"  #{rank:3d}: {row['feature']:45s} {row['importance']:10.0f}")
    
    # Save predictions
    for key, r in results.items():
        pred_df = pd.DataFrame({
            'datetime': r['dt_test'],
            'actual': r['y_test'],
            'predicted': r['y_pred'],
        })
        pred_df.to_csv(f'BSE_temp_predictions.csv', index=False)