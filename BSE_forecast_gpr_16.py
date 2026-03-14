"""
WWTP Forecasting — Gaussian Process Regression (Temperature-Aware)
==================================================================
GPR version of BSE_forecast_temp.py.
Key additions over LightGBM baseline:
  - Calibrated uncertainty estimates (prediction intervals)
  - RBF + WhiteKernel (captures smooth trends + noise floor)
  - Sparse subsampling to keep GP tractable (O(n³) cost)
  - Feature selection by correlation to reduce dimensionality
Targets: NH4, NO  |  Horizons: 4h, 10h, 16h
"""

import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# GPR subsample cap — GP is O(n³); keep training tractable
GPR_MAX_TRAIN_SAMPLES = 3000
# Number of top correlated features to keep per model
GPR_N_FEATURES = 20
# Horizons to evaluate
HORIZONS = [4, 10]

# =============================================================================
# 1. LOAD AND MERGE  (identical to BSE_forecast_temp.py)
# =============================================================================

def load_and_prepare(wwtp_path, temp_path):
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

    temp = pd.read_csv(temp_path)
    temp['datetime'] = pd.to_datetime(temp['datetime'])
    temp = temp.sort_values('datetime').reset_index(drop=True)
    temp = temp.rename(columns={'temperature_2m': 'temp_c'})

    df = df.merge(temp[['datetime', 'temp_c']], on='datetime', how='left')
    matched = df['temp_c'].notna().sum()
    print(f"Temperature merge: {matched}/{len(df)} rows matched ({matched/len(df)*100:.1f}%)")
    df['temp_c'] = df['temp_c'].ffill().bfill()

    keep_cols = ['datetime', 'NH4', 'NO', 'PO4', 'aeration_ee', 'est_blowers',
                 'DO_1', 'DO_2', 'redox', 'pH', 'sludge_age', 'aerobic_sludge_age',
                 'TS_1', 'TS_2', 'total_flow', 'flow_1', 'flow_2', 'flow_3', 'flow_4',
                 'temp_c']
    df = df[keep_cols].copy()
    print(f"Temperature range: {df['temp_c'].min():.1f}°C to {df['temp_c'].max():.1f}°C")
    return df


# =============================================================================
# 2. FEATURE ENGINEERING  (identical to BSE_forecast_temp.py)
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

    target_lags = [horizon + i for i in [0, 1, 2, 3, 4, 6, 12, 24, 48, 168]]
    for lag in target_lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    for w in [4, 12, 24, 48, 168]:
        df[f'{target_col}_rmean_{w}_h{horizon}'] = df[target_col].shift(horizon).rolling(window=w, min_periods=1).mean()
        df[f'{target_col}_rstd_{w}_h{horizon}'] = df[target_col].shift(horizon).rolling(window=w, min_periods=1).std()

    for p in [1, 4, 24]:
        df[f'{target_col}_diff_{p}_h{horizon}'] = df[target_col].shift(horizon).diff(p)

    other = 'NO' if target_col == 'NH4' else 'NH4'
    for lag in [horizon, horizon+1, horizon+4, horizon+12, horizon+24]:
        df[f'{other}_lag_{lag}'] = df[other].shift(lag)
    df[f'{other}_rmean_24_h{horizon}'] = df[other].shift(horizon).rolling(24, min_periods=1).mean()

    for lag in [horizon, horizon+1, horizon+4, horizon+12, horizon+24]:
        df[f'aeration_lag_{lag}'] = df['aeration_ee'].shift(lag)
        df[f'blowers_lag_{lag}'] = df['est_blowers'].shift(lag)
    df[f'aeration_rmean_12_h{horizon}'] = df['aeration_ee'].shift(horizon).rolling(12, min_periods=1).mean()
    df[f'aeration_rmean_24_h{horizon}'] = df['aeration_ee'].shift(horizon).rolling(24, min_periods=1).mean()

    for lag in [horizon, horizon+4, horizon+12, horizon+24]:
        df[f'total_flow_lag_{lag}'] = df['total_flow'].shift(lag)
    df[f'total_flow_rmean_24_h{horizon}'] = df['total_flow'].shift(horizon).rolling(24, min_periods=1).mean()
    df[f'total_flow_rstd_24_h{horizon}'] = df['total_flow'].shift(horizon).rolling(24, min_periods=1).std()

    for var in ['redox', 'pH', 'PO4', 'sludge_age', 'TS_1']:
        for lag in [horizon, horizon+12, horizon+24]:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)

    if df['DO_1'].notna().sum() > 100:
        for lag in [horizon, horizon+4, horizon+12]:
            df[f'DO_1_lag_{lag}'] = df['DO_1'].shift(lag)

    if use_temp and 'temp_c' in df.columns:
        for lag in [horizon, horizon+4, horizon+12, horizon+24]:
            df[f'temp_lag_{lag}'] = df['temp_c'].shift(lag)
        df[f'temp_rmean_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean()
        df[f'temp_rmean_72_h{horizon}'] = df['temp_c'].shift(horizon).rolling(72, min_periods=1).mean()
        df[f'temp_rmean_168_h{horizon}'] = df['temp_c'].shift(horizon).rolling(168, min_periods=1).mean()
        df[f'temp_rstd_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).std()
        df[f'temp_diff_24_h{horizon}'] = df['temp_c'].shift(horizon).diff(24)
        df[f'temp_diff_72_h{horizon}'] = df['temp_c'].shift(horizon).diff(72)
        df[f'temp_diff_168_h{horizon}'] = df['temp_c'].shift(horizon).diff(168)
        df[f'temp_min_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).min()
        df[f'temp_max_24_h{horizon}'] = df['temp_c'].shift(horizon).rolling(24, min_periods=1).max()
        df[f'temp_x_aeration_h{horizon}'] = (
            df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean() *
            df['aeration_ee'].shift(horizon).rolling(24, min_periods=1).mean()
        )
        df[f'temp_below_zero_h{horizon}'] = (
            df['temp_c'].shift(horizon).rolling(24, min_periods=1).mean() < 0
        ).astype(int)
        df[f'cold_hours_72_h{horizon}'] = (
            (df['temp_c'].shift(horizon) < 5).rolling(72, min_periods=1).sum()
        )

    df['target'] = df[target_col].shift(-horizon)
    return df


# =============================================================================
# 3. GPR-SPECIFIC HELPERS
# =============================================================================

def select_features_by_correlation(X_train, y_train, n_features=GPR_N_FEATURES):
    """Pick top n features by absolute Pearson correlation with the target."""
    corr = X_train.corrwith(pd.Series(y_train, index=X_train.index)).abs()
    corr = corr.dropna().sort_values(ascending=False)
    selected = corr.head(n_features).index.tolist()
    return selected


def subsample_training(X_train, y_train, max_samples=GPR_MAX_TRAIN_SAMPLES, random_state=42):
    """Randomly subsample when training set exceeds max_samples."""
    if len(X_train) <= max_samples:
        return X_train, y_train
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_train), size=max_samples, replace=False)
    idx.sort()
    return X_train.iloc[idx], y_train.iloc[idx]


def build_kernel():
    """
    RBF captures smooth covariance across the feature space.
    WhiteKernel accounts for independent measurement noise.
    ConstantKernel scales the overall signal amplitude.
    Upper RBF bound raised to 1e3 so the optimiser doesn't get pinned
    at the ceiling when training on larger (3000+) datasets.
    """
    amplitude = C(1.0, (1e-3, 1e3))
    rbf = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    noise = WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10.0))
    return amplitude * rbf + noise


# =============================================================================
# 4. TRAIN / EVALUATE
# =============================================================================

def train_and_evaluate_gpr(df, target_col, horizon, test_frac=0.15):
    print(f"\n{'='*60}")
    print(f"GPR | TARGET: {target_col} | HORIZON: {horizon}h")
    print(f"{'='*60}")

    feat_df = build_features(df, target_col, horizon, use_temp=True)

    drop_cols = ['datetime', 'NH4', 'NO', 'PO4', 'aeration_ee', 'est_blowers',
                 'DO_1', 'DO_2', 'redox', 'pH', 'sludge_age', 'aerobic_sludge_age',
                 'TS_1', 'TS_2', 'total_flow', 'flow_1', 'flow_2', 'flow_3', 'flow_4',
                 'temp_c']

    feature_cols = [c for c in feat_df.columns if c not in drop_cols + ['target']]

    mask = feat_df['target'].notna()
    feat_df = feat_df[mask].copy()

    valid_features = [c for c in feature_cols if feat_df[c].notna().sum() > len(feat_df) * 0.3]
    feat_df = feat_df.dropna(subset=[f'{target_col}_lag_{horizon}'])

    X_all = feat_df[valid_features].fillna(0)
    y_all = feat_df['target']
    dt_all = feat_df['datetime']

    split_idx = int(len(X_all) * (1 - test_frac))
    X_train_full = X_all.iloc[:split_idx]
    y_train_full = y_all.iloc[:split_idx]
    X_test = X_all.iloc[split_idx:]
    y_test = y_all.iloc[split_idx:]
    dt_test = dt_all.iloc[split_idx:]

    # Feature selection on training data
    selected_features = select_features_by_correlation(X_train_full, y_train_full,
                                                        n_features=GPR_N_FEATURES)
    X_train_sel = X_train_full[selected_features]
    X_test_sel = X_test[selected_features]

    print(f"Features after selection: {len(selected_features)}")
    print(f"  Top 5: {selected_features[:5]}")

    # Subsample for tractability
    X_train_sub, y_train_sub = subsample_training(X_train_sel, y_train_full)
    print(f"Training samples (subsampled): {len(X_train_sub)} / {len(X_train_sel)}")
    print(f"Test samples: {len(X_test_sel)}")

    # Scale features and target (GPR is sensitive to scale)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_sub)
    X_test_scaled = x_scaler.transform(X_test_sel)

    y_mean = y_train_sub.mean()
    y_std = y_train_sub.std()
    y_train_norm = (y_train_sub - y_mean) / y_std

    # Fit GPR
    kernel = build_kernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=8,   # more restarts needed to escape local optima at large n
        normalize_y=False,        # we normalise manually to control the scale-back
        alpha=1e-2,               # larger jitter for numerical stability at 3000+ samples
        random_state=42,
    )
    print("Fitting GPR (this may take a moment)...")
    gpr.fit(X_train_scaled, y_train_norm)
    print(f"Optimised kernel: {gpr.kernel_}")

    # Predict with uncertainty
    y_pred_norm, y_std_norm = gpr.predict(X_test_scaled, return_std=True)

    # Scale back
    y_pred = y_pred_norm * y_std + y_mean
    y_sigma = y_std_norm * y_std          # predictive std in original units

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    y_naive = X_test[f'{target_col}_lag_{horizon}'].values
    naive_mask = ~np.isnan(y_naive)
    naive_mae = mean_absolute_error(y_test[naive_mask], y_naive[naive_mask])

    # Calibration: fraction of actuals inside 90% PI
    z90 = 1.645
    inside_90 = np.mean(
        (y_test.values >= y_pred - z90 * y_sigma) &
        (y_test.values <= y_pred + z90 * y_sigma)
    )
    # Average 90% PI width
    pi_width_90 = np.mean(2 * z90 * y_sigma)

    print(f"\n--- Results ---")
    print(f"MAE:  {mae:.3f} mg/l")
    print(f"RMSE: {rmse:.3f} mg/l")
    print(f"R²:   {r2:.3f}")
    print(f"Naive MAE: {naive_mae:.3f} mg/l")
    print(f"Skill vs naive: {(1 - mae/naive_mae)*100:.1f}%")
    print(f"\n--- Uncertainty Calibration ---")
    print(f"90% PI coverage: {inside_90*100:.1f}%  (ideal: 90%)")
    print(f"Avg 90% PI width: {pi_width_90:.3f} mg/l")

    return {
        'model': gpr,
        'target': target_col,
        'horizon': horizon,
        'selected_features': selected_features,
        'x_scaler': x_scaler,
        'y_mean': y_mean, 'y_std': y_std,
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'naive_mae': naive_mae,
        'skill': (1 - mae / naive_mae),
        'inside_90': inside_90,
        'pi_width_90': pi_width_90,
        'y_test': y_test.values,
        'y_pred': y_pred,
        'y_sigma': y_sigma,
        'dt_test': dt_test.values,
    }


# =============================================================================
# 5. VISUALISATION
# =============================================================================

def plot_results(results, targets, horizons):
    n_plots = len(targets) * len(horizons)
    fig, axes = plt.subplots(n_plots, 1, figsize=(18, 5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('WWTP GPR Forecasting — Temperature-Aware with Uncertainty Bands',
                 fontsize=14, fontweight='bold')

    colors = {'NH4': '#3498db', 'NO': '#e74c3c'}
    ax_idx = 0
    for target in targets:
        for horizon in horizons:
            key = f'{target}_h{horizon}'
            if key not in results:
                continue
            r = results[key]
            ax = axes[ax_idx]
            ax_idx += 1

            dt = pd.to_datetime(r['dt_test'])
            y_pred = r['y_pred']
            y_sigma = r['y_sigma']
            color = colors.get(target, '#2ecc71')

            ax.plot(dt, r['y_test'], color='#333', alpha=0.7, linewidth=0.8, label='Actual', zorder=3)
            ax.plot(dt, y_pred, color=color, alpha=0.9, linewidth=0.9,
                    label=f'GPR mean (MAE={r["mae"]:.2f})', zorder=4)

            # 90% prediction interval
            ax.fill_between(
                dt,
                y_pred - 1.645 * y_sigma,
                y_pred + 1.645 * y_sigma,
                color=color, alpha=0.15, label=f'90% PI (cov={r["inside_90"]*100:.0f}%)'
            )
            # 68% prediction interval
            ax.fill_between(
                dt,
                y_pred - y_sigma,
                y_pred + y_sigma,
                color=color, alpha=0.25, label='68% PI'
            )

            ax.set_title(
                f'{target} — {horizon}h ahead  |  '
                f'MAE={r["mae"]:.3f}  R²={r["r2"]:.3f}  '
                f'Skill={r["skill"]*100:.1f}%  '
                f'90% PI coverage={r["inside_90"]*100:.1f}%',
                fontsize=11
            )
            ax.set_ylabel('mg/l')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    plt.tight_layout()
    out_path = 'output_data/GPR/BSE_gpr16_forecast_3000.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_path}")


def plot_calibration(results, targets, horizons):
    """Q-Q style calibration: empirical vs nominal coverage across quantile levels."""
    n_plots = len(targets) * len(horizons)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    nominal_levels = np.linspace(0.05, 0.95, 19)
    z_vals = {p: float(np.abs(np.percentile(np.random.randn(100_000), 50 - p*50))) for p in nominal_levels}

    ax_idx = 0
    for target in targets:
        for horizon in horizons:
            key = f'{target}_h{horizon}'
            if key not in results:
                continue
            r = results[key]
            ax = axes[ax_idx]
            ax_idx += 1

            empirical = []
            for p in nominal_levels:
                from scipy.stats import norm
                z = norm.ppf((1 + p) / 2)
                inside = np.mean(
                    (r['y_test'] >= r['y_pred'] - z * r['y_sigma']) &
                    (r['y_test'] <= r['y_pred'] + z * r['y_sigma'])
                )
                empirical.append(inside)

            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
            ax.plot(nominal_levels, empirical, 'o-', color='steelblue', linewidth=1.5,
                    markersize=4, label='GPR')
            ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.04, color='grey')
            ax.set_xlabel('Nominal coverage')
            ax.set_ylabel('Empirical coverage')
            ax.set_title(f'{target} h{horizon} — Calibration')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

    plt.suptitle('GPR Uncertainty Calibration', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = 'output_data/GPR/BSE_gpr16_calibration_3000.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved {out_path}")


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == '__main__':
    print("Loading and merging data...")
    df = load_and_prepare(
        'input_data/Lir_Labs_6_Monate_KW_20260312_converted_renamed.csv',
        'input_data/braunschweig_temperature.csv'
    )
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    targets = ['NH4', 'NO']
    results = {}

    for target in targets:
        for horizon in HORIZONS:
            key = f'{target}_h{horizon}'
            results[key] = train_and_evaluate_gpr(df, target, horizon)

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"GPR SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Skill%':>8} "
          f"{'90%PI cov':>10} {'PI width':>10}")
    print(f"{'-'*70}")
    for target in targets:
        for horizon in HORIZONS:
            key = f'{target}_h{horizon}'
            if key not in results:
                continue
            r = results[key]
            print(f"{key:<12} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['r2']:>8.3f} "
                  f"{r['skill']*100:>7.1f}% {r['inside_90']*100:>9.1f}% "
                  f"{r['pi_width_90']:>10.3f}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    plot_results(results, targets, HORIZONS)
    plot_calibration(results, targets, HORIZONS)

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------
    all_preds = []
    for key, r in results.items():
        pred_df = pd.DataFrame({
            'datetime': r['dt_test'],
            'target': r['target'],
            'horizon': r['horizon'],
            'actual': r['y_test'],
            'predicted': r['y_pred'],
            'pred_std': r['y_sigma'],
            'pi90_lower': r['y_pred'] - 1.645 * r['y_sigma'],
            'pi90_upper': r['y_pred'] + 1.645 * r['y_sigma'],
        })
        all_preds.append(pred_df)

    pd.concat(all_preds, ignore_index=True).to_csv('output_data/GPR/BSE_gpr16_predictions_3000.csv', index=False)
    print("\nSaved output_data/GPR/BSE_gpr16_predictions_3000.csv")
