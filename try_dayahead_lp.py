## ── Cell A ── Imports + constants ────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import linprog
from bsm2_python import BSM2CL
from tqdm import tqdm
from parse_dayahead_prices import parse_dayahead_xml, get_price_for_timestep, price_to_cost
from try_24hour import (
    SO4_MIN, SO4_MAX,
    SNH_LIMIT, SNH_SAFETY, SNH_FORCE_HIGH,
    SNO_LIMIT,
    PRICE_LOW, PRICE_HIGH,
    compute_setpoint,
)

SNH_LP_LIMIT = 1.5   # LP planning threshold — tighter than regulatory SNH_LIMIT=4.0

DAYS      = 300
STEPS     = DAYS * 24 * 60
SIM_START = pd.Timestamp("2024-07-09 01:00")

SNH_IDX = 10
SNO_IDX = 9


## ── Cell B ── Model calibration ──────────────────────────────────────────────

def calibrate_models() -> tuple[float, float, float, float]:
    """
    Fit two linear models from the existing controlled-run data.

    Energy model : ae_kw  ≈  a_coef * so4_set + b_coef
    SNH dynamics : dSNH_h ≈  load_per_hour - nitrif_coeff * so4_h   (hourly)

    Returns
    -------
    a_coef, b_coef, load_per_hour, nitrif_coeff
    """
    df = pd.read_pickle("output_data/sim_controlled_ae.pkl")

    # --- energy model ---
    x = df["so4_set"].values
    A = np.column_stack([x, np.ones(len(x))])
    (a_coef, b_coef), *_ = np.linalg.lstsq(A, df["ae_kw"].values, rcond=None)

    # --- SNH dynamics model (hourly resolution) ---
    df_h = df.resample("h").agg({"s_nh": "mean", "so4_set": "mean"})
    ds   = df_h["s_nh"].diff().dropna().values
    u_h  = df_h["so4_set"].values[:-1]
    B    = np.column_stack([np.ones(len(u_h)), -u_h])
    (load_per_hour, nitrif_coeff), *_ = np.linalg.lstsq(B, ds, rcond=None)

    print(
        f"Energy model : ae_kw = {a_coef:.3f}*u + {b_coef:.3f}\n"
        f"SNH model    : dSNH/h = {load_per_hour:.5f} - {nitrif_coeff:.5f}*u"
    )
    return float(a_coef), float(b_coef), float(load_per_hour), float(nitrif_coeff)


## ── Cell C ── LP helper functions ────────────────────────────────────────────

def _get_next_24h_prices(prices: pd.Series, tomorrow: pd.Timestamp) -> np.ndarray:
    """Extract 24 hourly prices for a calendar day; fills DST gaps with nearest neighbour."""
    idx     = pd.date_range(tomorrow, periods=24, freq="h")
    subset  = prices.reindex(idx, method="nearest", tolerance=pd.Timedelta("2h"))
    return subset.fillna(80.0).values


def plan_day_ahead(
    prices_next_24h: np.ndarray,
    s_nh_now: float,
    a_coef: float,
    b_coef: float,
    load_per_hour: float,
    nitrif_coeff: float,
    snh_lp_limit: float = SNH_LP_LIMIT,
) -> np.ndarray:
    """
    Solve a 24-variable LP for tomorrow's hourly DO setpoints.

    Minimize  sum_h( prices[h] * (a*u_h + b) )
              ≡ minimize  c @ u   where c = a_coef * prices   (b*price is constant)

    Subject to:
        SO4_MIN  ≤ u_h ≤ SO4_MAX   for all h
        SNH_0 + h*load - gamma*cumsum(u)[h-1] ≤ snh_lp_limit   for h = 1..24
        (lower-triangular constraint matrix)

    Returns
    -------
    np.ndarray, shape (24,)  — optimal hourly setpoints
    """
    n      = 24
    c      = a_coef * prices_next_24h
    bounds = [(SO4_MIN, SO4_MAX)] * n

    L     = np.tril(np.ones((n, n)))
    A_ub  = -nitrif_coeff * L
    b_ub  = snh_lp_limit - s_nh_now - np.arange(1, n + 1) * load_per_hour

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 0:
        return res.x.clip(SO4_MIN, SO4_MAX)
    # Infeasible / numerical failure — fall back to midpoint setpoint
    return np.full(n, (SO4_MIN + SO4_MAX) / 2.0)


## ── Cell D ── LP-controlled simulation ───────────────────────────────────────

a_coef, b_coef, load_per_hour, nitrif_coeff = calibrate_models()

prices  = parse_dayahead_xml("input_data/Day-ahead_prices_240709_260311_Hour.xml")
bsm2_cl = BSM2CL()

schedule: dict[tuple, float] = {}   # (date, hour) → precomputed DO setpoint

records = []
for idx, _ in enumerate(tqdm(bsm2_cl.simtime[:STEPS], desc=f"LP day-ahead ({DAYS} days)")):
    current_dt = SIM_START + pd.Timedelta(minutes=idx)
    price      = get_price_for_timestep(prices, current_dt)

    # ── Noon trigger: plan tomorrow's schedule ──────────────────────────────
    if current_dt.hour == 12 and current_dt.minute == 0:
        tomorrow    = (current_dt + pd.Timedelta(days=1)).normalize()
        p24         = _get_next_24h_prices(prices, tomorrow)
        s_nh_plan   = float(bsm2_cl.y_out5[SNH_IDX])
        u_plan      = plan_day_ahead(p24, s_nh_plan, a_coef, b_coef, load_per_hour, nitrif_coeff)
        for h, u_h in enumerate(u_plan):
            schedule[(tomorrow.date(), h)] = float(u_h)

    # ── Setpoint: LP schedule with reactive SNH emergency override ──────────
    s_nh_now = float(bsm2_cl.y_out5[SNH_IDX])
    key      = (current_dt.date(), current_dt.hour)
    lp_used  = key in schedule

    if lp_used:
        sp = schedule[key]
        if s_nh_now >= SNH_FORCE_HIGH:
            sp = SO4_MAX
        elif s_nh_now >= SNH_SAFETY:
            excess = (s_nh_now - SNH_SAFETY) / (SNH_FORCE_HIGH - SNH_SAFETY)
            sp     = sp + excess * (SO4_MAX - sp)
    else:
        # First ~12 h before first noon trigger — fall back to reactive
        sp = compute_setpoint(price, s_nh_now)

    sp = float(np.clip(sp, SO4_MIN, SO4_MAX))
    bsm2_cl.step(idx, sp)

    records.append({
        "dt":      current_dt,
        "ae_kw":   bsm2_cl.ae,
        "price":   price,
        "cost":    price_to_cost(bsm2_cl.ae, price),
        "s_nh":    float(bsm2_cl.y_out5[SNH_IDX]),
        "s_no":    float(bsm2_cl.y_out5[SNO_IDX]),
        "so4_set": sp,
        "lp_used": lp_used,
    })

df_lp = pd.DataFrame(records).set_index("dt")
df_lp.to_pickle("output_data/sim_lp_dayahead.pkl")

print(f"\n{len(df_lp):,} steps saved → output_data/sim_lp_dayahead.pkl")
print(df_lp.describe().round(3))


## ── Cell E ── 5-panel simulation plot ────────────────────────────────────────

fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(f"BSM2 Day-Ahead LP Control — {DAYS} days", fontsize=13, fontweight="bold")

ax1  = axes[0]
ax1r = ax1.twinx()
ax1.step(df_lp.index, df_lp["price"],   lw=0.9, color="darkorange", where="post", label="Price [€/MWh]")
ax1.axhline(PRICE_LOW,  color="darkorange", lw=0.7, ls="--", alpha=0.6, label=f"PRICE_LOW {PRICE_LOW:.0f}")
ax1.axhline(PRICE_HIGH, color="red",        lw=0.7, ls="--", alpha=0.6, label=f"PRICE_HIGH {PRICE_HIGH:.0f}")
ax1r.step(df_lp.index, df_lp["so4_set"], lw=1.2, color="steelblue", where="post", label="SO₄ setpoint")
ax1r.axhline(1.5, color="steelblue", lw=0.7, ls=":", alpha=0.5, label="Baseline 1.5 mg/L")
ax1r.set_ylim(0, SO4_MAX + 0.5)
ax1.set_ylabel("Price [€/MWh]", color="darkorange")
ax1r.set_ylabel("DO Setpoint [mg/L]", color="steelblue")
ax1.set_title("Electricity Price → LP DO Setpoint", fontsize=10)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right", ncol=2)
ax1.grid(True, alpha=0.25)

axes[1].plot(df_lp.index, df_lp["ae_kw"], lw=0.7, color="steelblue")
axes[1].set_ylabel("Aeration\nEnergy [kW]")
axes[1].set_title("Aeration Energy", fontsize=10)
axes[1].grid(True, alpha=0.25)

axes[2].fill_between(df_lp.index, df_lp["cost"], color="crimson", alpha=0.6, lw=0)
axes[2].set_ylabel("Cost\n[€/min]")
axes[2].annotate(
    f"Total: €{df_lp['cost'].sum():,.2f}",
    xy=(0.01, 0.88), xycoords="axes fraction",
    fontsize=9, color="darkred",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
axes[2].set_title("Cost per Timestep", fontsize=10)
axes[2].grid(True, alpha=0.25)

axes[3].plot(df_lp.index, df_lp["s_nh"], lw=0.7, color="seagreen", label="S_NH")
axes[3].axhline(SNH_LIMIT,      color="red",    lw=1.0, ls="--", label=f"Limit {SNH_LIMIT} mg/L")
axes[3].axhline(SNH_SAFETY,     color="orange", lw=0.8, ls=":",  label=f"Safety {SNH_SAFETY} mg/L")
axes[3].axhline(SNH_FORCE_HIGH, color="tomato", lw=0.8, ls=":",  label=f"Force-high {SNH_FORCE_HIGH} mg/L")
axes[3].axhline(SNH_LP_LIMIT,   color="navy",   lw=0.8, ls="--", alpha=0.6, label=f"LP limit {SNH_LP_LIMIT} mg/L")
axes[3].set_ylabel("Ammonium\n[mg/L]")
axes[3].set_title("Effluent Ammonium (Compliance)", fontsize=10)
axes[3].legend(fontsize=7, loc="upper right")
axes[3].grid(True, alpha=0.25)

axes[4].plot(df_lp.index, df_lp["s_no"], lw=0.7, color="mediumpurple", label="S_NO")
axes[4].axhline(SNO_LIMIT, color="red", lw=1.0, ls="--", label=f"Limit {SNO_LIMIT} mg/L")
axes[4].set_ylabel("Nitrate\n[mg/L]")
axes[4].set_xlabel("Date")
axes[4].set_title("Effluent Nitrate (Compliance)", fontsize=10)
axes[4].legend(fontsize=7, loc="upper right")
axes[4].grid(True, alpha=0.25)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.tight_layout()
plt.savefig(f"output_data/lp_dayahead_{DAYS}d.png", dpi=150)
print(f"Plot saved → output_data/lp_dayahead_{DAYS}d.png")


## ── Cell F ── Comparison vs reactive baseline ────────────────────────────────

df_rx = pd.read_pickle("output_data/sim_controlled_ae.pkl")

saving_eur = df_rx["cost"].sum() - df_lp["cost"].sum()
saving_pct = saving_eur / df_rx["cost"].sum() * 100
print(f"\nLP saves EUR {saving_eur:,.2f} ({saving_pct:.2f}%) over reactive across {DAYS} days")

fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig2.suptitle(f"LP Day-Ahead vs Reactive Controller — {DAYS} days", fontsize=13, fontweight="bold")

# Panel 1: DO setpoints overlaid
axes2[0].step(df_lp.index, df_lp["so4_set"], lw=0.7, color="steelblue", where="post", label="LP day-ahead", alpha=0.8)
axes2[0].step(df_rx.index, df_rx["so4_set"], lw=0.7, color="darkorange", where="post", label="Reactive", alpha=0.6)
axes2[0].set_ylabel("DO Setpoint [mg/L]")
axes2[0].set_title("DO Setpoint Comparison", fontsize=10)
axes2[0].legend(fontsize=8)
axes2[0].grid(True, alpha=0.25)

# Panel 2: Cumulative cost
axes2[1].plot(df_lp.index, df_lp["cost"].cumsum(), lw=1.0, color="steelblue", label="LP day-ahead")
axes2[1].plot(df_rx.index, df_rx["cost"].cumsum(), lw=1.0, color="darkorange", label="Reactive")
axes2[1].set_ylabel("Cumulative Cost [€]")
axes2[1].set_title("Cumulative Energy Cost", fontsize=10)
axes2[1].legend(fontsize=8)
axes2[1].annotate(
    f"LP saves €{saving_eur:,.0f} ({saving_pct:.1f}%)",
    xy=(0.01, 0.88), xycoords="axes fraction",
    fontsize=9, color="steelblue",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
axes2[1].grid(True, alpha=0.25)

# Panel 3: SNH compliance both strategies
axes2[2].plot(df_lp.index, df_lp["s_nh"], lw=0.7, color="steelblue", label="LP day-ahead", alpha=0.8)
axes2[2].plot(df_rx.index, df_rx["s_nh"], lw=0.7, color="darkorange", label="Reactive", alpha=0.6)
axes2[2].axhline(SNH_LIMIT,    color="red",  lw=1.0, ls="--", label=f"Limit {SNH_LIMIT} mg/L")
axes2[2].axhline(SNH_LP_LIMIT, color="navy", lw=0.8, ls="--", alpha=0.6, label=f"LP limit {SNH_LP_LIMIT} mg/L")
axes2[2].set_ylabel("Ammonium [mg/L]")
axes2[2].set_xlabel("Date")
axes2[2].set_title("Effluent Ammonium — Compliance Check", fontsize=10)
axes2[2].legend(fontsize=8)
axes2[2].grid(True, alpha=0.25)

for ax in axes2:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))

plt.tight_layout()
plt.savefig("output_data/lp_vs_reactive_comparison.png", dpi=150)
print("Comparison plot saved → output_data/lp_vs_reactive_comparison.png")
