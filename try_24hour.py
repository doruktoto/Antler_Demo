# Min: 50 €/MWh Max: 100 €/MWh
## ── Cell A ── Control parameters + compute_setpoint ──────────────────────────

import numpy as np

# DO setpoint rails [mg/L]
SO4_MIN = 0.2
SO4_MAX = 2.0

# Price thresholds [€/MWh]
PRICE_LOW  =  50.0   # below → max aeration (cheap energy)
PRICE_HIGH = 90.0   # above → min aeration (expensive energy)

# NH compliance & safety [mg/L]
SNH_LIMIT      = 4.0   # regulatory hard limit
SNH_SAFETY     = 3.2   # 80% of limit — start proportional ramp
SNH_FORCE_HIGH = 3.6   # 90% of limit — emergency: force max DO

# Nitrate compliance [mg/L]
SNO_LIMIT = 10.0


def compute_setpoint(price: float, s_nh: float) -> float:
    """
    Price-responsive DO setpoint with NH safety cascade.

    Low price  → high DO setpoint (build nitrification buffer).
    High price → low DO setpoint (save energy, spend buffer).
    NH override: if ammonium approaches compliance limit, force aeration
    regardless of price.

    Parameters
    ----------
    price : float   Electricity price [€/MWh]
    s_nh  : float   Effluent ammonium from previous timestep [mg/L]

    Returns
    -------
    float  DO setpoint for bsm2_cl.step() [mg/L]
    """
    # 1. Price-based base setpoint (linear interpolation)
    if price <= PRICE_LOW:
        sp = SO4_MAX
    elif price >= PRICE_HIGH:
        sp = SO4_MIN
    else:
        frac = (price - PRICE_LOW) / (PRICE_HIGH - PRICE_LOW)
        sp = SO4_MAX - frac * (SO4_MAX - SO4_MIN)

    # 2. NH safety cascade (overrides price signal upward)
    if s_nh >= SNH_FORCE_HIGH:
        sp = SO4_MAX                               # emergency: ignore price
    elif s_nh >= SNH_SAFETY:
        excess = (s_nh - SNH_SAFETY) / (SNH_FORCE_HIGH - SNH_SAFETY)
        sp = sp + excess * (SO4_MAX - sp)          # proportional ramp

    return float(np.clip(sp, SO4_MIN, SO4_MAX))




# ── Cell B ── Controlled simulation + plot ────────────────────────────────────

from bsm2_python import BSM2CL
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from parse_dayahead_prices import parse_dayahead_xml, get_price_for_timestep, price_to_cost

DAYS      = 300
STEPS     = DAYS * 24 * 60
SIM_START = pd.Timestamp("2024-07-09 01:00")

SNH_IDX = 10
SNO_IDX = 9

prices  = parse_dayahead_xml("input_data/Day-ahead_prices_240709_260311_Hour.xml")
bsm2_cl = BSM2CL()

records = []
for idx, _ in enumerate(tqdm(bsm2_cl.simtime[:STEPS], desc=f"Controlled ({DAYS} days)")):
    current_dt = SIM_START + pd.Timedelta(minutes=idx)
    price      = get_price_for_timestep(prices, current_dt)

    # Read state BEFORE step (measure → compute → actuate)
    s_nh_now = float(bsm2_cl.y_out5[SNH_IDX])
    sp       = compute_setpoint(price, s_nh_now)

    bsm2_cl.step(idx, sp)

    records.append({
        "dt":      current_dt,
        "ae_kw":   bsm2_cl.ae,
        "price":   price,
        "cost":    price_to_cost(bsm2_cl.ae, price),
        "s_nh":    float(bsm2_cl.y_out5[SNH_IDX]),
        "s_no":    float(bsm2_cl.y_out5[SNO_IDX]),
        "so4_set": sp,
    })

df_ctrl = pd.DataFrame(records).set_index("dt")
df_ctrl.to_pickle("output_data/sim_controlled_ae.pkl")

print(f"\n{len(df_ctrl):,} steps saved → /output_data/sim_controlled_ae.pkl")
print(df_ctrl.describe().round(3))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
fig.suptitle(f"BSM2 Price-Responsive Control — {DAYS} days", fontsize=13, fontweight="bold")

# Panel 1: Price (left axis) + DO setpoint (right axis)
ax1 = axes[0]
ax1r = ax1.twinx()
ax1.step(df_ctrl.index, df_ctrl["price"],   lw=0.9, color="darkorange", where="post", label="Price [€/MWh]")
ax1.axhline(PRICE_LOW,  color="darkorange", lw=0.7, ls="--", alpha=0.6, label=f"PRICE_LOW {PRICE_LOW:.0f}")
ax1.axhline(PRICE_HIGH, color="red",        lw=0.7, ls="--", alpha=0.6, label=f"PRICE_HIGH {PRICE_HIGH:.0f}")
ax1r.step(df_ctrl.index, df_ctrl["so4_set"], lw=1.2, color="steelblue", where="post", label="SO₄ setpoint")
ax1r.axhline(1.5, color="steelblue", lw=0.7, ls=":", alpha=0.5, label="Baseline 1.5 mg/L")
ax1r.set_ylim(0, SO4_MAX + 0.5)
ax1.set_ylabel("Price [€/MWh]", color="darkorange")
ax1r.set_ylabel("DO Setpoint [mg/L]", color="steelblue")
ax1.set_title("Electricity Price → DO Setpoint Response", fontsize=10)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="upper right", ncol=2)
ax1.grid(True, alpha=0.25)

# Panel 2: Aeration energy
axes[1].plot(df_ctrl.index, df_ctrl["ae_kw"], lw=0.7, color="steelblue")
axes[1].set_ylabel("Aeration\nEnergy [kW]")
axes[1].set_title("Aeration Energy", fontsize=10)
axes[1].grid(True, alpha=0.25)

# Panel 3: Cost per timestep
axes[2].fill_between(df_ctrl.index, df_ctrl["cost"], color="crimson", alpha=0.6, lw=0)
axes[2].set_ylabel("Cost\n[€/min]")
axes[2].annotate(
    f"Total: €{df_ctrl['cost'].sum():,.2f}",
    xy=(0.01, 0.88), xycoords="axes fraction",
    fontsize=9, color="darkred",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
)
axes[2].set_title("Cost per Timestep", fontsize=10)
axes[2].grid(True, alpha=0.25)

# Panel 4: Ammonium
axes[3].plot(df_ctrl.index, df_ctrl["s_nh"], lw=0.7, color="seagreen", label="S_NH")
axes[3].axhline(SNH_LIMIT,      color="red",     lw=1.0, ls="--", label=f"Limit {SNH_LIMIT} mg/L")
axes[3].axhline(SNH_SAFETY,     color="orange",  lw=0.8, ls=":",  label=f"Safety {SNH_SAFETY} mg/L")
axes[3].axhline(SNH_FORCE_HIGH, color="tomato",  lw=0.8, ls=":",  label=f"Force-high {SNH_FORCE_HIGH} mg/L")
axes[3].set_ylabel("Ammonium\n[mg/L]")
axes[3].set_title("Effluent Ammonium (Compliance)", fontsize=10)
axes[3].legend(fontsize=7, loc="upper right")
axes[3].grid(True, alpha=0.25)

# Panel 5: Nitrate
axes[4].plot(df_ctrl.index, df_ctrl["s_no"], lw=0.7, color="mediumpurple", label="S_NO")
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
plt.savefig(f"output_data/controlled_{DAYS}d.png", dpi=150)
print(f"\nPlot saved → controlled_{DAYS}d.png")