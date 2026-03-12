"""
Day-ahead price XML parser → pandas Series (€/MWh, hourly index)
Filters to Germany/Luxembourg component only, ignores all others.
No timezone awareness — naive datetimes, simpler to work with.

Usage:
    prices = parse_dayahead_xml("prices.xml")
    price   = get_price_for_timestep(prices, current_dt)
    cost    = price_to_cost(bsm2_cl.ae, price)
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


TARGET_COMPONENT = "Germany/Luxembourg"


def parse_dayahead_xml(filepath: str | Path) -> pd.Series:
    """
    Parse day-ahead XML → pandas Series.
    Only reads the Germany/Luxembourg component. All others are ignored.

    Returns
    -------
    pd.Series
        Index : pd.DatetimeIndex, naive (no timezone), hourly
        Values : float, price in €/MWh
        Name   : 'price_eur_per_mwh'
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    records = []
    for component in root.iter("Component"):
        if component.findtext("Component_name", "").strip() != TARGET_COMPONENT:
            continue
        for vd in component.iter("Value_detail"):
            raw = vd.findtext("Value").strip()
            if raw == "-":
                continue  # missing data point — skip
            date_str = vd.findtext("Date").strip()
            time_str = vd.findtext("Time_of_day").strip()
            value    = float(raw)
            dt       = pd.to_datetime(f"{date_str} {time_str}", format="%b %d, %Y %I:%M %p")
            records.append((dt, value))

    if not records:
        raise ValueError(
            f"No data found for '{TARGET_COMPONENT}'. "
            f"Run list_components(filepath) to see what's in your XML."
        )

    index, values = zip(*records)
    series = pd.Series(
        data=list(values),
        index=pd.DatetimeIndex(list(index)),
        name="price_eur_per_mwh",
        dtype=float,
    ).sort_index()

    # Drop duplicate timestamps (DST clock-back creates two entries for the same hour)
    series = series[~series.index.duplicated(keep="first")]

    return series


def list_components(filepath: str | Path) -> list[str]:
    """List all component names in the XML — useful for debugging."""
    tree = ET.parse(filepath)
    return [c.findtext("Component_name", "").strip() for c in tree.getroot().iter("Component")]


def get_price_for_timestep(
    prices: pd.Series,
    current_dt: pd.Timestamp,
    fallback: float = 80.0,
) -> float:
    """
    Get price for a simulation timestep.
    BSM2CL runs at 1-minute resolution; prices are hourly.
    Floors current_dt to the hour and looks up the matching price.

    Parameters
    ----------
    prices      : output of parse_dayahead_xml()
    current_dt  : naive pd.Timestamp aligned to simulation time
    fallback    : price if current_dt is outside data range, default 80 €/MWh

    Returns
    -------
    float, €/MWh
    """
    hour_dt = current_dt.floor("h")
    try:
        return float(prices.loc[hour_dt])
    except KeyError:
        idx = prices.index.get_indexer([hour_dt], method="nearest")[0]
        return float(prices.iloc[idx]) if idx != -1 else fallback


def price_to_cost(power_kw: float, price_eur_per_mwh: float, timestep_minutes: float = 1.0) -> float:
    """
    Convert aeration power to € cost for one simulation timestep.

    Parameters
    ----------
    power_kw          : self.ae from BSM2, in kW
    price_eur_per_mwh : €/MWh from get_price_for_timestep()
    timestep_minutes  : 1.0 for BSM2CL

    Returns
    -------
    float, € for this timestep
    """
    energy_kwh = power_kw * (timestep_minutes / 60.0)
    return energy_kwh * (price_eur_per_mwh / 1000.0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sample = """<?xml version="1.0" encoding="UTF-8"?>
<Categories>
  <Category>
    <Component><Component_name>Germany/Luxembourg</Component_name><Unit>EUR/MWh</Unit>
      <Values>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>1:00 AM</Time_of_day><Value>56.01</Value></Value_detail>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>2:00 AM</Time_of_day><Value>57.30</Value></Value_detail>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>3:00 AM</Time_of_day><Value>60.00</Value></Value_detail>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>7:00 AM</Time_of_day><Value>88.53</Value></Value_detail>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>11:00 AM</Time_of_day><Value>0.97</Value></Value_detail>
      </Values>
    </Component>
    <Component><Component_name>&#8709; DE/LU neighbours</Component_name><Unit>EUR/MWh</Unit>
      <Values>
        <Value_detail><Date>Feb 27, 2026</Date><Time_of_day>1:00 AM</Time_of_day><Value>999.99</Value></Value_detail>
      </Values>
    </Component>
  </Category>
</Categories>"""
        tmp = Path("/tmp/sample_prices.xml")
        tmp.write_text(sample, encoding="utf-8")
        filepath = tmp
    else:
        filepath = Path(sys.argv[1])

    print(f"Components in file: {list_components(filepath)}")
    print()
    prices = parse_dayahead_xml(filepath)
    print(prices.to_string())
    print(f"\nEntries : {len(prices):,}")
    print(f"Range   : {prices.index[0]} → {prices.index[-1]}")
    print(f"Min     : {prices.min():.2f} €/MWh")
    print(f"Max     : {prices.max():.2f} €/MWh")
    print(f"\nSample cost — 50 kW for 1 min at peak price:")
    print(f"  €{price_to_cost(50.0, prices.max()):.5f}")