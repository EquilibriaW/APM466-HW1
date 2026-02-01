#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FACE_VALUE = 100.0
COUPON_FREQ = 2  # semi-annual


# Recommended 10-bond ladder for a 0–5Y curve that *includes* one bond beyond 5Y
# to avoid extrapolation at the 5Y node.
# (ISINs are from the provided `bonds_final_price_matrix.csv`.)
DEFAULT_SELECTED_ISINS: list[str] = [
    "CA135087R556",  # CAN 4.0 May 26
    "CA135087R978",  # CAN 4.0 Aug 26
    "CA135087L930",  # CAN 1.0 Sep 26
    "CA135087M847",  # CAN 1.25 Mar 27
    "CA135087N837",  # CAN 2.75 Sep 27
    "CA135087T958",  # CAN 2.25 Feb 28
    "CA135087Q491",  # CAN 3.25 Sep 28
    "CA135087Q988",  # CAN 4.0 Mar 29
    "CA135087S471",  # CAN 2.75 Mar 30
    "CA135087T792",  # CAN 2.75 Mar 31
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="APM466 yield/spot/forward analysis (v2).")

    p.add_argument(
        "--data",
        default="bonds_final_price_matrix.csv",
        help="Path to bonds_final_price_matrix.csv (wide format).",
    )
    p.add_argument("--outdir", default="analysis_outputs", help="Output directory.")

    # Selection controls
    p.add_argument(
        "--select-count",
        type=int,
        default=10,
        help="Number of bonds to auto-select (ignored if --isins/--fixed-isins is used).",
    )
    p.add_argument(
        "--max-years",
        type=float,
        default=5.0,
        help="Max maturity (years) for AUTO selection only.",
    )
    p.add_argument(
        "--fixed-isins",
        action="store_true",
        help="Use the built-in recommended 10-bond ISIN list.",
    )
    p.add_argument(
        "--isins",
        default="",
        help="Comma-separated ISIN list to use (overrides auto selection).",
    )
    p.add_argument(
        "--isins-file",
        default="",
        help="Path to a text/CSV file containing an ISIN column or 1 ISIN per line.",
    )

    # Interp/extrap controls
    p.add_argument(
        "--no-extrap",
        action="store_true",
        help="Do not extrapolate curve nodes outside available maturities (return NaN outside range).",
    )
    p.add_argument(
        "--linear-extrap",
        action="store_true",
        help="If extrapolation is needed, use linear extrapolation from the nearest two points \
              (default NumPy behavior is constant endpoint).",
    )

    # Input-format helpers
    p.add_argument(
        "--year-hint",
        type=int,
        default=2026,
        help="Year for Jan05-style columns (bonds_clean_matrix format).",
    )
    p.add_argument("--no-plots", action="store_true", help="Disable plot generation.")

    return p.parse_args()


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(c))]


def monthday_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"[A-Za-z]{3}\d{2}", str(c))]


def normalize_monthday_columns(df: pd.DataFrame, year: int) -> pd.DataFrame:
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    rename = {}
    for col in monthday_columns(df):
        mon = col[:3].title()
        day = col[3:]
        if mon in month_map:
            iso = f"{year:04d}-{month_map[mon]:02d}-{int(day):02d}"
            rename[col] = iso
    return df.rename(columns=rename) if rename else df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "ISIN" in df.columns:
        rename["ISIN"] = "isin"
    if "Coupon" in df.columns:
        rename["Coupon"] = "coupon"
    if "Coupon_Rate" in df.columns:
        rename["Coupon_Rate"] = "coupon_rate"
    if "Maturity_Date" in df.columns:
        rename["Maturity_Date"] = "maturity_date"
    if "Issue_Date" in df.columns:
        rename["Issue_Date"] = "issue_date"
    if "Years_to_Maturity" in df.columns:
        rename["Years_to_Maturity"] = "years_to_maturity"
    return df.rename(columns=rename)


def years_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    return (d2.date() - d1.date()).days / 365.25


def parse_coupon_rate(row: pd.Series) -> Optional[float]:
    val = row.get("coupon_rate")
    if pd.notna(val):
        try:
            return float(val)
        except Exception:
            pass
    coupon = row.get("coupon")
    if isinstance(coupon, str):
        m = re.search(r"-?\d+(\.\d+)?", coupon.replace(",", "."))
        if m:
            return float(m.group(0))
    return None


def ytm_from_price(
    price: float,
    coupon_rate: float,
    maturity: pd.Timestamp,
    settle: pd.Timestamp,
    *,
    face: float = FACE_VALUE,
    freq: int = COUPON_FREQ,
) -> float:
    """Bond YTM solved by bisection.

    Note: We keep the same simplifying convention as the original script:
      - equally-spaced coupon times
      - no accrued interest adjustment

    Minor robustness change vs original: use ceil() for number of periods so the
    period count doesn't jump due to rounding when moving settle by 1 day.
    """

    years = years_between(settle, maturity)
    if years <= 0 or price <= 0:
        return float("nan")

    n_periods = max(1, int(math.ceil(years * freq)))
    coupon = face * coupon_rate / 100.0 / freq

    def pv(rate: float) -> float:
        denom = 1.0 + rate / freq
        total = 0.0
        for k in range(1, n_periods + 1):
            total += coupon / (denom**k)
        total += face / (denom**n_periods)
        return total

    lo, hi = -0.95, 1.0
    while pv(hi) > price and hi < 5.0:
        hi *= 1.5
    if pv(lo) < price:
        return float("nan")

    for _ in range(120):
        mid = (lo + hi) / 2.0
        val = pv(mid)
        if abs(val - price) < 1e-8:
            return mid
        if val > price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def select_bonds_auto(df: pd.DataFrame, date_cols: list[str], *, count: int, max_years: float) -> pd.DataFrame:
    """Original auto-selection logic (quantile buckets by maturity, preferring completeness)."""
    df = df.copy()
    first_date = pd.to_datetime(date_cols[0])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    df["years_to_maturity"] = df["maturity_date"].apply(
        lambda d: years_between(first_date, d) if pd.notna(d) else math.nan
    )
    df = df[(df["years_to_maturity"] > 0) & (df["years_to_maturity"] <= max_years)]
    df["non_missing"] = df[date_cols].notna().sum(axis=1)
    df = df.sort_values("maturity_date")

    if len(df) <= count:
        return df

    buckets = min(count, len(df))
    df["bucket"] = pd.qcut(df["maturity_date"].rank(method="first"), q=buckets, duplicates="drop")
    picked = (
        df.sort_values(["bucket", "non_missing", "maturity_date"], ascending=[True, False, True])
        .groupby("bucket", as_index=False, observed=True)
        .head(1)
        .drop(columns=["bucket"])
    )

    if len(picked) < count:
        remaining = df.loc[~df.index.isin(picked.index)].sort_values(
            ["non_missing", "maturity_date"], ascending=[False, True]
        )
        picked = pd.concat([picked, remaining.head(count - len(picked))], ignore_index=True)

    return picked


def load_isins_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        cols = [c for c in df.columns if str(c).lower() in {"isin", "isIN".lower()}]
        if not cols:
            raise ValueError(f"No 'isin' column found in {path}")
        return [str(x).strip() for x in df[cols[0]].dropna().tolist()]
    # default: treat as text file, 1 per line
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def compute_ytm_matrix(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    ytm = pd.DataFrame(index=df.index, columns=date_cols, dtype="float64")
    for idx, row in df.iterrows():
        maturity = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        coupon_rate = parse_coupon_rate(row)
        if pd.isna(maturity) or coupon_rate is None:
            continue
        for col in date_cols:
            price = row.get(col)
            if pd.isna(price):
                continue
            settle = pd.to_datetime(col)
            ytm.loc[idx, col] = ytm_from_price(float(price), coupon_rate, maturity, settle)
    return ytm


def coupon_schedule(years: float, *, freq: int) -> list[float]:
    n = max(1, int(math.ceil(years * freq)))
    return [years - (n - k) / freq for k in range(1, n + 1)]


def interpolate_spot(t: float, known_times: list[float], known_rates: list[float]) -> Optional[float]:
    if not known_times:
        return None
    if len(known_times) == 1:
        return known_rates[0]
    return float(np.interp(t, known_times, known_rates))


def bootstrap_spot_rates(bonds: Iterable[dict], *, freq: int = COUPON_FREQ) -> list[tuple[float, float]]:
    results: list[tuple[float, float]] = []
    known_times: list[float] = []
    known_rates: list[float] = []

    for bond in sorted(bonds, key=lambda b: b["maturity_years"]):
        years = float(bond["maturity_years"])
        price = bond.get("price")
        coupon_rate = bond.get("coupon_rate")
        if price is None or coupon_rate is None or years <= 0:
            continue

        coupon = FACE_VALUE * coupon_rate / 100.0 / freq
        times = coupon_schedule(years, freq=freq)

        pv_prev = 0.0
        for t in times[:-1]:
            r_t = interpolate_spot(t, known_times, known_rates)
            if r_t is None:
                pv_prev = None
                break
            pv_prev += coupon / ((1.0 + r_t / freq) ** (freq * t))

        if pv_prev is None:
            continue

        remaining = price - pv_prev
        if remaining <= 0:
            continue

        t_n = times[-1]
        denom = FACE_VALUE + coupon
        spot = freq * ((denom / remaining) ** (1.0 / (freq * t_n)) - 1.0)

        known_times.append(t_n)
        known_rates.append(spot)
        results.append((t_n, spot))

    return results


def _linear_extrap_1d(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Like np.interp, but linearly extrapolates using the nearest two points."""
    out = np.interp(x, xp, fp)  # inside range ok
    if len(xp) < 2:
        return out

    # left side
    left_mask = x < xp[0]
    if np.any(left_mask):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[left_mask] = fp[0] + slope * (x[left_mask] - xp[0])

    # right side
    right_mask = x > xp[-1]
    if np.any(right_mask):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[right_mask] = fp[-1] + slope * (x[right_mask] - xp[-1])

    return out


def interpolate_curve(
    maturities: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    allow_extrap: bool,
    linear_extrap: bool,
) -> np.ndarray:
    order = np.argsort(maturities)
    maturities = maturities[order]
    values = values[order]

    if len(maturities) == 0:
        return np.full_like(targets, np.nan, dtype=float)

    if not allow_extrap:
        return np.interp(targets, maturities, values, left=np.nan, right=np.nan)

    if linear_extrap:
        return _linear_extrap_1d(targets, maturities, values)

    # Default NumPy behavior (constant outside range)
    return np.interp(targets, maturities, values)


def forward_from_spot(s1, s2, s3, s4, s5, *, freq: int = COUPON_FREQ) -> dict:
    def fwd(s_t, s_tn, t, n):
        if pd.isna(s_t) or pd.isna(s_tn):
            return np.nan
        a = (1 + s_tn / freq) ** (freq * (t + n)) / (1 + s_t / freq) ** (freq * t)
        return freq * (a ** (1.0 / (freq * n)) - 1.0)

    return {
        "1yr_1yr": fwd(s1, s2, 1, 1),
        "1yr_2yr": fwd(s1, s3, 1, 2),
        "1yr_3yr": fwd(s1, s4, 1, 3),
        "1yr_4yr": fwd(s1, s5, 1, 4),
    }


def log_return_cov(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_index()
    df = df.where(df > 0)
    returns = np.log(df.shift(-1) / df).iloc[:-1].dropna()
    return returns.cov()


def pca_from_cov(cov: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    if cov.empty:
        return pd.Series(dtype=float), pd.DataFrame()
    vals, vecs = np.linalg.eigh(cov.values)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    eigvals = pd.Series(vals, index=[f"PC{i+1}" for i in range(len(vals))])
    eigvecs = pd.DataFrame(vecs, index=cov.index, columns=eigvals.index)
    return eigvals, eigvecs


def plot_curves(
    df: pd.DataFrame,
    x_values: list[float],
    columns: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for idx, row in df.iterrows():
        y_vals = row[columns].to_numpy(dtype=float)
        if np.isnan(y_vals).all():
            continue
        ax.plot(x_values, y_vals, alpha=0.7, linewidth=1.0, label=str(idx))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(df) <= 12:
        ax.legend(fontsize=7, ncols=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Input data is empty. Provide a non-empty dataset.")

    df = normalize_monthday_columns(df, args.year_hint)
    df = standardize_columns(df)

    date_cols = date_columns(df)
    if not date_cols:
        raise ValueError("No ISO date columns (YYYY-MM-DD) found.")
    if "maturity_date" not in df.columns or "isin" not in df.columns:
        raise ValueError("Missing required columns (isin, maturity_date).")

    # --- Bond selection ---
    chosen_isins: list[str] = []
    if args.fixed_isins:
        chosen_isins = DEFAULT_SELECTED_ISINS.copy()
    if args.isins.strip():
        chosen_isins = [x.strip() for x in args.isins.split(",") if x.strip()]
    if args.isins_file.strip():
        chosen_isins = load_isins_from_file(Path(args.isins_file))

    if chosen_isins:
        selected = df[df["isin"].isin(chosen_isins)].copy()
        missing = sorted(set(chosen_isins) - set(selected["isin"].tolist()))
        if missing:
            raise ValueError(f"These ISINs were not found in the dataset: {missing}")
        # Preserve the user-specified order, then sort by maturity for nicer output
        selected["_order"] = selected["isin"].apply(lambda x: chosen_isins.index(x))
        selected = selected.sort_values(["_order", "maturity_date"]).drop(columns=["_order"])
    else:
        selected = select_bonds_auto(df, date_cols, count=args.select_count, max_years=args.max_years)

    # Compute YTM (only once for whole df, then subselect)
    ytm_all = compute_ytm_matrix(df, date_cols)
    ytm_selected = ytm_all.loc[selected.index]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Recompute years_to_maturity relative to first date for reporting
    first_date = pd.to_datetime(date_cols[0])
    selected = selected.copy()
    selected["maturity_date"] = pd.to_datetime(selected["maturity_date"], errors="coerce")
    selected["years_to_maturity"] = selected["maturity_date"].apply(
        lambda d: years_between(first_date, d) if pd.notna(d) else math.nan
    )

    selected_out = selected[
        ["isin", "coupon", "coupon_rate", "maturity_date", "issue_date", "years_to_maturity"]
    ].copy()
    selected_out.to_csv(outdir / "selected_bonds.csv", index=False)
    ytm_selected.to_csv(outdir / "selected_bonds_ytm.csv")

    # --- Build daily curves ---
    target_years = np.array([1, 2, 3, 4, 5], dtype=float)
    allow_extrap = not args.no_extrap

    yield_curves = []
    spot_curves = []
    forward_curves = []

    for col in date_cols:
        settle = pd.to_datetime(col)
        mats: list[float] = []
        yields: list[float] = []
        prices: list[float] = []
        coupons: list[float] = []

        for idx, row in selected.iterrows():
            maturity = pd.to_datetime(row.get("maturity_date"), errors="coerce")
            if pd.isna(maturity):
                continue
            years = years_between(settle, maturity)
            if years <= 0:
                continue
            y = ytm_selected.loc[idx, col]
            price = row.get(col)
            coupon_rate = parse_coupon_rate(row)
            if pd.isna(y) or pd.isna(price) or coupon_rate is None:
                continue
            mats.append(years)
            yields.append(float(y))
            prices.append(float(price))
            coupons.append(float(coupon_rate))

        mats_arr = np.array(mats, dtype=float)
        yields_arr = np.array(yields, dtype=float)

        if len(mats_arr) == 0:
            yield_vals = np.full_like(target_years, np.nan)
        else:
            yield_vals = interpolate_curve(
                mats_arr,
                yields_arr,
                target_years,
                allow_extrap=allow_extrap,
                linear_extrap=args.linear_extrap,
            )

        yield_curves.append({"date": col, **{f"y{int(t)}": v for t, v in zip(target_years, yield_vals)}})

        # Spot curve via bootstrap
        bond_list = [
            {"maturity_years": m, "price": p, "coupon_rate": c, "ytm": y}
            for m, p, c, y in zip(mats, prices, coupons, yields)
        ]
        spot_pairs = bootstrap_spot_rates(bond_list, freq=COUPON_FREQ)
        if spot_pairs:
            smats = np.array([m for m, _ in spot_pairs], dtype=float)
            srates = np.array([r for _, r in spot_pairs], dtype=float)
            spot_vals = interpolate_curve(
                smats,
                srates,
                target_years,
                allow_extrap=allow_extrap,
                linear_extrap=args.linear_extrap,
            )
        else:
            spot_vals = np.full_like(target_years, np.nan)

        spot_curves.append({"date": col, **{f"s{int(t)}": v for t, v in zip(target_years, spot_vals)}})

        s1, s2, s3, s4, s5 = spot_vals
        forward_curves.append({"date": col, **forward_from_spot(s1, s2, s3, s4, s5)})

    df_yield = pd.DataFrame(yield_curves).set_index("date")[[f"y{i}" for i in range(1, 6)]]
    df_spot = pd.DataFrame(spot_curves).set_index("date")[[f"s{i}" for i in range(1, 6)]]
    df_forward = pd.DataFrame(forward_curves).set_index("date")[["1yr_1yr", "1yr_2yr", "1yr_3yr", "1yr_4yr"]]

    df_yield.to_csv(outdir / "yield_curve_daily.csv")
    df_spot.to_csv(outdir / "spot_curve_daily.csv")
    df_forward.to_csv(outdir / "forward_curve_daily.csv")

    if not args.no_plots:
        plot_curves(
            df_yield,
            x_values=[1, 2, 3, 4, 5],
            columns=[f"y{i}" for i in range(1, 6)],
            title="Daily 1–5Y Yield Curves (YTM)",
            xlabel="Maturity (Years)",
            ylabel="Yield (annual, decimal)",
            out_path=outdir / "yield_curve_daily.png",
        )
        plot_curves(
            df_spot,
            x_values=[1, 2, 3, 4, 5],
            columns=[f"s{i}" for i in range(1, 6)],
            title="Daily 1–5Y Spot Curves",
            xlabel="Maturity (Years)",
            ylabel="Spot Rate (annual, decimal)",
            out_path=outdir / "spot_curve_daily.png",
        )
        plot_curves(
            df_forward,
            x_values=[2, 3, 4, 5],
            columns=["1yr_1yr", "1yr_2yr", "1yr_3yr", "1yr_4yr"],
            title="Daily 1Y Forward Curves (1Y–2Y through 1Y–5Y)",
            xlabel="Forward End (Years)",
            ylabel="Forward Rate (annual, decimal)",
            out_path=outdir / "forward_curve_daily.png",
        )

    cov_yield = log_return_cov(df_yield)
    cov_forward = log_return_cov(df_forward)
    cov_yield.to_csv(outdir / "cov_yield.csv")
    cov_forward.to_csv(outdir / "cov_forward.csv")

    eigvals_y, eigvecs_y = pca_from_cov(cov_yield)
    eigvals_f, eigvecs_f = pca_from_cov(cov_forward)
    eigvals_y.to_csv(outdir / "pca_yield_eigenvalues.csv", header=["eigenvalue"])
    eigvecs_y.to_csv(outdir / "pca_yield_eigenvectors.csv")
    eigvals_f.to_csv(outdir / "pca_forward_eigenvalues.csv", header=["eigenvalue"])
    eigvecs_f.to_csv(outdir / "pca_forward_eigenvectors.csv")

    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()
