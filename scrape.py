"""
APM466 Bond Data Scraper (Markets Insider)

Goal
----
Collect all Canadian Government bonds (Frankfurt exchange) with maturity < 10 years
from the assignment date window, and export:
  - bonds_final_price_matrix.csv
  - bonds_clean_matrix.csv
  - bonds_final.json

Primary source pages:
  - Finder: shortterm + midterm queries (Canada, CAD, gov bonds)
  - Bond snapshot page (ISIN, coupon, issue date, maturity date)
  - Chart_GetChartData (historical close prices)

Dependencies
------------
beautifulsoup4
pandas
numpy
requests
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ======================================================================================
# 1) Date configuration (single source of truth)
# ======================================================================================
START_DATE = pd.to_datetime("2026-01-05")  # inclusive
# Assignment window is Jan 5–19, 2026 (10 weekdays). Default to the 10 weekdays: Jan 5–16.
END_DATE = pd.to_datetime("2026-01-16")  # inclusive

FROM_YYYYMMDD = START_DATE.strftime("%Y%m%d")
TO_YYYYMMDD = END_DATE.strftime("%Y%m%d")

START = START_DATE.date()
END = END_DATE.date()
EXPECTED_DATES = list(pd.bdate_range(START_DATE, END_DATE).date)


# ======================================================================================
# 2) HTTP settings
# ======================================================================================
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}
TIMEOUT_SEC = 30

# Finder URLs from assignment prompt (Canada, CAD, gov bonds).
FINDER_URLS = {
    "shortterm": (
        "https://markets.businessinsider.com/bonds/finder?"
        "borrower=71&maturity=shortterm&yield=&bondtype=2%2c3%2c4%2c16"
        "&coupon=&currency=184&rating=&country=19"
    ),
    "midterm": (
        "https://markets.businessinsider.com/bonds/finder?"
        "borrower=71&maturity=midterm&yield=&bondtype=2%2c3%2c4%2c16"
        "&coupon=&currency=184&rating=&country=19"
    ),
}

MAX_YEARS_DEFAULT = 10.0
DEFAULT_OUTDIR = "scraped_data"


# ======================================================================================
# 3) Bond inputs (bond_page_url, chart_json_url used only to extract tkData)
# ======================================================================================
RAW_BONDS: list[tuple[str, str]] = [
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202326-bond-2026-ca135087r226?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C130654501%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202026-bond-2026-ca135087l518?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C57601476%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202326-bond-2026-ca135087p816?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C124578091%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202426-bond-2026-ca135087r556?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C132969994%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_201526-bond-2026-ca135087e679?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C28975906%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202426-bond-2026-ca135087r978?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C135402145%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202126-bond-2026-ca135087l930?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C111141014%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202426-bond-2026-ca135087s398?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C137893857%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202427-bond-2027-ca135087s547?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C139591564%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202127-bond-2027-ca135087m847?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C114329463%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202527-bond-2027-ca135087s885?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C142700048%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_201627-bond-2027-ca135087f825?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C33461441%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/8_000-canada-government-of-bond-2027-ca135087vw17?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C490501%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202527-bond-2027-ca135087t461?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C145462038%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202227-bond-2027-ca135087p733?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C123653782%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202227-bond-2027-ca135087n837?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C119036843%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202527-bond-2027-ca135087t610?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C148260836%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202528-bond-2028-ca135087t958?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C151027258%2C16%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202228-bond-2028-ca135087p576?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C122651336%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_201728-bond-2028-ca135087h235?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C37720230%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202328-bond-2028-ca135087q491?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C126528699%2C1330%2C184&from=20251115&to=20260115"),

    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202532-bond-2032-ca135087s968?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C142567197%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202030_series_l443-bond-2030-ca135087l443?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C57501684%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/5_750-canada-government-of-bond-2029-ca135087wl43?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C847359%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202229-bond-2029-ca135087n670?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C117855494%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202434-bond-2034-ca135087r713?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C133356706%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202430-bond-2030-ca135087s471?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C138913105%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202429-bond-2029-ca135087r895?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C134387959%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202531-bond-2031-ca135087t792?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C149759344%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202131-bond-2031-ca135087m276?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C111293704%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202131-bond-2031-ca135087n266?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C114466942%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202535-bond-2035-ca135087t537?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C146822867%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202530-bond-2030-ca135087t388?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C144389623%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202334-bond-2034-ca135087r481?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C131843886%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202232-bond-2032-ca135087p329?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C120914999%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/5_750-canada-government-of-bond-2033-ca135087xg49?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C1321208%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202329-bond-2029-ca135087q988?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C130328741%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202333-bond-2033-ca135087q723?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C128007448%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202232-bond-2032-ca135087n597?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C117606959%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202434-bond-2034-ca135087s216?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C136888775%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_202535-bond-2035-ca135087s620?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C141608780%2C1330%2C184&from=20251115&to=20260115"),
    ("https://markets.businessinsider.com/bonds/canadacd-bonds_201829-bond-2029-ca135087j397?miRedirects=1",
     "https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Bond&tkData=1%2C42965744%2C1330%2C184&from=20251115&to=20260115")
]


# ======================================================================================
# 4) Build BONDS list (bond_page_url + tkData) and deduplicate for uniqueness
# ======================================================================================
def extract_tkdata(chart_json_url: str) -> str:
    u = unquote(chart_json_url)
    m = re.search(r"(?:\?|&)tkData=([^&]+)", u)
    if not m:
        raise ValueError("tkData not found in chart_json_url")
    return m.group(1)


def configure_dates(start_date: str, end_date: str) -> None:
    global START_DATE, END_DATE, FROM_YYYYMMDD, TO_YYYYMMDD, START, END, EXPECTED_DATES
    START_DATE = pd.to_datetime(start_date)
    END_DATE = pd.to_datetime(end_date)
    FROM_YYYYMMDD = START_DATE.strftime("%Y%m%d")
    TO_YYYYMMDD = END_DATE.strftime("%Y%m%d")
    START = START_DATE.date()
    END = END_DATE.date()
    EXPECTED_DATES = list(pd.bdate_range(START_DATE, END_DATE).date)


def extract_tkdata_from_html(html: str) -> Optional[str]:
    # Try data attribute variants first.
    m = re.search(r"data-tkdata=[\"']([^\"']+)[\"']", html, flags=re.I)
    if m:
        return unquote(m.group(1))

    # Try tkData in querystring form.
    m = re.search(r"tkData=([0-9%2C,]+)", html, flags=re.I)
    if m:
        return unquote(m.group(1))

    # Try JS assignments like: tkData: "1,2,3,4" or tkdata='1,2,3,4'
    m = re.search(r"tkdata\s*[:=]\s*[\"']([^\"']+)[\"']", html, flags=re.I)
    if m:
        return m.group(1)

    # Try JSON-like field with double quotes.
    m = re.search(r'"tkdata"\s*:\s*"([^"]+)"', html, flags=re.I)
    if m:
        return m.group(1)

    # Try a chart endpoint URL embedded in JS.
    m = re.search(r"Chart_GetChartData\?[^\"']+", html, flags=re.I)
    if m:
        return extract_tkdata(m.group(0))
    return None


def add_or_replace_param(url: str, **params: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def fetch_html(session: requests.Session, url: str, *, retries: int = 3, sleep_s: float = 0.5) -> str:
    for attempt in range(1, retries + 1):
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT_SEC)
        if resp.status_code == 200:
            resp.encoding = resp.apparent_encoding or resp.encoding
            return resp.text
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(sleep_s * attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def extract_links_from_finder(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    best_table = None
    best_links = 0

    for table in tables:
        links = [a for a in table.find_all("a", href=True) if "/bonds/" in a["href"]]
        if len(links) > best_links:
            best_links = len(links)
            best_table = table

    if not best_table:
        return []

    urls = []
    for row in best_table.find_all("tr"):
        link = row.find("a", href=True)
        if not link:
            continue
        href = link["href"].strip()
        if "/bonds/" not in href:
            continue
        if href.startswith("/"):
            href = "https://markets.businessinsider.com" + href
        urls.append(href)
    return urls


def load_bonds_from_finder(
    session: requests.Session,
    base_urls: dict[str, str],
    *,
    max_pages: int = 50,
    sleep_s: float = 0.5,
) -> list[dict]:
    all_urls: list[str] = []
    for url in base_urls.values():
        seen_first: Optional[str] = None
        for page in range(1, max_pages + 1):
            page_url = add_or_replace_param(url, p=str(page))
            html = fetch_html(session, page_url, sleep_s=sleep_s)
            links = extract_links_from_finder(html)
            if not links:
                break
            if seen_first and links[0] == seen_first:
                break
            if page == 1:
                seen_first = links[0]
            all_urls.extend(links)
            time.sleep(sleep_s)

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for u in all_urls:
        if u in seen:
            continue
        seen.add(u)
        deduped.append({"bond_page_url": u, "tkData": None, "chart_json_url": None})
    return deduped


def load_bonds_from_raw(raw_bonds: Iterable[Tuple[str, str]]) -> list[dict]:
    bond_records = []
    for bond_page_url, chart_json_url in raw_bonds:
        try:
            tkdata = extract_tkdata(chart_json_url)
        except Exception:
            tkdata = None
        bond_records.append(
            {
                "bond_page_url": bond_page_url,
                "tkData": tkdata,
                "chart_json_url": chart_json_url,
            }
        )

    df_input = pd.DataFrame(bond_records)
    df_unique = df_input.drop_duplicates(subset=["tkData"], keep="first").copy()
    df_unique = df_unique.drop_duplicates(subset=["bond_page_url"], keep="first").copy()
    return df_unique.to_dict(orient="records")


def load_bonds_from_file(path: str) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(path)

    if file_path.suffix.lower() == ".json":
        df = pd.read_json(file_path)
    else:
        df = pd.read_csv(file_path)

    if "bond_page_url" not in df.columns:
        raise ValueError("Input file must include a bond_page_url column.")

    records = []
    for _, row in df.iterrows():
        bond_page_url = row.get("bond_page_url")
        chart_json_url = row.get("chart_json_url")
        tkdata = row.get("tkData")
        if pd.isna(tkdata) or tkdata == "":
            if isinstance(chart_json_url, str) and chart_json_url:
                tkdata = extract_tkdata(chart_json_url)
            else:
                tkdata = None
        records.append(
            {
                "bond_page_url": bond_page_url,
                "tkData": tkdata,
                "chart_json_url": chart_json_url,
            }
        )

    df_input = pd.DataFrame(records)
    df_unique = df_input.drop_duplicates(subset=["tkData"], keep="first").copy()
    df_unique = df_unique.drop_duplicates(subset=["bond_page_url"], keep="first").copy()
    return df_unique.to_dict(orient="records")


# ======================================================================================
# 5) Working-code helper functions
# ======================================================================================
def normalize_date(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.date().isoformat()


def parse_coupon(value: Optional[str]) -> tuple[Optional[str], Optional[float]]:
    if not value or not isinstance(value, str):
        return None, None
    raw = value.replace(" ", "").replace("%", "")
    raw = raw.replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", raw)
    if not m:
        return None, None
    rate = float(m.group(0))
    coupon_str = f"{rate:.4f}%"
    return coupon_str, rate


def years_to_maturity(maturity_iso: Optional[str]) -> Optional[float]:
    if not maturity_iso:
        return None
    maturity = pd.to_datetime(maturity_iso, errors="coerce")
    if pd.isna(maturity):
        return None
    years = (maturity.date() - START).days / 365.25
    return round(years, 4)


def find_field(info: dict, keywords: list[str]) -> Optional[str]:
    if not info:
        return None
    for key, val in info.items():
        key_l = str(key).lower()
        if all(k in key_l for k in keywords):
            return val
    return None


def extract_coupon_value(info: dict) -> Optional[str]:
    if not info:
        return None
    if "Coupon" in info:
        return info.get("Coupon")
    for key, val in info.items():
        key_l = str(key).lower()
        if "coupon" in key_l and "date" not in key_l and "payment" not in key_l and "start" not in key_l:
            return val
    return None


def extract_issue_date(info: dict) -> Optional[str]:
    return find_field(info, ["issue", "date"])


def extract_maturity_date(info: dict) -> Optional[str]:
    return find_field(info, ["maturity", "date"])


def expected_date_labels() -> dict[str, str]:
    return {d.isoformat(): d.strftime("%b%d") for d in EXPECTED_DATES}


def isin_from_bond_url(url: str) -> str | None:
    m = re.search(r"\b(ca[0-9a-z]{10})\b", url, re.I)
    return m.group(1).upper() if m else None


def chart_url_for_tkdata(tkdata: str) -> str:
    return (
        "https://markets.businessinsider.com/Ajax/Chart_GetChartData"
        f"?instrumentType=Bond&tkData={tkdata}&from={FROM_YYYYMMDD}&to={TO_YYYYMMDD}"
    )


def iter_points(obj):
    if isinstance(obj, list):
        if obj and all(isinstance(x, list) and len(x) >= 2 for x in obj):
            for x in obj:
                yield x[0], x[1]
        for x in obj:
            yield from iter_points(x)
    elif isinstance(obj, dict):
        keys = {str(k).lower(): k for k in obj.keys()}
        tkey = next((keys[k] for k in ["x", "t", "time", "date", "timestamp"] if k in keys), None)
        vkey = next((keys[k] for k in ["y", "v", "value", "close", "price", "last"] if k in keys), None)
        if tkey and vkey:
            yield obj[tkey], obj[vkey]
        for v in obj.values():
            yield from iter_points(v)


def to_date(t):
    if isinstance(t, str):
        dt = pd.to_datetime(t, errors="coerce")
        return None if pd.isna(dt) else dt.date()
    try:
        t = float(t)
    except Exception:
        return None
    unit = "ms" if t > 1e11 else "s"
    dt = pd.to_datetime(t, unit=unit, errors="coerce")
    return None if pd.isna(dt) else dt.date()


def fetch_bond_info(session: requests.Session, bond_page_url: str) -> tuple[dict, Optional[str]]:
    html = session.get(bond_page_url, headers=HEADERS, timeout=TIMEOUT_SEC).text
    soup = BeautifulSoup(html, "html.parser")

    # NOTE: ISIN_from_url is intentionally NOT included in the output dict
    # to satisfy the requirement to remove that redundant column.
    out = {"bond_page_url": bond_page_url}

    tbody = soup.select_one("tbody.table__tbody")
    rows = tbody.select("tr.table__tr") if tbody else soup.select("table tr")

    current_group = None
    for tr in rows:
        tds = tr.select("td.table__td") if tbody else tr.find_all(["th", "td"])
        if not tds:
            continue

        if tbody and len(tds) == 1 and "group" in (tds[0].get("class") or []):
            current_group = tds[0].get_text(" ", strip=True)
            continue

        if len(tds) >= 2:
            key = tds[0].get_text(" ", strip=True)
            val = tds[1].get_text(" ", strip=True)

            if not key or not val:
                continue

            final_key = key
            if final_key in out and current_group:
                final_key = f"{current_group}::{key}"
            elif final_key in out:
                final_key = f"{key}__2"

            val_norm = val.replace("\xa0", " ").strip()
            if re.fullmatch(r"-?\d+,\d+", val_norm):
                val_norm = val_norm.replace(",", ".")
            out[final_key] = val_norm

    if "ISIN" in out and isinstance(out["ISIN"], str):
        out["ISIN"] = out["ISIN"].strip().upper()

    return out, extract_tkdata_from_html(html)


def fetch_price_series(
    session: requests.Session,
    tkdata: str,
    *,
    interpolate: bool = True,
) -> tuple[pd.Series, str]:
    chart_url = chart_url_for_tkdata(tkdata)
    resp = session.get(chart_url, headers=HEADERS, timeout=TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()

    pts = []
    for t, v in iter_points(data):
        d = to_date(t)
        if d is None:
            continue
        try:
            val = float(v)
        except Exception:
            continue
        pts.append((d, val))

    if not pts:
        raise RuntimeError("No time-series points parsed from Chart_GetChartData.")

    df = pd.DataFrame(pts, columns=["date", "price"])
    df = df.groupby("date", as_index=False)["price"].last().sort_values("date")
    df = df[(df["date"] >= START) & (df["date"] <= END)]
    df = df[[pd.to_datetime(d).weekday() < 5 for d in df["date"]]]

    series = df.set_index("date")["price"]
    # Reindex to expected weekdays
    series = series.reindex(EXPECTED_DATES)
    series.index = pd.to_datetime(series.index)

    if interpolate and series.notna().sum() >= 2:
        series = series.interpolate(method="time", limit_area="inside")

    return series, chart_url




def run_pipeline(
    bonds: list[dict],
    *,
    outdir: str,
    interpolate: bool,
    max_years: float,
    export_long: bool,
    sleep_s: float,
    max_bonds: Optional[int],
) -> None:
    session = requests.Session()

    errors = []
    records = []
    long_rows = []

    if max_bonds:
        bonds = bonds[:max_bonds]

    for bond in bonds:
        bond_page_url = bond.get("bond_page_url")
        if not bond_page_url:
            continue

        try:
            info, tkdata_from_page = fetch_bond_info(session, bond_page_url)
        except Exception as exc:
            errors.append((bond_page_url, f"snapshot_error: {exc}"))
            continue

        tkdata = bond.get("tkData") or tkdata_from_page
        if not tkdata and bond.get("chart_json_url"):
            try:
                tkdata = extract_tkdata(bond["chart_json_url"])
            except Exception:
                tkdata = None

        if not tkdata:
            errors.append((bond_page_url, "Missing tkData (cannot fetch chart data)."))
            continue

        isin = info.get("ISIN") if isinstance(info.get("ISIN"), str) else None
        if not isin:
            isin = isin_from_bond_url(bond_page_url)

        coupon_raw = extract_coupon_value(info)
        coupon_str, coupon_rate = parse_coupon(coupon_raw)

        maturity_raw = extract_maturity_date(info)
        issue_raw = extract_issue_date(info)
        maturity_iso = normalize_date(maturity_raw)
        issue_iso = normalize_date(issue_raw)
        yrs = years_to_maturity(maturity_iso)

        if not maturity_iso or yrs is None:
            errors.append((bond_page_url, "Missing or invalid maturity date."))
            continue
        if yrs > max_years:
            continue

        try:
            series, chart_url = fetch_price_series(session, tkdata, interpolate=interpolate)
        except Exception as exc:
            errors.append((bond_page_url, f"chart_error: {exc}"))
            continue

        record = {
            "isin": isin,
            "coupon": coupon_str,
            "coupon_rate": coupon_rate,
            "maturity_date": maturity_iso,
            "issue_date": issue_iso,
            "years_to_maturity": yrs,
        }
        for d in EXPECTED_DATES:
            key = d.isoformat()
            val = series.get(pd.to_datetime(d))
            record[key] = None if pd.isna(val) else float(val)

        records.append(record)

        if export_long:
            for d in EXPECTED_DATES:
                key = d.isoformat()
                long_rows.append(
                    {
                        "isin": isin,
                        "date": key,
                        "close_price": record.get(key),
                        "coupon": coupon_str,
                        "coupon_rate": coupon_rate,
                        "maturity_date": maturity_iso,
                        "issue_date": issue_iso,
                        "years_to_maturity": yrs,
                        "bond_page_url": bond_page_url,
                        "chart_json_url": chart_url,
                        "tkData": tkdata,
                    }
                )

        time.sleep(sleep_s)

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_final = pd.DataFrame.from_records(records)
    for d in EXPECTED_DATES:
        col = d.isoformat()
        if col not in df_final.columns:
            df_final[col] = np.nan

    if not df_final.empty and "maturity_date" in df_final.columns:
        df_final = df_final.sort_values("maturity_date").reset_index(drop=True)

    final_csv = out_path / "bonds_final_price_matrix.csv"
    final_json = out_path / "bonds_final.json"
    df_final.to_csv(final_csv, index=False)
    with final_json.open("w", encoding="utf-8") as f:
        json.dump(df_final.to_dict(orient="records"), f, indent=2)

    # Clean matrix with Jan05-style columns
    labels = expected_date_labels()
    rename_map = {
        "isin": "ISIN",
        "coupon": "Coupon",
        "coupon_rate": "Coupon_Rate",
        "maturity_date": "Maturity_Date",
        "issue_date": "Issue_Date",
        "years_to_maturity": "Years_to_Maturity",
    }
    df_clean = df_final.rename(columns=rename_map)
    df_clean = df_clean.rename(columns=labels)
    clean_csv = out_path / "bonds_clean_matrix.csv"
    df_clean.to_csv(clean_csv, index=False)

    if export_long and long_rows:
        long_csv = out_path / "bonds_master_long.csv"
        pd.DataFrame.from_records(long_rows).to_csv(long_csv, index=False)

    if errors:
        errors_csv = out_path / "scrape_errors.csv"
        pd.DataFrame(errors, columns=["bond_page_url", "error"]).to_csv(errors_csv, index=False)
        print(f"Errors written to: {errors_csv}")

    print(f"Wrote: {final_csv}, {clean_csv}, {final_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Markets Insider bond data.")
    parser.add_argument("--start-date", default="2026-01-05", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2026-01-16", help="End date (YYYY-MM-DD).")
    parser.add_argument("--input", help="CSV/JSON file with bond_page_url and tkData or chart_json_url.")
    parser.add_argument("--use-raw", action="store_true", help="Use RAW_BONDS list instead of finder.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory.")
    parser.add_argument("--max-bonds", type=int, help="Limit number of bonds for quick runs.")
    parser.add_argument("--max-years", type=float, default=MAX_YEARS_DEFAULT, help="Max years to maturity.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep between bond requests.")
    parser.add_argument("--finder-max-pages", type=int, default=50, help="Max pages per finder query.")
    parser.add_argument("--finder-sleep", type=float, default=0.5, help="Sleep between finder pages.")
    parser.add_argument("--no-interpolate", action="store_true", help="Disable interpolation of missing prices.")
    parser.add_argument("--export-long", action="store_true", help="Export long-format CSV (debug).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_dates(args.start_date, args.end_date)

    if args.input:
        bonds = load_bonds_from_file(args.input)
    elif args.use_raw:
        bonds = load_bonds_from_raw(RAW_BONDS)
    else:
        session = requests.Session()
        bonds = load_bonds_from_finder(
            session,
            FINDER_URLS,
            max_pages=args.finder_max_pages,
            sleep_s=args.finder_sleep,
        )

    run_pipeline(
        bonds,
        outdir=args.outdir,
        interpolate=not args.no_interpolate,
        max_years=args.max_years,
        export_long=args.export_long,
        sleep_s=args.sleep,
        max_bonds=args.max_bonds,
    )


if __name__ == "__main__":
    main()
