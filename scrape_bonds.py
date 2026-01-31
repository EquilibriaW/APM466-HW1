#!/usr/bin/env python3
"""
Scrape Markets Insider bond finder results and bond snapshot details, then
export to CSV and JSON. Intended for the Canada Gov bond screens in the prompt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URLS = {
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

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def normalize_key(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def add_or_replace_param(url: str, **params: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update(params)
    new_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def fetch_html(
    session: requests.Session,
    url: str,
    *,
    retries: int = 3,
    sleep_s: float = 0.5,
    use_selenium: bool = False,
    driver=None,
    wait_css: str = "table",
) -> str:
    if use_selenium:
        return fetch_html_selenium(driver, url, wait_css=wait_css)
    for attempt in range(1, retries + 1):
        resp = session.get(url, headers=DEFAULT_HEADERS, timeout=30)
        if resp.status_code == 200:
            resp.encoding = resp.apparent_encoding or resp.encoding
            return resp.text
        # polite backoff for 429/5xx
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(sleep_s * attempt)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def build_webdriver(*, headless: bool, driver_path: Optional[str]):
    # Lazy import so the script works without selenium unless explicitly requested.
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1400,900")

    resolved_path = driver_path or shutil.which("chromedriver")
    if not resolved_path:
        raise RuntimeError(
            "chromedriver not found. Install it or pass --driver-path /path/to/chromedriver."
        )
    service = Service(resolved_path)
    return webdriver.Chrome(service=service, options=options)


def fetch_html_selenium(driver, url: str, *, wait_css: str = "table") -> str:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    driver.get(url)
    if wait_css:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_css)))
    return driver.page_source


def pick_finder_table(tables: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    for df in tables:
        cols = {str(c).strip().lower() for c in df.columns}
        if "maturity" in cols and ("coupon" in cols or "yield" in cols):
            return df
    return tables[0] if tables else None


def extract_links_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    best_table = None
    best_rows = 0
    for table in tables:
        rows = table.find_all("tr")
        link_count = len(table.find_all("a", href=True))
        if link_count and len(rows) > best_rows:
            best_rows = len(rows)
            best_table = table

    if not best_table:
        return []

    links = []
    tbody = best_table.find("tbody")
    rows = tbody.find_all("tr") if tbody else best_table.find_all("tr")[1:]
    for row in rows:
        a = row.find("a", href=True)
        if not a:
            continue
        href = a["href"].strip()
        if href.startswith("/"):
            href = "https://markets.businessinsider.com" + href
        links.append(href)
    return links


def parse_finder_page(html: str) -> Tuple[pd.DataFrame, List[str]]:
    tables = pd.read_html(html)
    df = pick_finder_table(tables)
    if df is None:
        return pd.DataFrame(), []
    df = df.copy()
    df.columns = [normalize_key(str(c)) for c in df.columns]
    links = extract_links_from_html(html)
    if links:
        if len(links) == len(df):
            df["detail_url"] = links
        else:
            # Best-effort alignment; keep list for later inspection.
            df["detail_url"] = links[: len(df)] + [None] * max(0, len(df) - len(links))
    return df, links


def scrape_finder(
    session: requests.Session,
    base_url: str,
    maturity_bucket: str,
    *,
    max_pages: int = 50,
    sleep_s: float = 0.5,
    use_selenium: bool = False,
    driver=None,
) -> pd.DataFrame:
    all_rows: List[pd.DataFrame] = []
    seen_first_link: Optional[str] = None
    for page in range(1, max_pages + 1):
        url = add_or_replace_param(base_url, p=str(page))
        html = fetch_html(
            session,
            url,
            sleep_s=sleep_s,
            use_selenium=use_selenium,
            driver=driver,
            wait_css="table",
        )
        df, links = parse_finder_page(html)
        if df.empty:
            break

        first_link = links[0] if links else None
        if seen_first_link and first_link == seen_first_link:
            # Pagination looped; stop.
            break
        if first_link and page == 1:
            seen_first_link = first_link

        df["maturity_bucket"] = maturity_bucket
        df["source_url"] = base_url
        df["page"] = page
        all_rows.append(df)
        time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def parse_snapshot(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    data: Dict[str, str] = {}

    # Capture a headline name if available.
    title = soup.find("h1")
    if title:
        data["bond_name"] = title.get_text(" ", strip=True)

    # Parse key-value rows.
    for row in soup.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) != 2:
            continue
        key = cells[0].get_text(" ", strip=True)
        val = cells[1].get_text(" ", strip=True)
        if not key or not val:
            continue
        norm_key = normalize_key(key)
        # Preserve first occurrence if duplicates exist.
        if norm_key not in data:
            data[norm_key] = val

    # Try to capture the current price shown on the page.
    price_block = soup.find("div", class_="price-section__values")
    if price_block:
        price_span = price_block.find("span")
        if price_span:
            data.setdefault("last_price", price_span.get_text(" ", strip=True))

    return data


def scrape_snapshot(session: requests.Session, url: str, *, sleep_s: float = 0.5) -> Dict[str, str]:
    html = fetch_html(session, url, sleep_s=sleep_s)
    time.sleep(sleep_s)
    return parse_snapshot(html)


def scrape_all(
    *,
    base_urls: Dict[str, str],
    max_pages: int,
    sleep_s: float,
    include_details: bool,
    use_selenium: bool,
    headless: bool,
    driver_path: Optional[str],
) -> pd.DataFrame:
    session = requests.Session()
    driver = None
    if use_selenium:
        driver = build_webdriver(headless=headless, driver_path=driver_path)

    finder_frames: List[pd.DataFrame] = []
    try:
        for bucket, url in base_urls.items():
            df = scrape_finder(
                session,
                url,
                bucket,
                max_pages=max_pages,
                sleep_s=sleep_s,
                use_selenium=use_selenium,
                driver=driver,
            )
            if not df.empty:
                finder_frames.append(df)
    finally:
        if driver:
            driver.quit()

    if not finder_frames:
        return pd.DataFrame()

    finder = pd.concat(finder_frames, ignore_index=True)
    finder = finder.drop_duplicates(subset=["detail_url"]).reset_index(drop=True)
    finder["scrape_ts"] = dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    if not include_details:
        return finder

    records: List[Dict[str, str]] = []
    driver = None
    if use_selenium:
        driver = build_webdriver(headless=headless, driver_path=driver_path)
    try:
        for _, row in finder.iterrows():
            detail_url = row.get("detail_url")
            record = row.to_dict()
            if detail_url:
                try:
                    html = fetch_html(
                        session,
                        detail_url,
                        sleep_s=sleep_s,
                        use_selenium=use_selenium,
                        driver=driver,
                        wait_css="table",
                    )
                    details = parse_snapshot(html)
                    record.update(details)
                except Exception as exc:
                    record["detail_error"] = str(exc)
            records.append(record)
    finally:
        if driver:
            driver.quit()

    return pd.DataFrame.from_records(records)


def write_outputs(df: pd.DataFrame, outdir: Path, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{prefix}.csv"
    json_path = outdir / f"{prefix}.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)


def parse_base_urls(args: argparse.Namespace) -> Dict[str, str]:
    if not args.base_url:
        return BASE_URLS.copy()
    parsed: Dict[str, str] = {}
    for item in args.base_url:
        if "=" not in item:
            raise ValueError("--base-url must be in the form label=url")
        label, url = item.split("=", 1)
        parsed[label.strip()] = url.strip()
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scrape Markets Insider bond finder data.")
    parser.add_argument("--outdir", default="data", help="Output directory for CSV/JSON.")
    parser.add_argument("--max-pages", type=int, default=50, help="Max pages per finder query.")
    parser.add_argument("--sleep", type=float, default=0.5, help="Sleep between requests (seconds).")
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="Only scrape finder results; skip bond detail pages.",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        help="Override base URLs. Format: label=https://... (repeatable).",
    )
    parser.add_argument(
        "--selenium",
        action="store_true",
        help="Use Selenium + Chrome instead of requests (for JS-rendered pages).",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run Chrome with a visible window (only with --selenium).",
    )
    parser.add_argument(
        "--driver-path",
        help="Path to chromedriver (only with --selenium).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    base_urls = parse_base_urls(args)
    include_details = not args.skip_details
    use_selenium = bool(args.selenium)
    headless = not args.no_headless

    df = scrape_all(
        base_urls=base_urls,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
        include_details=include_details,
        use_selenium=use_selenium,
        headless=headless,
        driver_path=args.driver_path,
    )
    if df.empty:
        print("No data scraped. Check your URLs or network access.")
        return

    outdir = Path(args.outdir)
    prefix = "bonds_master" if include_details else "bonds_list"
    write_outputs(df, outdir, prefix)
    print(f"Wrote {len(df)} rows to {outdir}/{prefix}.csv and .json")


if __name__ == "__main__":
    main()
