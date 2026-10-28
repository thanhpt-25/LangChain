"""Download the latest 10-K of each ticker from SEC EDGAR into data/raw/."""
import json
import time

import requests

from finassist.config import TICKERS, get_settings

CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810"}


def main() -> None:
    s = get_settings()
    s.raw_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": s.sec_user_agent}
    for t in TICKERS:
        sub = requests.get(f"https://data.sec.gov/submissions/CIK{CIKS[t]}.json", headers=headers, timeout=30)
        sub.raise_for_status()
        recent = sub.json()["filings"]["recent"]
        idx = next(i for i, f in enumerate(recent["form"]) if f == "10-K")
        acc = recent["accessionNumber"][idx].replace("-", "")
        doc = recent["primaryDocument"][idx]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(CIKS[t])}/{acc}/{doc}"
        time.sleep(0.3)
        html = requests.get(url, headers=headers, timeout=60)
        html.raise_for_status()
        (s.raw_dir / f"{t}.html").write_text(html.text)
        meta = {
            "ticker": t,
            "fiscal_year": int(recent["reportDate"][idx][:4]),
            "source_url": url,
            "filing_date": recent["filingDate"][idx],
        }
        (s.raw_dir / f"{t}.json").write_text(json.dumps(meta, indent=2))
        print(t, meta)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
