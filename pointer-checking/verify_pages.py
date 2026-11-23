"""Collect independent repository-page evidence for a pilot review."""

import argparse
from collections import defaultdict
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import quote

import requests

from check_pointers import utc_now


class PageEvidence(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.title = ""
        self.title_complete = False
        self.canonical = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title" and not self.title_complete:
            self.in_title = True
        if tag == "link" and attrs.get("rel") == "canonical":
            self.canonical = attrs.get("href")

    def handle_endtag(self, tag):
        if tag == "title":
            if self.in_title:
                self.title_complete = True
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def fetch_page(session, url):
    result = {"url": url, "checked_at": utc_now()}
    time.sleep(0.25)
    try:
        response = session.get(url, timeout=30)
        page = PageEvidence()
        page.feed(response.text)
        result.update(http_status=response.status_code, final_url=response.url,
                      title=page.title, canonical=page.canonical,
                      redirects=[{"status": r.status_code, "url": r.url}
                                 for r in response.history])
    except requests.RequestException:
        result["error"] = "request_error"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--models", type=int, default=10)
    args = parser.parse_args()
    if args.models < 1:
        parser.error("models must be positive")
    with (args.pilot_dir / "pointers.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in rows:
        groups[row["model_id"]].append(row)
    # Include ambiguous/changed references before straightforward successes.
    selected = sorted(groups, key=lambda mid: (
        not any(r["status"] != "metadata_accessible" or r["id_changed"] == "True"
                for r in groups[mid]), mid))[:args.models]
    session = requests.Session()
    pages = {}
    records = []
    for model_id in selected:
        model_url = "https://huggingface.co/" + quote(model_id, safe="/")
        record = {"model_id": model_id, "source_page": fetch_page(session, model_url),
                  "references": []}
        for row in groups[model_id]:
            prefix = "datasets/" if row["dependency_type"] == "dataset" else ""
            url = "https://huggingface.co/" + prefix + quote(row["dependency_id"], safe="/")
            if url not in pages:
                pages[url] = fetch_page(session, url)
            reference = {"dependency_type": row["dependency_type"],
                "dependency_id": row["dependency_id"], "source_tags": json.loads(row["source_tags"]),
                "api_status": row["status"], "api_resolved_id": row["resolved_id"],
                "page": pages[url]}
            if row["resolved_id"] and row["resolved_id"] != row["dependency_id"]:
                resolved_url = "https://huggingface.co/" + prefix + quote(row["resolved_id"], safe="/")
                if resolved_url not in pages:
                    pages[resolved_url] = fetch_page(session, resolved_url)
                reference["resolved_page"] = pages[resolved_url]
            record["references"].append(reference)
        records.append(record)
        # Save after each model so the evidence survives interrupted runs.
        (args.pilot_dir / "page_verification.json").write_text(
            json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"Verified pages for {model_id}", flush=True)


if __name__ == "__main__":
    main()
