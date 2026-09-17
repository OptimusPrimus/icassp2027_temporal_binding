#!/usr/bin/env python3
"""Search and download long Freesound ambience recordings for synthetic SED.

Requires an OAuth2 access token in FREESOUND_OAUTH_TOKEN. Original-quality
downloads cannot be made with a Freesound API key alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_ROOT = "https://freesound.org/apiv2"

# Queries are intentionally broad. Foreground-class compatibility is handled
# after download by build_allowed_event_classes.py, not during Freesound search.
CATEGORIES: dict[str, dict[str, Any]] = {
    "home_living_bedroom": {
        "label": "Home—living/bedroom",
        "queries": [
            "home ambience",
            "apartment ambience",
            "living room ambience",
            "bedroom ambience",
            "room tone",
            "indoor ambience",
        ],
    },
    "kitchen_dining": {
        "label": "Kitchen/dining",
        "queries": [
            "kitchen ambience",
            "dining room ambience",
            "restaurant ambience",
            "cafeteria ambience",
            "cafe ambience",
            "indoor dining ambience",
        ],
    },
    "bathroom_laundry": {
        "label": "Bathroom/laundry",
        "queries": [
            "bathroom ambience",
            "laundry ambience",
            "laundry room ambience",
            "bathroom room tone",
            "utility room ambience",
            "interior room tone",
        ],
    },
    "office_public_interior": {
        "label": "Office/public interior",
        "queries": [
            "office ambience",
            "public interior ambience",
            "library ambience",
            "lobby ambience",
            "hallway ambience",
            "indoor crowd ambience",
        ],
    },
    "urban_suburban_outdoors": {
        "label": "Urban/suburban outdoors",
        "queries": [
            "city ambience",
            "urban ambience",
            "street ambience",
            "suburban ambience",
            "neighborhood ambience",
            "outdoor city atmosphere",
        ],
    },
    "transport_industrial": {
        "label": "Transport/industrial",
        "queries": [
            "transport ambience",
            "station ambience",
            "subway ambience",
            "airport ambience",
            "industrial ambience",
            "factory ambience",
        ],
    },
    "rural_farm": {
        "label": "Rural/farm",
        "queries": [
            "rural ambience",
            "countryside ambience",
            "farm ambience",
            "field ambience",
            "meadow ambience",
            "country atmosphere",
        ],
    },
    "forest_woodland": {
        "label": "Forest/woodland",
        "queries": [
            "forest ambience",
            "woodland ambience",
            "woods ambience",
            "nature ambience",
            "jungle ambience",
            "trees ambience",
        ],
    },
    "park_garden": {
        "label": "Park/garden",
        "queries": [
            "park ambience",
            "garden ambience wind",
            "garden ambience",
            "public park atmosphere",
            "city park ambience",
            "outdoor garden atmosphere",
        ],
    },
    "waterside_coastal": {
        "label": "Waterside/coastal",
        "queries": [
            "harbor ambience",
            "waterside ambience",
            "coastal ambience",
            "seaside ambience",
            "riverside ambience",
            "lake ambience",
        ],
    },
}

FIELDS = ",".join(
    ["id", "name", "url", "tags", "description", "username", "license", "duration",
     "type", "filesize", "samplerate", "channels", "num_downloads", "avg_rating", "download"]
)

ARTIFICIAL_SOUND_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bartificial\b",
        r"\bsynthetic\b",
        r"\bsynth(?:esized|esised|etic|esis|esizer|esiser)?\b",
        r"\bai[- ]?generated\b",
        r"\ba\.i\.[- ]?generated\b",
        r"\bgenerated\b",
        r"\balgorithmic\b",
        r"\bprocedural\b",
        r"\bsimulat(?:ed|ion)\b",
        r"\bemulat(?:ed|ion)\b",
        r"\bfake\b",
        r"\bvirtual\b",
        r"\bsound[- ]design(?:ed)?\b",
    ]
]


def api_json(path: str, token: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_ROOT}{path}?{urlencode(params)}"
    req = Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "synthetic-sed-ambience-collector/1.0"})
    try:
        with urlopen(req, timeout=45) as response:
            return json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Freesound API returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Freesound: {exc.reason}") from exc


def sound_search_text(sound: dict[str, Any]) -> str:
    tags = sound.get("tags") or []
    if isinstance(tags, list):
        tag_text = " ".join(str(tag) for tag in tags)
    else:
        tag_text = str(tags)
    return " ".join(
        [
            str(sound.get("name") or ""),
            tag_text,
            str(sound.get("description") or ""),
        ]
    )


def is_probably_artificial(sound: dict[str, Any]) -> bool:
    text = sound_search_text(sound)
    return any(pattern.search(text) for pattern in ARTIFICIAL_SOUND_PATTERNS)


def find_candidates(category: dict[str, Any], token: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    page_size = min(150, max(args.limit * 4, 50))
    queries = list(dict.fromkeys(category["queries"]))
    accepted: list[dict[str, Any]] = []
    seen_ids = set()
    for query in queries:
        data = api_json("/search/", token, {
            "query": query,
            "filter": f"duration:[{args.min_duration} TO 600]",
            "sort": args.sort,
            "fields": FIELDS,
            "page_size": page_size,
            "group_by_pack": 1,
        })
        for sound in data.get("results", []):
            sound_id = sound.get("id")
            if sound_id in seen_ids:
                continue
            seen_ids.add(sound_id)
            if is_probably_artificial(sound):
                continue
            sound["search_query"] = query
            accepted.append(sound)
            if len(accepted) >= args.limit:
                return accepted
    return accepted


def safe_name(value: str, max_len: int = 100) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (clean or "sound")[:max_len]


def existing_download_for_sound(sound_id: int, output: Path) -> Path | None:
    for path in sorted(output.glob(f"**/{sound_id}_*")):
        if path.is_file() and not path.name.endswith(".part"):
            return path
    return None


def download_sound(sound: dict[str, Any], output: Path, directory: Path, token: str, overwrite: bool) -> tuple[Path, bool]:
    sound_id = int(sound["id"])
    suffix = "." + safe_name(str(sound.get("type") or "audio").lower(), 10)
    destination = directory / f"{sound_id}_{safe_name(str(sound.get('name', 'sound')))}{suffix}"
    if not overwrite:
        existing = existing_download_for_sound(sound_id, output)
        if existing is not None:
            return existing, True
    req = Request(
        sound.get("download") or f"{API_ROOT}/sounds/{sound_id}/download/",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "synthetic-sed-ambience-collector/1.0"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(req, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
        return destination, False
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_metadata(output: Path, rows: list[dict[str, Any]]) -> None:
    (output / "metadata.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = [
        "category", "category_label", "id", "name", "duration", "username",
        "license", "url", "local_file", "search_query",
    ]
    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("freesound_ambience"))
    parser.add_argument("--limit", type=int, default=20, help="candidates per category")
    parser.add_argument("--min-duration", type=float, default=60.0, help="hard minimum seconds (default: 60)")
    parser.add_argument("--sort", choices=["duration_desc", "downloads_desc", "rating_desc", "score"], default="score")
    parser.add_argument("--category", action="append", choices=sorted(CATEGORIES), help="only process this category; repeatable")
    parser.add_argument("--search-only", action="store_true", help="save candidate metadata without downloading audio")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between downloads")
    args = parser.parse_args()
    if not 1 <= args.limit:
        parser.error("--limit must be between larget than 1")
    if args.min_duration < 10:
        parser.error("--min-duration must be at least 10 seconds")
    if args.delay < 0:
        parser.error("--delay cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    token = os.environ.get("FREESOUND_OAUTH_TOKEN", "").strip()
    if not token:
        print("error: set FREESOUND_OAUTH_TOKEN to a Freesound OAuth2 access token", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    keys = args.category or list(CATEGORIES)
    metadata: list[dict[str, Any]] = []
    for key in keys:
        category = CATEGORIES[key]
        print(f"[{key}] searching…", flush=True)
        candidates = find_candidates(category, token, args)
        directory = args.output / key
        directory.mkdir(exist_ok=True)
        print(f"[{key}] selected {len(candidates)} candidate(s)", flush=True)
        for sound in candidates:
            row = dict(sound)
            row.update({"category": key, "category_label": category["label"], "local_file": ""})
            if not args.search_only:
                try:
                    path, skipped = download_sound(sound, args.output, directory, token, args.overwrite)
                    row["local_file"] = str(path.relative_to(args.output))
                    action = "skipped existing" if skipped else "downloaded"
                    print(f"  {action} {sound['id']}: {sound['name']}", flush=True)
                except Exception as exc:
                    row["download_error"] = str(exc)
                    print(f"  failed {sound['id']}: {exc}", file=sys.stderr, flush=True)
                time.sleep(args.delay)
            metadata.append(row)
        write_metadata(args.output, metadata)

    print(f"Metadata: {args.output / 'metadata.csv'} and {args.output / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
