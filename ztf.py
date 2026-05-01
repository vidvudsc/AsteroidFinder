#!/usr/bin/env python3
"""
Download a consistent public ZTF FITS image sequence for testing a moving-object pipeline.

What it does:
  1. Queries IRSA/ZTF science-image metadata around a target RA/Dec.
  2. Filters by date range and optional filter, e.g. zr/zg/zi.
  3. Groups images by same night + field + CCD + quadrant + filter.
  4. Picks the best group with enough frames and useful time span.
  5. Downloads either full sciimg.fits files or fixed-center cutouts.

Why this matters:
  Your current asteroid pipeline links detections in pixel coordinates, so you should
  not mix different ZTF fields/CCDs/quadrants. Use one group only.

Requires:
  pip install requests astropy
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u


IRSA_HOSTS = [
    "https://irsa.ipac.caltech.edu",
    "https://irsawebops2.ipac.caltech.edu",
]
SEARCH_PATH = "/ibe/search/ztf/products/sci"
DATA_PATH = "/ibe/data/ztf/products/sci"
SEARCH_URL = f"{IRSA_HOSTS[0]}{SEARCH_PATH}"
DATA_ROOT = f"{IRSA_HOSTS[0]}{DATA_PATH}"

COLUMNS = ",".join(
    [
        "ra",
        "dec",
        "field",
        "ccdid",
        "qid",
        "filtercode",
        "imgtypecode",
        "filefracday",
        "obsdate",
        "obsjd",
        "exptime",
        "seeing",
        "airmass",
        "maglimit",
        "moonillf",
        "infobits",
    ]
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_coord(ra_text: str, dec_text: str) -> tuple[float, float]:
    """
    Accepts decimal degrees or sexagesimal, e.g.
      --ra 232.5 --dec -16
      --ra "15:30:00" --dec "-16:00:00"
    """
    try:
        ra = float(ra_text)
        dec = float(dec_text)
        return ra, dec
    except ValueError:
        pass

    coord = SkyCoord(ra_text, dec_text, unit=(u.hourangle, u.deg), frame="icrs")
    return float(coord.ra.deg), float(coord.dec.deg)


def parse_time_to_jd(text: str) -> float:
    """
    Accepts:
      2023-04-25
      2023-04-25T08:00:00
    """
    if "T" not in text and len(text.strip()) == 10:
        text = text.strip() + "T00:00:00"
    return float(Time(text, scale="utc").jd)


def angular_size_to_deg(text: str | None) -> float | None:
    if not text:
        return None
    value = text.strip().lower()
    for suffix, scale in (("arcmin", 1.0 / 60.0), ("arcsec", 1.0 / 3600.0), ("deg", 1.0)):
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * scale
    return float(value)


def next_day(date_text: str) -> str:
    d = dt.date.fromisoformat(date_text)
    return (d + dt.timedelta(days=1)).isoformat()


def build_where(start_jd: float, end_jd: float, filtercode: str | None) -> str:
    parts = [
        f"obsjd >= {start_jd:.8f}",
        f"obsjd <= {end_jd:.8f}",
        "imgtypecode = 'o'",
    ]
    if filtercode:
        parts.append(f"filtercode = '{filtercode}'")
    return " AND ".join(parts)


def query_ztf_metadata(
    ra_deg: float,
    dec_deg: float,
    start_jd: float,
    end_jd: float,
    filtercode: str | None,
    size_deg: float,
) -> list[dict[str, Any]]:
    params = {
        "POS": f"{ra_deg:.8f},{dec_deg:.8f}",
        "SIZE": f"{size_deg:.6f}",
        "INTERSECT": "CENTER",
        "COLUMNS": COLUMNS,
        "WHERE": build_where(start_jd, end_jd, filtercode),
        "ct": "csv",
    }

    last_error: Exception | None = None
    for host in IRSA_HOSTS:
        url = f"{host}{SEARCH_PATH}"
        log(f"Querying IRSA/ZTF metadata: {host}")
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            log(f"  failed: {exc}")
    else:
        assert last_error is not None
        raise last_error

    text = r.text.strip()
    if not text:
        return []

    # IRSA ct=csv is normally plain CSV. This also tolerates comment-ish lines.
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("\\")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))

    rows: list[dict[str, Any]] = []
    for row in reader:
        try:
            row["obsjd"] = float(row["obsjd"])
            row["field"] = int(row["field"])
            row["ccdid"] = int(row["ccdid"])
            row["qid"] = int(row["qid"])
            row["filefracday"] = str(row["filefracday"]).strip()
            row["filtercode"] = str(row["filtercode"]).strip()
            row["imgtypecode"] = str(row.get("imgtypecode") or "o").strip()
            row["seeing"] = safe_float(row.get("seeing"))
            row["airmass"] = safe_float(row.get("airmass"))
            row["maglimit"] = safe_float(row.get("maglimit"))
        except Exception:
            continue

        if len(row["filefracday"]) < 14:
            continue

        rows.append(row)

    rows.sort(key=lambda x: x["obsjd"])
    return rows


def safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def night_from_filefracday(row: dict[str, Any]) -> str:
    # filefracday = YYYYMMDDdddddd
    return row["filefracday"][:8]


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        night_from_filefracday(row),
        row["field"],
        row["ccdid"],
        row["qid"],
    )


def minutes_between(a_jd: float, b_jd: float) -> float:
    return abs(b_jd - a_jd) * 1440.0


def quality_score(rows: list[dict[str, Any]]) -> float:
    """
    Score a candidate sequence.
    Priorities:
      - more frames
      - useful time span
      - deeper mag limit
      - lower seeing
      - lower airmass
    """
    if not rows:
        return -1e99

    n = len(rows)
    span_min = minutes_between(rows[0]["obsjd"], rows[-1]["obsjd"])

    maglimits = [r["maglimit"] for r in rows if r.get("maglimit") is not None]
    seeings = [r["seeing"] for r in rows if r.get("seeing") is not None]
    airmasses = [r["airmass"] for r in rows if r.get("airmass") is not None]

    med_mag = median(maglimits) if maglimits else 20.0
    med_seeing = median(seeings) if seeings else 2.5
    med_airmass = median(airmasses) if airmasses else 1.5

    # Cap span contribution so very long same-night gaps do not dominate.
    span_bonus = min(span_min, 120.0)

    return (
        n * 1000.0
        + span_bonus * 2.0
        + med_mag * 10.0
        - med_seeing * 20.0
        - med_airmass * 10.0
    )


def median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def choose_best_sequence(
    rows: list[dict[str, Any]],
    min_frames: int,
    max_frames: int,
    min_span_min: float,
    max_span_min: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(group_key(row), []).append(row)

    candidates: list[list[dict[str, Any]]] = []

    for _, group in groups.items():
        group = sorted(group, key=lambda x: x["obsjd"])
        if len(group) < min_frames:
            continue

        # Try sliding windows up to max_frames.
        for i in range(len(group)):
            for j in range(i + min_frames, min(len(group), i + max_frames) + 1):
                window = group[i:j]
                span = minutes_between(window[0]["obsjd"], window[-1]["obsjd"])
                if min_span_min <= span <= max_span_min:
                    candidates.append(window)

    if candidates:
        return max(candidates, key=quality_score)

    # Fallback: choose the largest same-night/group sequence even if span is imperfect.
    fallback_groups = [sorted(g, key=lambda x: x["obsjd"]) for g in groups.values() if len(g) >= min_frames]
    if not fallback_groups:
        return []

    best = max(fallback_groups, key=quality_score)
    return best[:max_frames]


def ztf_sciimg_url(
    row: dict[str, Any],
    cutout_center: tuple[float, float] | None,
    cutout_size: str | None,
    *,
    host: str = IRSA_HOSTS[0],
) -> str:
    """
    Build URL from IRSA's documented ZTF science exposure pattern.
    """
    ffd = row["filefracday"]
    year = ffd[0:4]
    month = ffd[4:6]
    day = ffd[6:8]
    fracday = ffd[8:14]

    paddedfield = f"{int(row['field']):06d}"
    paddedccdid = f"{int(row['ccdid']):02d}"
    filtercode = row["filtercode"]
    imgtypecode = row.get("imgtypecode") or "o"
    qid = str(row["qid"])

    filename = f"ztf_{ffd}_{paddedfield}_{filtercode}_c{paddedccdid}_{imgtypecode}_q{qid}_sciimg.fits"
    url = f"{host}{DATA_PATH}/{year}/{month}{day}/{fracday}/{filename}"

    if cutout_center and cutout_size:
        ra_deg, dec_deg = cutout_center
        query = urlencode(
            {
                "center": f"{ra_deg:.8f},{dec_deg:.8f}",
                "size": cutout_size,
                "gzip": "false",
            }
        )
        url = f"{url}?{query}"

    return url


def output_filename(row: dict[str, Any], is_cutout: bool) -> str:
    ffd = row["filefracday"]
    paddedfield = f"{int(row['field']):06d}"
    paddedccdid = f"{int(row['ccdid']):02d}"
    filtercode = row["filtercode"]
    qid = str(row["qid"])
    suffix = "cutout_sciimg.fits" if is_cutout else "sciimg.fits"
    return f"ztf_{ffd}_{paddedfield}_{filtercode}_c{paddedccdid}_q{qid}_{suffix}"


def download_file(url: str, dest: Path, overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        log(f"Exists, skipping: {dest.name}")
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"Downloading {dest.name}")

    last_error: Exception | None = None
    for attempt_url in _candidate_data_urls(url):
        try:
            with requests.get(attempt_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            break
        except requests.RequestException as exc:
            last_error = exc
            log(f"  failed from {attempt_url.split('/ibe/')[0]}: {exc}")
            if tmp.exists():
                tmp.unlink()
    else:
        assert last_error is not None
        raise last_error

    tmp.replace(dest)


def _candidate_data_urls(url: str) -> list[str]:
    urls = [url]
    for host in IRSA_HOSTS:
        for existing in IRSA_HOSTS:
            if url.startswith(existing):
                candidate = host + url[len(existing) :]
                if candidate not in urls:
                    urls.append(candidate)
    return urls


def write_manifest(out_dir: Path, selected: list[dict[str, Any]], urls: list[str]) -> None:
    manifest = out_dir / "ztf_download_manifest.csv"
    fields = [
        "local_file",
        "url",
        "obsdate",
        "obsjd",
        "field",
        "ccdid",
        "qid",
        "filtercode",
        "filefracday",
        "exptime",
        "seeing",
        "airmass",
        "maglimit",
        "moonillf",
    ]

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, url in zip(selected, urls):
            out = {k: row.get(k, "") for k in fields}
            out["url"] = url
            out["local_file"] = output_filename(row, "center=" in url)
            writer.writerow(out)


def print_sequence_summary(rows: list[dict[str, Any]]) -> None:
    first = rows[0]
    last = rows[-1]
    span = minutes_between(first["obsjd"], last["obsjd"])

    print()
    print("Selected ZTF sequence")
    print("---------------------")
    print(f"Frames:      {len(rows)}")
    print(f"Night:       {night_from_filefracday(first)}")
    print(f"Field:       {first['field']}")
    print(f"CCD/QID:     c{first['ccdid']:02d} q{first['qid']}")
    print(f"Filter:      {first['filtercode']}")
    print(f"Start UTC:   {first.get('obsdate', '')}")
    print(f"End UTC:     {last.get('obsdate', '')}")
    print(f"Span:        {span:.1f} min")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a consistent public ZTF FITS sequence from IRSA.")

    p.add_argument("--ra", required=True, help="Target RA, decimal degrees or sexagesimal hours, e.g. 15:30:00")
    p.add_argument("--dec", required=True, help="Target Dec, decimal degrees or sexagesimal degrees, e.g. -16:00:00")
    p.add_argument(
    "--allow-mixed-fallback",
    action="store_true",
    help="If no strict same-field sequence is found, download the first matching metadata rows anyway. Use only for smoke testing.",
    )      
    date_group = p.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", help="UTC date YYYY-MM-DD. Searches that one UTC day.")
    date_group.add_argument("--start", help="UTC start time/date, e.g. 2023-04-01 or 2023-04-01T00:00:00")

    p.add_argument("--end", help="UTC end time/date. Required if --start is used.")
    p.add_argument("--filter", choices=["zg", "zr", "zi"], default=None, help="Optional ZTF filter.")
    p.add_argument("--size-deg", type=float, default=0.0, help="Spatial query size in degrees. 0 means target point only.")

    p.add_argument("--min-frames", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=12)
    p.add_argument("--min-span-min", type=float, default=10.0)
    p.add_argument("--max-span-min", type=float, default=180.0)

    p.add_argument("--out-dir", default="data/ztf_sequence/input")
    p.add_argument("--cutout-size", default=None, help='Download fixed-center cutouts, e.g. "20arcmin" or "0.3deg".')
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.start and not args.end:
        raise SystemExit("--end is required when using --start")

    ra_deg, dec_deg = parse_coord(args.ra, args.dec)

    if args.date:
        start_text = args.date
        end_text = next_day(args.date)
    else:
        start_text = args.start
        end_text = args.end

    start_jd = parse_time_to_jd(start_text)
    end_jd = parse_time_to_jd(end_text)

    size_deg = args.size_deg
    if size_deg <= 0 and args.cutout_size:
        size_deg = angular_size_to_deg(args.cutout_size) or 0.0

    rows = query_ztf_metadata(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        start_jd=start_jd,
        end_jd=end_jd,
        filtercode=args.filter,
        size_deg=size_deg,
    )

    print(f"Metadata rows found: {len(rows)}")
    if not rows:
        print("No ZTF science images found. Try a wider date range or remove --filter.")
        return 1

    selected = choose_best_sequence(
        rows=rows,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        min_span_min=args.min_span_min,
        max_span_min=args.max_span_min,
    )

    if not selected:
        if args.allow_mixed_fallback:
            print("WARNING: No strict same-night same-field/CCD/quadrant sequence found.")
            print("Using mixed fallback rows for smoke testing only.")
            selected = rows[: args.max_frames]
        else:
            print("No same-night same-field/CCD/quadrant/filter sequence met your minimum frame requirement.")
            print("Try --allow-mixed-fallback for a smoke test, or use a wider --start/--end range.")
            return 1

    print_sequence_summary(selected)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cutouts = bool(args.cutout_size)
    urls = [
        ztf_sciimg_url(
            row,
            cutout_center=(ra_deg, dec_deg) if use_cutouts else None,
            cutout_size=args.cutout_size,
        )
        for row in selected
    ]

    write_manifest(out_dir, selected, urls)

    if args.dry_run:
        print("Dry run. URLs:")
        for url in urls:
            print(url)
        print(f"\nManifest written to: {out_dir / 'ztf_download_manifest.csv'}")
        return 0

    for row, url in zip(selected, urls):
        dest = out_dir / output_filename(row, is_cutout=use_cutouts)
        download_file(url, dest, overwrite=args.overwrite)

    print(f"\nDownloaded {len(selected)} FITS files to:")
    print(out_dir)
    print()
    print("Then run your pipeline like:")
    print(f"python3 main.py --input {out_dir} --out-dir data/ztf_test/results --min-tracklet 3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
