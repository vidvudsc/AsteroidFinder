from __future__ import annotations

import csv
import html
from pathlib import Path


def generate_html_report(output_dir: str | Path, path: str | Path | None = None) -> Path:
    """Generate a single HTML report for a demo output directory."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(path) if path is not None else out_dir / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    hot_rows = _read_csv(out_dir / "hot_pixel_report.csv")
    known_summary = _read_csv(out_dir / "known_objects_summary.csv")
    alignment = _read_csv(out_dir / "alignment_report.csv")
    forced = _read_csv(out_dir / "known_object_forced_photometry.csv")

    report_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'>",
                "<title>AsteroidFinder Report</title>",
                "<style>",
                ":root{color-scheme:light} body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:0;line-height:1.38;color:#182026;background:#eef0ed}",
                "header{padding:28px 34px;background:#1f2930;color:white} main{padding:24px 34px;max-width:1500px;margin:auto}",
                "h1,h2,h3{margin:0 0 12px} section{margin:0 0 30px}.panel{background:white;border:1px solid #d1d5d0;border-radius:8px;padding:16px}",
                "table{border-collapse:collapse;background:white;width:100%;font-size:13px} td,th{border:1px solid #d4d8d2;padding:5px 7px;text-align:left} th{background:#eef1ed;position:sticky;top:0}",
                ".tablewrap{max-height:420px;overflow:auto;border:1px solid #d4d8d2}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}",
                "img{max-width:100%;border:1px solid #879087;background:#111;border-radius:6px}.muted{color:#62706b}.links a{display:inline-block;margin:0 12px 8px 0;color:#075985}",
                ".metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.metric{background:white;border:1px solid #d1d5d0;border-radius:8px;padding:12px}.metric b{font-size:24px;display:block}",
                "</style></head><body>",
                "<header><h1>AsteroidFinder Report</h1>",
                f"<p class='muted'>Output folder: {html.escape(str(out_dir))}</p></header><main>",
                _metrics_section(hot_rows, known_summary, forced, alignment),
                _links_section(out_dir),
                _image_section("Blink", [out_dir / "blink_aligned.gif", out_dir / "blink_aligned_inverted.gif", out_dir / "known_objects_annotated.gif"], out_dir),
                _image_section("Stack And Masks", [out_dir / "04_previews" / "stack_median.jpg", out_dir / "07_inverted" / "stack_median_inverted.jpg", out_dir / "06_hot_pixel_masks" / "hot_pixels_001.png"], out_dir),
                _table_section("Hot Pixels", hot_rows),
                _table_section("Alignment", alignment),
                _table_section("Known Objects Summary", known_summary),
                _table_section("Forced Known-Object Photometry", forced[:40]),
                "</main></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    return report_path


def _metrics_section(
    hot_rows: list[dict[str, str]],
    known_summary: list[dict[str, str]],
    forced: list[dict[str, str]],
    alignment: list[dict[str, str]],
) -> str:
    total_hot = sum(int(row.get("hot_pixels", "0") or 0) for row in hot_rows)
    rms_values = [float(row["rms_error_px"]) for row in alignment if row.get("rms_error_px")]
    median_rms = "" if not rms_values else f"{sorted(rms_values)[len(rms_values)//2]:.3f}px"
    metrics = [
        ("Hot replacements", str(total_hot)),
        ("Known objects", str(len(known_summary))),
        ("Forced measurements", str(len(forced))),
        ("Median align RMS", median_rms or "n/a"),
    ]
    cards = "".join(f"<div class='metric'><span>{html.escape(label)}</span><b>{html.escape(value)}</b></div>" for label, value in metrics)
    return f"<section class='metrics'>{cards}</section>"


def _links_section(out_dir: Path) -> str:
    files = [
        "hot_pixel_report.csv",
        "hot_pixel_coordinates.csv",
        "alignment_report.csv",
        "known_objects.csv",
        "known_objects_summary.csv",
        "known_object_forced_photometry.csv",
        "mpc_observations.txt",
        "tracks.csv",
    ]
    links = []
    for name in files:
        if (out_dir / name).exists():
            links.append(f"<a href='{html.escape(name)}'>{html.escape(name)}</a>")
    return f"<section><h2>Files</h2><p class='links'>{''.join(links)}</p></section>"


def _image_section(title: str, paths: list[Path], base_dir: Path) -> str:
    blocks = []
    for path in paths:
        if path.exists():
            src = str(path.relative_to(base_dir))
            blocks.append(
                f"<div class='panel'><h3>{html.escape(path.name)}</h3><img src='{html.escape(src)}'></div>"
            )
    if not blocks:
        return ""
    return f"<section><h2>{html.escape(title)}</h2><div class='grid'>{''.join(blocks)}</div></section>"


def _table_section(title: str, rows: list[dict[str, str]]) -> str:
    if not rows:
        return f"<section class='panel'><h2>{html.escape(title)}</h2><p class='muted'>No rows.</p></section>"
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(key)}</th>" for key in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(row.get(key, ''))}</td>" for key in headers) + "</tr>")
    return f"<section class='panel'><h2>{html.escape(title)}</h2><div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div></section>"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
