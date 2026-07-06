# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Generate a Star History chart for a set of GitHub repositories.

Unlike the public ``api.star-history.com`` image endpoint, this script talks to
the GitHub REST API directly with an authenticated token, so it is not subject
to the 60 requests/hour anonymous rate limit that caused repos to silently drop
out of the chart (missing lines) and their legend avatars to break. With a token
every requested repository is fetched and plotted.

Usage:
    REPO_LIST="owner/repo,owner/repo,..." GITHUB_TOKEN=... uv run generate_chart.py

Outputs ``star-history.svg`` (light) and ``star-history-dark.svg`` (dark) next to
this script.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Number of stargazer pages sampled per repo. Keeps the request count tiny and
# constant regardless of how many stars a repo has (matches star-history's own
# sampling approach).
MAX_REQUEST_PAGES = 20
PER_PAGE = 100
# GitHub only exposes up to 400 pages of stargazers.
MAX_PAGES = 400

# Distinct, colour-blind-friendly palette (extend if REPO_LIST grows).
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _request(url: str, token: str, star_json: bool = False) -> tuple[list | dict, dict]:
    headers = {
        "Accept": "application/vnd.github.star+json"
        if star_json
        else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "star-history-generator",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        remaining = exc.headers.get("x-ratelimit-remaining", "?")
        raise urllib.error.HTTPError(
            exc.url,
            exc.code,
            f"{exc.reason} (ratelimit-remaining={remaining}): {body}",
            exc.headers,
            None,
        ) from None


def _sample_pages(page_count: int) -> tuple[list[int], bool]:
    """Return the page numbers to fetch and whether that covers every stargazer."""
    if page_count <= MAX_REQUEST_PAGES:
        return list(range(1, page_count + 1)), True
    pages = sorted(
        {
            round(1 + i * (page_count - 1) / (MAX_REQUEST_PAGES - 1))
            for i in range(MAX_REQUEST_PAGES)
        }
    )
    return pages, False


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def star_records(repo: str, token: str) -> list[tuple[datetime, int]]:
    """Return a chronological list of (timestamp, cumulative_star_count) points."""
    info, _ = _request(f"https://api.github.com/repos/{repo}", token)
    total = int(info.get("stargazers_count", 0))
    if total == 0:
        return []

    page_count = min((total + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    pages, fetched_all = _sample_pages(page_count)

    records: list[tuple[datetime, int]] = []
    if fetched_all:
        stars: list[datetime] = []
        for page in pages:
            data, _ = _request(
                f"https://api.github.com/repos/{repo}/stargazers"
                f"?per_page={PER_PAGE}&page={page}",
                token,
                star_json=True,
            )
            stars.extend(_parse_ts(item["starred_at"]) for item in data)
        stars.sort()
        # Thin out to at most MAX_REQUEST_PAGES points for a clean line.
        step = max(1, len(stars) // MAX_REQUEST_PAGES)
        for i in range(0, len(stars), step):
            records.append((stars[i], i + 1))
        if records and records[-1][1] != len(stars):
            records.append((stars[-1], len(stars)))
    else:
        for page in pages:
            data, _ = _request(
                f"https://api.github.com/repos/{repo}/stargazers"
                f"?per_page={PER_PAGE}&page={page}",
                token,
                star_json=True,
            )
            if data:
                count = max(PER_PAGE * (page - 1), 1)
                records.append((_parse_ts(data[0]["starred_at"]), count))

    # Extend the line to "now" with the current total.
    records.append((datetime.now(timezone.utc), total))
    records.sort(key=lambda r: r[0])
    return records


def render(series: list[tuple[str, list[tuple[datetime, int]]]], out: Path, *, dark: bool) -> None:
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#30363d" if dark else "#d0d7de"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for (repo, records), colour in zip(series, PALETTE):
        if not records:
            continue
        xs = [ts for ts, _ in records]
        ys = [max(count, 1) for _, count in records]
        ax.plot(xs, ys, label=repo, color=colour, linewidth=2.2, solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_ylabel("GitHub Stars", color=fg)
    ax.set_title("Star History", color=fg, fontsize=15, fontweight="bold", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()

    ax.grid(True, which="major", color=grid, linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(grid)

    legend = ax.legend(loc="upper left", fontsize=8, frameon=True)
    legend.get_frame().set_edgecolor(grid)
    legend.get_frame().set_facecolor("none")
    for text in legend.get_texts():
        text.set_color(fg)

    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repos = [r.strip() for r in os.environ.get("REPO_LIST", "").split(",") if r.strip()]
    if not repos:
        print("REPO_LIST is empty", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("warning: no GITHUB_TOKEN set; falling back to anonymous rate limits", file=sys.stderr)

    series: list[tuple[str, list[tuple[datetime, int]]]] = []
    failures = 0
    for repo in repos:
        try:
            records = star_records(repo, token)
            series.append((repo, records))
            print(f"ok: {repo} ({records[-1][1] if records else 0} stars, {len(records)} points)")
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as exc:
            failures += 1
            print(f"error: {repo}: {exc}", file=sys.stderr)

    if not series:
        print("no data fetched for any repository", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    render(series, here / "star-history.svg", dark=False)
    render(series, here / "star-history-dark.svg", dark=True)
    print(f"wrote charts for {len(series)} repos ({failures} failed)")
    # Fail the job if any repo dropped out, so we notice instead of silently
    # shipping an incomplete chart like the old anonymous API did.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
