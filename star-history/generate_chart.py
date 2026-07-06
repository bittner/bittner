# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib"]
# ///
"""Generate a Star History chart for a set of GitHub repositories.

Unlike the public ``api.star-history.com`` image endpoint, this script talks to
the GitHub REST API directly with an authenticated token, so it is not subject
to the 60 requests/hour anonymous rate limit that caused repos to silently drop
out of the chart (missing lines) and their legend avatars to break. With a token
every requested repository is fetched and plotted.

To keep the request count tiny and constant regardless of a repo's star count,
at most ``MAX_REQUEST_PAGES`` stargazer pages are sampled per repo (the same
approach star-history uses); GitHub itself only exposes up to ``MAX_PAGES``.
``PALETTE`` is a distinct, colour-blind-friendly set of line colours; extend it
if ``REPO_LIST`` grows beyond its length.

Usage:
    REPO_LIST="owner/repo,owner/repo,..." GITHUB_TOKEN=... uv run generate_chart.py

Outputs two transparent hand-drawn (xkcd) SVGs next to this script, a light and
a dark variant, so the README can serve the right one per theme via
``<picture>``. The xkcd look renders authentically because the ``xkcd Script``
font is vendored under ``fonts/``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import (  # noqa: E402
    AnnotationBbox,
    DrawingArea,
    HPacker,
    OffsetImage,
    TextArea,
    VPacker,
)
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

_FONT = Path(__file__).resolve().parent / "fonts" / "xkcd-script.ttf"
if _FONT.exists():
    fm.fontManager.addfont(str(_FONT))

MAX_REQUEST_PAGES = 20
PER_PAGE = 100
MAX_PAGES = 400
PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#b07aa1",
    "#76b7b2",
    "#edc948",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]
Series = list[tuple[str, list[tuple[datetime, int]], "np.ndarray | None"]]


def _request(url: str, token: str, star_json: bool = False) -> tuple[Any, dict]:
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
    """Return the page numbers to fetch and whether that covers every stargazer.

    Small repos are fetched in full; large ones are sampled down to at most
    ``MAX_REQUEST_PAGES`` evenly spaced pages spanning the first and last.

    >>> _sample_pages(3)
    ([1, 2, 3], True)
    >>> pages, fetched_all = _sample_pages(1000)
    >>> fetched_all
    False
    >>> len(pages) <= MAX_REQUEST_PAGES, pages[0], pages[-1]
    (True, 1, 1000)
    """
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
    """Parse a GitHub ISO-8601 UTC timestamp.

    >>> _parse_ts("2020-01-02T03:04:05Z")
    datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    """
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _stargazer_page(repo: str, token: str, page: int) -> list[dict]:
    """Return one page of stargazers, each carrying its ``starred_at`` time."""
    data, _ = _request(
        f"https://api.github.com/repos/{repo}/stargazers?per_page={PER_PAGE}&page={page}",
        token,
        star_json=True,
    )
    return data


def _exact_records(
    repo: str, token: str, pages: list[int]
) -> list[tuple[datetime, int]]:
    """Build points from every stargazer, thinned to at most one per page."""
    stars = sorted(
        _parse_ts(item["starred_at"])
        for page in pages
        for item in _stargazer_page(repo, token, page)
    )
    step = max(1, len(stars) // MAX_REQUEST_PAGES)
    records = [(stars[i], i + 1) for i in range(0, len(stars), step)]
    if records and records[-1][1] != len(stars):
        records.append((stars[-1], len(stars)))
    return records


def _sampled_records(
    repo: str, token: str, pages: list[int]
) -> list[tuple[datetime, int]]:
    """Approximate points from the first stargazer of each sampled page."""
    records = []
    for page in pages:
        data = _stargazer_page(repo, token, page)
        if data:
            records.append(
                (_parse_ts(data[0]["starred_at"]), max(PER_PAGE * (page - 1), 1))
            )
    return records


def star_records(repo: str, token: str) -> list[tuple[datetime, int]]:
    """Return chronological (timestamp, cumulative_star_count) points for a repo.

    A final point pins the line to the current total at fetch time.
    """
    info, _ = _request(f"https://api.github.com/repos/{repo}", token)
    total = int(info.get("stargazers_count", 0))
    if total == 0:
        return []

    page_count = min((total + PER_PAGE - 1) // PER_PAGE, MAX_PAGES)
    pages, fetched_all = _sample_pages(page_count)
    build = _exact_records if fetched_all else _sampled_records
    records = build(repo, token, pages)

    records.append((datetime.now(timezone.utc), total))
    records.sort(key=lambda r: r[0])
    return records


def owner_avatar(repo: str, token: str) -> np.ndarray | None:
    """Return the repo owner's avatar as an RGBA array, or None if unavailable."""
    info, _ = _request(f"https://api.github.com/repos/{repo}", token)
    url = info.get("owner", {}).get("avatar_url")
    if not url:
        return None
    sized = url + ("&" if "?" in url else "?") + "s=48"
    req = urllib.request.Request(
        sized, headers={"User-Agent": "star-history-generator"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


def _legend_row(repo: str, avatar, colour: str, fg: str) -> HPacker:
    """Build one legend row (colour swatch, avatar, name) linking to the repo."""
    url = f"https://github.com/{repo}"
    swatch = DrawingArea(10, 10, 0, 0)
    square = Rectangle((0, 0), 9, 9, color=colour)
    square.set_url(url)
    swatch.add_artist(square)
    children = [swatch]
    if avatar is not None:
        icon = OffsetImage(avatar, zoom=0.28)
        icon.get_children()[0].set_url(url)
        children.append(icon)
    name = TextArea(repo, textprops={"color": fg, "fontsize": 8})
    name.get_children()[0].set_url(url)
    children.append(name)
    return HPacker(children=children, align="center", pad=0, sep=4)


def _legend(ax, series: Series, *, fg: str, grid: str, legend_bg: str) -> None:
    """Anchor a star-history-style legend with an avatar icon per repository."""
    rows = [
        _legend_row(repo, avatar, colour, fg)
        for (repo, records, avatar), colour in zip(series, PALETTE, strict=False)
        if records
    ]
    box = VPacker(children=rows, align="left", pad=0, sep=3)
    anchored = AnnotationBbox(
        box,
        (0.0, 1.0),
        xycoords="axes fraction",
        box_alignment=(0, 1),
        frameon=True,
        pad=0.4,
        bboxprops={"edgecolor": grid, "facecolor": legend_bg},
    )
    ax.add_artist(anchored)


def _draw(series: Series, out: Path, *, fg: str, grid: str, legend_bg: str) -> None:
    """Draw and save one transparent chart inked in ``fg``.

    The background is transparent so the README's ``<picture>`` element can
    serve the light or dark variant to match the viewer's theme.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for (_repo, records, _avatar), colour in zip(series, PALETTE, strict=False):
        if not records:
            continue
        xs = [ts for ts, _ in records]
        ys = [max(count, 1) for _, count in records]
        ax.plot(xs, ys, color=colour, linewidth=2.4, solid_capstyle="round")

    ax.set_yscale("log")
    ax.set_ylabel("GitHub Stars", color=fg)
    title = ax.set_title("Star History", color=fg, fontsize=18, loc="center", pad=14)
    title.set_path_effects([pe.withStroke(linewidth=1.4, foreground=fg)])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    ax.grid(True, which="major", color=grid, linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=fg)
    for spine in ax.spines.values():
        spine.set_color(fg)

    _legend(ax, series, fg=fg, grid=grid, legend_bg=legend_bg)

    fig.tight_layout()
    fig.savefig(out, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def render(series: Series, out: Path, *, dark: bool) -> None:
    """Render the hand-drawn (xkcd) chart, inked for a dark or light theme."""
    fg = "#c9d1d9" if dark else "#24292f"
    grid = "#30363d" if dark else "#d0d7de"
    legend_bg = "#0d1117" if dark else "#ffffff"
    with plt.xkcd():
        _draw(series, out, fg=fg, grid=grid, legend_bg=legend_bg)


def main() -> int:
    repos = [r.strip() for r in os.environ.get("REPO_LIST", "").split(",") if r.strip()]
    if not repos:
        print("REPO_LIST is empty", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print(
            "warning: no GITHUB_TOKEN set; falling back to anonymous rate limits",
            file=sys.stderr,
        )

    series: Series = []
    failures = 0
    for repo in repos:
        try:
            records = star_records(repo, token)
            try:
                avatar = owner_avatar(repo, token)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError):
                avatar = None
            series.append((repo, records, avatar))
            stars = records[-1][1] if records else 0
            print(f"ok: {repo} ({stars} stars, {len(records)} points)")
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            KeyError,
            ValueError,
        ) as exc:
            failures += 1
            print(f"error: {repo}: {exc}", file=sys.stderr)

    if not series:
        print("no data fetched for any repository", file=sys.stderr)
        return 1

    here = Path(__file__).resolve().parent
    render(series, here / "star-history.svg", dark=False)
    render(series, here / "star-history-dark.svg", dark=True)
    print(f"wrote charts for {len(series)} repos ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
