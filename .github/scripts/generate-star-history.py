#!/usr/bin/env python3
"""
Generate a Star History Chart for multiple repositories.
This script fetches star data from GitHub API and creates an SVG chart.
"""

import os
import sys
import json
import requests
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter
import numpy as np

# Configuration
REPOS = [
    "bittner/pyclean",
    "painless-software/python-cli-test-helpers",
    "painless-software/django-probes",
    "behave/behave-django",
    "jazzband/django-analytical",
    "behave/behave",
]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
OUTPUT_DIR = Path("star-history")
OUTPUT_FILE = OUTPUT_DIR / "star-history.svg"

# API pagination settings
MAX_PAGES = 100  # Maximum number of pages to fetch (limits to 10,000 stars)
PER_PAGE = 100  # Number of stargazers per page

# Chart settings
CHART_DPI = 100  # Resolution for the output SVG

# Color palette for different repositories
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def get_star_history(repo):
    """
    Fetch star history for a repository using GitHub API.
    Returns a list of tuples (date, cumulative_stars).
    """
    owner, name = repo.split("/")
    url = f"https://api.github.com/repos/{owner}/{name}/stargazers"
    
    headers = {
        "Accept": "application/vnd.github.v3.star+json",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    
    stars = []
    page = 1
    
    print(f"Fetching star data for {repo}...")
    
    while True:
        params = {"page": page, "per_page": PER_PAGE}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching data for {repo}: {response.status_code}")
            print(f"Response: {response.text}")
            break
        
        data = response.json()
        if not data:
            break
        
        for item in data:
            starred_at = datetime.strptime(item["starred_at"], "%Y-%m-%dT%H:%M:%SZ")
            stars.append(starred_at)
        
        print(f"  Fetched page {page} ({len(data)} stars)")
        
        # Check if there are more pages
        if len(data) < PER_PAGE:
            break
        
        page += 1
        
        # Rate limiting protection - GitHub API has limits
        if page > MAX_PAGES:
            print(f"  Reached maximum page limit for {repo}")
            break
    
    # Convert to cumulative star counts
    stars.sort()
    cumulative = [(date, idx + 1) for idx, date in enumerate(stars)]
    
    print(f"  Total stars: {len(stars)}")
    return cumulative


def generate_chart(repo_data):
    """
    Generate an SVG chart from repository star data.
    """
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    # Plot each repository
    for idx, (repo, data) in enumerate(repo_data.items()):
        if not data:
            continue
        
        dates = [d[0] for d in data]
        counts = [d[1] for d in data]
        
        # Use logarithmic scale for better visualization of different sized projects
        ax.plot(dates, counts, label=repo, color=COLORS[idx % len(COLORS)], 
                linewidth=2, marker='o', markersize=0.5, alpha=0.8)
    
    # Formatting
    ax.set_yscale('log')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stars (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('GitHub Star History', fontsize=16, fontweight='bold', pad=20)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    plt.xticks(rotation=45, ha='right')
    
    # Format y-axis to show actual numbers, not scientific notation
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save as SVG
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, format='svg', bbox_inches='tight', dpi=CHART_DPI)
    print(f"\nChart saved to {OUTPUT_FILE}")


def main():
    """Main function to orchestrate the star history generation."""
    if not GITHUB_TOKEN:
        print("Warning: GITHUB_TOKEN not set. API rate limits will be restrictive.")
        print("Set GITHUB_TOKEN environment variable for better rate limits.")
    
    # Fetch data for all repositories
    repo_data = {}
    for repo in REPOS:
        data = get_star_history(repo)
        repo_data[repo] = data
    
    # Generate the chart
    if any(repo_data.values()):
        generate_chart(repo_data)
        print("\nStar history chart generated successfully!")
        return 0
    else:
        print("\nError: No data fetched for any repository.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
