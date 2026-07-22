"""Tools used by the PR multi-agent system.

Each tool is intentionally "dumb" — it does one deterministic job (talk to
GitHub, run a keyword risk heuristic, send an email) and returns a plain
dict. None of them raise on expected failure modes; they return an
"error"/"success" key instead, so a single flaky API call can't crash the
whole graph. The agents (in agents.py) decide what to do with the results.
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_pr_details(repo: str, pr_number: int) -> dict:
    """Fetch PR title, description, branches, diff stats and changed files from GitHub.

    Returns a dict with either the PR summary or an "error" key on failure.
    Never raises — callers should check for the "error" key.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        return {"error": "GITHUB_TOKEN is not set in environment variables."}

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    try:
        response = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        return {"error": f"Network error while fetching PR: {exc}"}

    if response.status_code != 200:
        return {"error": f"Failed to fetch PR details: {response.status_code} - {response.text}"}

    pr_data = response.json()

    files_url = pr_data.get("url", url) + "/files"
    files_response = requests.get(files_url, headers=headers, timeout=15)
    files = files_response.json() if files_response.status_code == 200 else []

    file_summaries = [
        {
            "filename": f.get("filename"),
            "status": f.get("status"),  # added / modified / removed
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
        }
        for f in files[:20]  # limit to 20 files
    ]

    reviews_url = url + "/reviews"
    reviews_response = requests.get(reviews_url, headers=headers, timeout=15)
    review_count = len(reviews_response.json()) if reviews_response.status_code == 200 else 0

    return {
        "title": pr_data.get("title"),
        "number": pr_data.get("number"),
        "author": pr_data.get("user", {}).get("login"),
        "state": pr_data.get("state"),
        "pr_url": pr_data.get("html_url"),
        "base_branch": pr_data.get("base", {}).get("ref"),  # target branch
        "head_branch": pr_data.get("head", {}).get("ref"),  # source branch
        "body": pr_data.get("body") or "No description provided.",
        "created_at": pr_data.get("created_at"),
        "commits": pr_data.get("commits", 0),
        "additions": pr_data.get("additions", 0),
        "deletions": pr_data.get("deletions", 0),
        "changed_files": pr_data.get("changed_files", 0),
        "review_count": review_count,
        "files": file_summaries,
    }


RISK_KEYWORDS_HIGH = ["password", "secret", "credential", "token", "private_key", "api_key"]
RISK_KEYWORDS_MEDIUM = ["config", ".env", "env.", "key", "auth", "permission", "dockerfile", "ci.yml"]


def run_risk_heuristic(pr_details: dict) -> dict:
    """Deterministic keyword-based first-pass signal over PR title/body/filenames.

    This does not replace the risk agent's LLM reasoning — it just gives it a
    cheap, explainable starting signal to reason over.
    """
    haystack_parts = [
        pr_details.get("title", "") or "",
        pr_details.get("body", "") or "",
    ] + [f.get("filename", "") or "" for f in pr_details.get("files", [])]
    haystack = " ".join(haystack_parts).lower()

    matched_high = [kw for kw in RISK_KEYWORDS_HIGH if kw in haystack]
    matched_medium = [kw for kw in RISK_KEYWORDS_MEDIUM if kw in haystack]

    if matched_high:
        level = "HIGH"
    elif matched_medium:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "heuristic_level": level,
        "matched_high_risk_keywords": matched_high,
        "matched_medium_risk_keywords": matched_medium,
        "changed_files": pr_details.get("changed_files", 0),
        "additions": pr_details.get("additions", 0),
        "deletions": pr_details.get("deletions", 0),
    }


def send_email(to_email: str, subject: str, body: str) -> dict:
    """Send a plain-text email via SMTP. Returns {"success": bool, "error"?: str}."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        return {"success": False, "error": "SMTP configuration is incomplete in environment variables."}

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return {"success": True}
    except Exception as exc:  # noqa: BLE001 - we want to report any SMTP failure, not crash
        return {"success": False, "error": str(exc)}