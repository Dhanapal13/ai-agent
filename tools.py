# tools available to the agent
import os
import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool("fetch_pr_details", return_direct=True)
def fetch_pr_details(repo: str, pr_number: int) -> dict:    
    """
    Fetch details of a GitHub PR using the GitHub API.
    This includes the PR title, description, changed files, and any relevant metadata.
    """
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set in environment variables.")
    
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PR details: {response.status_code} - {response.text}")
    
    pr_data = response.json()

    # Fetch changed files
    files_url = pr_data.get("url") + "/files"
    files_response = requests.get(files_url, headers=headers)
    files = files_response.json() if files_response.status_code == 200 else []
 
    file_summaries = [
        {
            "filename": f.get("filename"),
            "status": f.get("status"),           # added/modified/removed
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "changes": f.get("changes", 0),
        }
        for f in files[:20]  # limit to 20 files
    ]
 
    # Fetch review comments count
    reviews_url = url + "/reviews"
    reviews_response = requests.get(reviews_url, headers=headers)
    review_count = len(reviews_response.json()) if reviews_response.status_code == 200 else 0
 
    summary = {
        "title": pr_data.get("title"),
        "number": pr_data.get("number"),
        "state": pr_data.get("state"),
        "author": pr_data.get("user", {}).get("login"),
        "author_url": pr_data.get("user", {}).get("html_url"),
        "pr_url": pr_data.get("html_url"),
        "base_branch": pr_data.get("base", {}).get("ref"),      # target branch
        "head_branch": pr_data.get("head", {}).get("ref"),      # source branch
        "body": pr_data.get("body") or "No description provided.",
        "created_at": pr_data.get("created_at"),
        "commits": pr_data.get("commits", 0),
        "additions": pr_data.get("additions", 0),
        "deletions": pr_data.get("deletions", 0),
        "changed_files": pr_data.get("changed_files", 0),
        "files": file_summaries,
        "review_count": review_count,
        "draft": pr_data.get("draft", False),
        "mergeable": pr_data.get("mergeable"),
        "labels": [l["name"] for l in pr_data.get("labels", [])],
    }
 
    return json.dumps(summary, indent=2)

@tool("send_email", return_direct=True)
def send_email(to_email: str, subject: str, body: str):
    """
    Send an email using SMTP.
    This function can be used by the agent to communicate findings or requests to PR authors or stakeholders.
    """
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    
    if not all([smtp_server, smtp_user, smtp_password]):
        raise ValueError("SMTP configuration is incomplete in environment variables.")
    
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {to_email} with subject '{subject}'")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


def assess_pr_risk(pr_details: str) -> str:
    """
    Assess the security risk of a PR based on its details.
    This is a placeholder function that can be expanded with actual risk assessment logic.
    For now, it simply returns a dummy risk level based on the presence of certain keywords.
    """
    risk_level = "Low"
    if any(keyword in pr_details.lower() for keyword in ["password", "secret", "credential"]):
        risk_level = "High"
    elif any(keyword in pr_details.lower() for keyword in ["config", "env", "key"]):
        risk_level = "Medium"
    
    return f"Risk Assessment: {risk_level}"

TOOLS = [fetch_pr_details, send_email, assess_pr_risk]
