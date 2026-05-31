"""
    FastAPI Webhook Server
"""
import os
import hmac
import hashlib
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from langchain.agents import create_agent
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from agent import process_pr_with_agent, create_pr_agent

load_dotenv()

# App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Github PR Agent", lifespan=lifespan)

processing_prs = set()  # to track processed PRs and avoid duplicates

# ── Signature Verification ─────────────────────────────────────────
def verify_github_signature(payload_body: bytes, signature_header: Optional[str]) -> bool:
    """
    Verify GitHub's HMAC-SHA256 webhook signature.
    This ensures requests genuinely come from GitHub, not an attacker.
    """
    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if not secret:
        print("⚠️  No GITHUB_WEBHOOK_SECRET set — skipping signature check (not recommended for production)")
        return True  # allow in dev without secret
 
    if not signature_header or not signature_header.startswith("sha256="):
        return False
 
    expected = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload_body,
        digestmod=hashlib.sha256
    ).hexdigest()
 
    received = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, received)


# Routes
@app.get("/health")
async def health():
    return "Agent Running"

@app.post("/webhook/github")
async def github_webhook(request: Request, bg_tasks: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_github_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    
    event_type= request.headers.get("X-Github-Event", "")
    delivery_id = request.headers.get("X-Github-Delivery", "unknown")

    if event_type != "pull_request":
        return JSONResponse(status_code=200, content={"message": "Event not handled"})

    # parse payload
    payload = await request.json()
    payload_action = payload.get("action")
    if payload_action not in ["opened", "ready_for_review", "reopened"]:
        return JSONResponse(status_code=200, content=
                            {"message": f"PR action '{payload_action}' ignored, only 'opened', 'ready_for_review', 'reopened' are processed    "})

    # extract PR details
    pr = payload.get("pull_request", {})
    pr_number = pr.get("number")
    pr_url = pr.get("html_url")
    repo_name = payload.get("repository", {}).get("full_name")
    pr_title = pr.get("title")
    author = pr.get("user", {}).get("login")

    #  avoid duplicate processing
    pr_key = f"{repo_name}#{pr_number}"
    if pr_key in processing_prs:
        print(f"Skipping duplicate PR event: {event_type} for PR #{pr_key} in {repo_name} by {author}")
        return JSONResponse(status_code=200, content={"message": "Duplicate PR event ignored"})
    print(f"Received PR event: {event_type} for PR #{pr_number} in {repo_name} by {author}")


async def process_pr_in_background(pr_url, repo_name, pr_number, pr_key):
    processing_prs.add(pr_key)
    try:
        # process PR with async agent
        loop = asyncio.get_event_loop()
        agent = create_pr_agent()
        result = await loop.run_in_executor(None, lambda: process_pr_with_agent(agent, pr_url, repo_name, pr_number))   
        if result["status"] == "success":
            print(f"✅ Successfully processed PR #{pr_key} in {repo_name}")
        else:
            print(f"❌ Failed to process PR #{pr_key} in {repo_name}: {result.get('error', 'Unknown error')}")
    finally:
        processing_prs.discard(pr_key)


if __name__ == "__main__":
    import ngrok
    import uvicorn
    import os
    import dotenv

    load_dotenv()
    print("🚀 Starting ngrok tunnel...")

    # v1.x API - use ngrok.connect() directly (no 'addr' or 'proto' kwargs)
    listener = ngrok.connect(8000, authtoken_from_env=True)

    print(f"\n✅ Ngrok Tunnel Active: {listener.url()}")
    print(f"📌 GitHub Webhook URL → {listener.url()}/webhook\n")

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, log_level="info")