"""
    FastAPI Webhook Server
"""
import os
import hmac
import hashlib
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
import logging
from datetime import datetime

from fastapi.sse import EventSourceResponse
from langchain.agents import create_agent
import uvicorn
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from main_agent import process_pr_with_agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)

TRIGGER_ACTIONS = {"opened", "reopened", "synchronize", "ready_for_review"}

# App Setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🔧 Starting PR Agent...")
    visualise_graph()    
    yield
    logger.info("🛑 Shutting down PR Agent...")


app = FastAPI(title="Github PR Agent",
              description="A FastAPI server that listens for GitHub PR events and processes them with a Langgraph agent.",
              lifespan=lifespan)

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


def extract_pr_data(payload: dict):
    try:
        print(*payload)
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {}).get("full_name", "unknown/repo")
        pr_number = pr.get("number", "unknown")
        pr_url = pr.get("html_url", "unknown_url")
        logger.info(f"📥 Received PR event: action={action}, repo={repo}, pr_number={pr_number}, pr_url={pr_url}")

        if not all([action, pr_url, repo, pr_number]):
            return None  # missing critical info, skip processing
        return repo, pr_number, pr_url, action
    except Exception as e:
        logger.error(f"❌ Error extracting PR data: {e}")
        return None
        
# Routes

@app.get("/")
async def root():
    return {
        "service": "GitHub PR Agent",
        "powered_by": "Langgraph + FastAPI + Ollama",
        "message": "Welcome to the GitHub PR Agent! Send PR events to /webhook/github to trigger processing.",
        "endpoints": {
            "/health": "GET - Check if the agent is running",
            "/webhook/github": "POST - GitHub webhook endpoint for PR events",
            "/webhook/github/stream": "POST - GitHub webhook endpoint for PR events with streaming response (experimental)"
        
        }
    }

@app.get("/health")
async def health():
    return {"message": "Agent Running", "timestamp": datetime.now().isoformat()}

@app.post("/webhook/github")
async def github_webhook(request: Request, bg_tasks: BackgroundTasks):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    event_type = request.headers.get("X-GitHub-Event")

    if not verify_github_signature(raw_body, signature):
        logger.warning("⚠️  Invalid signature for incoming webhook")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    """ if event_type != "pull_request":
        logger.info(f"📩 Ignoring non-PR event: {event_type}")
        return JSONResponse(content={"message": f"Ignored event type: {event_type}"}, status_code=200)
    """
    try:
        payload = await request.json()        
    except Exception as e:
        logger.error(f"❌ Error parsing webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    pr_data = extract_pr_data(payload)

    if not pr_data:
        logger.warning("⚠️  Missing critical PR data, skipping processing")
        return JSONResponse(content={"message": "Missing critical PR data, skipping"}, status_code=200)
    
    repo_name, pr_number, pr_url, action = pr_data

    if action not in TRIGGER_ACTIONS:
        logger.info(f"📩 Ignoring PR event with action: {action}")
        return JSONResponse(content={"message": f"Ignored action: {action}"}, status_code=200)

    logger.info(f"🚀 Triggering PR processing for {repo_name}#{pr_number} (action: {action})")

    bg_tasks.add_task(process_pr_in_background, repo_name, pr_number, pr_url)


# SSE webhook endpoint (experimental)
@app.post("/webhook/github/stream")
async def github_webhook_stream(request: Request):
    raw_body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    event_type = request.headers.get("X-GitHub-Event")

    if not verify_github_signature(raw_body, signature):
        logger.warning("⚠️  Invalid signature for incoming webhook")
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    if event_type != "pull_request":
        logger.info(f"📩 Ignoring non-PR event: {event_type}")
        return JSONResponse(content={"message": f"Ignored event type: {event_type}"}, status_code=200)

    try:
        payload = await request.json()        
    except Exception as e:
        logger.error(f"❌ Error parsing webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    pr_data = extract_pr_data(payload)

    if not pr_data:
        logger.warning("⚠️  Missing critical PR data, skipping processing")
        return JSONResponse(content={"message": "Missing critical PR data, skipping"}, status_code=200)
    
    repo_name, pr_number, pr_url, action = pr_data

    if action not in TRIGGER_ACTIONS:
        logger.info(f"📩 Ignoring PR event with action: {action}")
        return JSONResponse(content={"message": f"Ignored action: {action}"}, status_code=200)

    logger.info(f"🚀 Triggering PR processing for {repo_name}#{pr_number} (action: {action})")
            

async def process_pr_in_background(repo_name: str, pr_number: int, pr_url: str) -> None:
    result = await process_pr_with_agent(repo_name, pr_number, pr_url)
    logger.info(f"✅ Finished processing PR {repo_name}#{pr_number}: {result}")
    if result.get("success"):
        logger.info(f"🎉 PR {repo_name}#{pr_number} processed successfully!")
    else:
        logger.error(f"❌ PR {repo_name}#{pr_number} processing failed: {result.get('error')}")



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