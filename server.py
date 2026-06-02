import asyncio
import json
import uuid
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()

CONFIG_FILE    = Path("config.json")
STATIONERY_DIR = Path("stationery")
STATIONERY_DIR.mkdir(exist_ok=True)


def _load() -> dict:
    return json.loads(CONFIG_FILE.read_text()) if CONFIG_FILE.exists() else {}

def _save(d: dict):
    CONFIG_FILE.write_text(json.dumps(d, indent=2))


# ── Settings ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/settings")
def get_settings():
    cfg = _load()
    return {
        "bfl_api_key":       cfg.get("bfl_api_key", ""),
        "active_stationery": cfg.get("active_stationery", ""),
    }


class SettingsIn(BaseModel):
    bfl_api_key:       str | None = None
    active_stationery: str | None = None

@app.post("/api/settings")
def update_settings(body: SettingsIn):
    cfg = _load()
    if body.bfl_api_key is not None:
        cfg["bfl_api_key"] = body.bfl_api_key.strip()
    if body.active_stationery is not None:
        cfg["active_stationery"] = body.active_stationery
    _save(cfg)
    return {"ok": True}


# ── Stationery list ─────────────────────────────────────────────────────────

@app.get("/api/stationery")
def list_stationery():
    images = sorted(STATIONERY_DIR.glob("*.jpeg"), key=lambda p: p.stat().st_mtime, reverse=True)
    cfg = _load()
    active = cfg.get("active_stationery", "")
    return [
        {"filename": p.name, "url": f"/stationery/{p.name}", "active": f"/stationery/{p.name}" == active}
        for p in images
    ]


# ── Stationery prompts ───────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "classic": (
        "Elegant personal letter stationery, portrait 8.5x11 ratio, cream ivory paper texture, "
        "ornate Victorian calligraphic border flourishes at the very top edge of the page in navy blue ink, "
        "subtle ruled horizontal writing lines across the lower two-thirds, faint left margin line in soft red, "
        "sophisticated muted palette of ivory and navy, no text, no words, flat lay top-down"
    ),
    "botanical": (
        "Elegant personal stationery paper, portrait 8.5x11 ratio, "
        "soft watercolor botanical illustration of delicate flowers and eucalyptus leaves, "
        "muted dusty rose and sage green tones clustered in top corners only, "
        "clean cream center with faint pale blue ruled writing lines, left margin line, "
        "no text, no words, premium paper, flat lay"
    ),
    "artdeco": (
        "Art Deco personal stationery letterhead, portrait 8.5x11 ratio, "
        "geometric gold and deep navy ornamental border frame at the very top, symmetrical architectural motifs, "
        "1920s luxury aesthetic, cream paper with subtle texture, "
        "faint ruled writing lines in lower portion, no text, no words, flat lay"
    ),
    "minimal": (
        "Ultra-minimalist personal stationery, portrait 8.5x11 ratio, "
        "warm white natural linen paper texture, single thin elegant border line at top, "
        "very faint ruled lines for writing, tiny delicate botanical ink sprig in upper-right corner only, "
        "Scandinavian clean aesthetic, no text, no words, flat lay"
    ),
    "vintage": (
        "Vintage aged personal letter stationery, portrait 8.5x11 ratio, "
        "warm sepia-toned antique paper with subtle yellowing at edges, "
        "ornate Victorian scrollwork decorative border at top, faint ruled writing lines, "
        "old-world charm, no text, no words, flat lay top-down"
    ),
}


# ── Generate ─────────────────────────────────────────────────────────────────

class GenerateIn(BaseModel):
    style: str = "classic"

@app.post("/api/generate-stationery")
async def generate_stationery(body: GenerateIn):
    cfg = _load()
    api_key = cfg.get("bfl_api_key", "")
    if not api_key:
        raise HTTPException(400, "BFL API key not configured — add it in ⚙ Settings")

    prompt = PROMPTS.get(body.style, PROMPTS["classic"])

    async with httpx.AsyncClient(timeout=180) as client:
        # 1 — submit
        resp = await client.post(
            "https://api.bfl.ai/v1/flux-pro-1.1",
            headers={"x-key": api_key, "Content-Type": "application/json"},
            json={
                "prompt":             prompt,
                "width":              832,   # closest multiples of 32 to 8.5×11
                "height":             1088,
                "output_format":      "jpeg",
                "prompt_upsampling":  True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        poll_url = data.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={data['id']}"

        # 2 — poll until Ready (up to 120 s)
        img_url = None
        for _ in range(60):
            await asyncio.sleep(2)
            r = await client.get(poll_url, headers={"x-key": api_key})
            r.raise_for_status()
            rd = r.json()
            status = rd.get("status", "")
            if status == "Ready":
                img_url = rd["result"]["sample"]
                break
            if status in ("Error", "Failed", "Content Moderated"):
                raise HTTPException(500, f"BFL generation failed: {status}")

        if not img_url:
            raise HTTPException(504, "BFL generation timed out")

        # 3 — download
        img_r = await client.get(img_url)
        img_r.raise_for_status()

    filename = f"{body.style}_{uuid.uuid4().hex[:8]}.jpeg"
    (STATIONERY_DIR / filename).write_bytes(img_r.content)

    cfg["active_stationery"] = f"/stationery/{filename}"
    _save(cfg)

    return {"url": f"/stationery/{filename}"}


# ── Static files ─────────────────────────────────────────────────────────────

app.mount("/stationery", StaticFiles(directory="stationery"), name="stationery")

@app.get("/")
def index():
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8090, reload=True)
