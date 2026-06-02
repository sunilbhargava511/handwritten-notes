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


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Settings ─────────────────────────────────────────────────────────────────

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


# ── Stationery list ───────────────────────────────────────────────────────────

@app.get("/api/stationery")
def list_stationery():
    images = sorted(STATIONERY_DIR.glob("*.jpeg"), key=lambda p: p.stat().st_mtime, reverse=True)
    cfg = _load()
    active = cfg.get("active_stationery", "")
    return [
        {"filename": p.name, "url": f"/stationery/{p.name}", "active": f"/stationery/{p.name}" == active}
        for p in images
    ]


# ── Stationery prompts ────────────────────────────────────────────────────────

# Suffix appended to every prompt to ensure stationery is usable for writing.
# NOTE: No ruled lines in the image — lines are handled by the CSS overlay.
_SUFFIX = (
    " Decorative elements ONLY at the very top header and narrow side margins. "
    "The lower three-quarters must be a clear, smooth, PLAIN writing area with NO ruled lines, "
    "NO horizontal lines, NO line markings of any kind — just clean paper texture. "
    "Portrait 8.5x11 paper ratio. No text, no words, no handwriting. Flat lay top-down."
)

PROMPTS: dict[str, str] = {
    "classic": (
        "Elegant personal letter stationery. Cream ivory paper texture. "
        "Ornate Victorian calligraphic border flourishes confined to the top 20% header band "
        "in navy blue ink, with a thin double rule line separating header from writing area."
        + _SUFFIX
    ),
    "botanical": (
        "Personal stationery with soft watercolor botanical illustration. "
        "Delicate flowers and eucalyptus leaves in dusty rose and sage green, "
        "clustered ONLY in the top header band and optionally small accents in the bottom corners. "
        "Clean cream writing area, no lines."
        + _SUFFIX
    ),
    "artdeco": (
        "Art Deco personal stationery letterhead. "
        "Geometric gold and deep navy ornamental motifs confined to the top 25% header band, "
        "symmetrical fan and diamond patterns, 1920s luxury aesthetic. "
        "Cream paper writing area, no ruled lines."
        + _SUFFIX
    ),
    "minimal": (
        "Ultra-minimalist personal stationery. "
        "Warm white natural linen paper. Single thin elegant border line at the very top. "
        "Tiny delicate botanical ink sprig ONLY in the upper-right corner of the header. "
        "Vast clean writing area below, no lines. Scandinavian aesthetic."
        + _SUFFIX
    ),
    "vintage": (
        "Vintage aged personal letter stationery. "
        "Warm sepia-toned antique paper with subtle yellowing. "
        "Ornate Victorian scrollwork border band at the very top only. "
        "Large open writing area below, no lines."
        + _SUFFIX
    ),
    "japanese": (
        "Japanese minimalist personal stationery. "
        "Delicate cherry blossom branch watercolor painting ONLY along the top header band, "
        "soft pink and white tones on warm ivory washi-like paper. "
        "Vast serene writing area below, no lines."
        + _SUFFIX
    ),
    "floral_border": (
        "Elegant personal stationery with a floral border frame. "
        "Detailed watercolor roses and greenery forming a decorative band ONLY at the top. "
        "Cream paper writing area, no lines."
        + _SUFFIX
    ),
}


# ── Generate ──────────────────────────────────────────────────────────────────

class GenerateIn(BaseModel):
    style: str = "classic"
    custom_prompt: str = ""    # free-text description; overrides preset if non-empty
    reference_b64: str = ""    # base64 inspiration image (with or without data URI prefix)

@app.post("/api/generate-stationery")
async def generate_stationery(body: GenerateIn):
    cfg = _load()
    api_key = cfg.get("bfl_api_key", "")
    if not api_key:
        raise HTTPException(400, "BFL API key not configured — add it in ⚙ Settings")

    # Build prompt
    if body.custom_prompt.strip():
        prompt = body.custom_prompt.strip() + _SUFFIX
    else:
        prompt = PROMPTS.get(body.style, PROMPTS["classic"])

    # Use flux-2-flex when a reference image is provided (supports image conditioning)
    has_ref = bool(body.reference_b64.strip())
    model   = "flux-2-flex" if has_ref else "flux-pro-1.1"

    req_body: dict = {
        "prompt":            prompt,
        "width":             832,
        "height":            1088,
        "output_format":     "jpeg",
        "prompt_upsampling": True,
    }

    if has_ref:
        ref = body.reference_b64
        if "," in ref:          # strip "data:image/...;base64," prefix
            ref = ref.split(",", 1)[1]
        req_body["input_image"] = ref

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"https://api.bfl.ai/v1/{model}",
            headers={"x-key": api_key, "Content-Type": "application/json"},
            json=req_body,
        )
        resp.raise_for_status()
        data     = resp.json()
        poll_url = data.get("polling_url") or f"https://api.bfl.ai/v1/get_result?id={data['id']}"

        img_url = None
        for _ in range(60):
            await asyncio.sleep(2)
            r = await client.get(poll_url, headers={"x-key": api_key})
            r.raise_for_status()
            rd     = r.json()
            status = rd.get("status", "")
            if status == "Ready":
                img_url = rd["result"]["sample"]
                break
            if status in ("Error", "Failed", "Content Moderated"):
                raise HTTPException(500, f"BFL generation failed: {status}")

        if not img_url:
            raise HTTPException(504, "BFL generation timed out")

        img_r = await client.get(img_url)
        img_r.raise_for_status()

    slug     = "custom" if body.custom_prompt.strip() else body.style
    filename = f"{slug}_{uuid.uuid4().hex[:8]}.jpeg"
    (STATIONERY_DIR / filename).write_bytes(img_r.content)

    cfg["active_stationery"] = f"/stationery/{filename}"
    _save(cfg)

    return {"url": f"/stationery/{filename}"}


# ── Static files ──────────────────────────────────────────────────────────────

app.mount("/stationery", StaticFiles(directory="stationery"), name="stationery")

@app.get("/")
def index():
    return FileResponse("index.html")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8090, reload=True)
