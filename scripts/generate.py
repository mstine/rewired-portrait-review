#!/usr/bin/env python3
"""
Gemini Nano Banana image generator for the Feral Architecture logo v1 pass.

Adapted from ~/RitualSync/psyche-infographic/scripts/generate-images.py.
Reads feral-logo-prompts.json and writes JPEG outputs to ../images/<id>-v<n>.jpg.
Idempotent — skips files that already exist. Requires GEMINI_API_KEY.

    python3 scripts/generate-logos.py
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_reference_image(path: Path) -> dict:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": b64}}

MODEL = "gemini-3.1-flash-image-preview"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILE = REPO_ROOT / "scripts" / "prompts.json"
OUTPUT_DIR = REPO_ROOT / "images"

RATE_LIMIT_DELAY_SECONDS = 2


def generate_one(parts: list, output_path: Path, api_key: str) -> tuple[int, str]:
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_body[:500]}") from e

    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"API error {err.get('code')} {err.get('status')}: {err.get('message','')[:300]}")

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected response shape: {json.dumps(data)[:500]}") from e

    for p in parts:
        if "inlineData" in p:
            img_bytes = base64.b64decode(p["inlineData"]["data"])
            output_path.write_bytes(img_bytes)
            return len(img_bytes), p["inlineData"].get("mimeType", "unknown")

    raise RuntimeError(f"no inlineData in response parts: {json.dumps(parts)[:500]}")


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set in environment", file=sys.stderr)
        return 1

    if not PROMPTS_FILE.exists():
        print(f"ERROR: prompts file not found at {PROMPTS_FILE}", file=sys.stderr)
        return 1

    with PROMPTS_FILE.open() as f:
        prompts = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = sum(len(item["variants"]) for item in prompts)
    done = 0
    skipped = 0
    failed = 0

    print(f"Generating {total} images to {OUTPUT_DIR}")
    print(f"Model: {MODEL}")
    print()

    for item in prompts:
        image_id = item["id"]
        title = item.get("title", image_id)
        print(f"== {image_id}: {title} ==")

        ref_parts = []
        ref_rels = item.get("reference_images") or ([item["reference_image"]] if item.get("reference_image") else [])
        missing = False
        for ref_rel in ref_rels:
            ref_path = REPO_ROOT / ref_rel
            if not ref_path.exists():
                print(f"  WARNING: reference not found at {ref_rel}, skipping item")
                failed += len(item["variants"])
                missing = True
                break
            ref_parts.append(load_reference_image(ref_path))
            print(f"  using reference: {ref_rel}")
        if missing:
            continue

        for idx, prompt in enumerate(item["variants"], start=1):
            output_path = OUTPUT_DIR / f"{image_id}-v{idx}.jpg"

            if output_path.exists():
                print(f"  v{idx}: SKIP (exists at {output_path.name})")
                skipped += 1
                continue

            parts = list(ref_parts)
            parts.append({"text": prompt})

            print(f"  v{idx}: generating... ", end="", flush=True)
            try:
                size, mime = generate_one(parts, output_path, api_key)
                print(f"OK ({size:,} bytes, {mime})")
                done += 1
                time.sleep(RATE_LIMIT_DELAY_SECONDS)
            except Exception as e:
                print(f"FAIL: {e}")
                failed += 1

    print()
    print(f"Done: {done} generated, {skipped} skipped, {failed} failed, {total} total")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
