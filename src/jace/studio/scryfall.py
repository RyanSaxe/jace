"""Local Scryfall card image and metadata cache for Jace Studio.

Downloads normal-resolution card art and oracle text on first request, stores
them atomically under `<cache_dir>/` with per-card locking to avoid request
stampedes, and serves legible high-contrast SVGs for tokens and offline mode.
"""

from __future__ import annotations

import html
import json
import re
import threading
import urllib.parse
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from pathlib import Path

__all__ = (
    "CardMetadata",
    "ScryfallCache",
    "render_token_svg",
)

_HTTP_OK = 200
_TOKEN_STYLES: dict[str, tuple[str, str, str]] = {
    "Ally": ("#fef3c7", "#92400e", "1/1 white Ally creature token."),
    "Soldier": ("#fef3c7", "#92400e", "1/1 or 2/2 white Soldier creature token."),
    "Clue": ("#e0f2fe", "#075985", "{2}, Sacrifice this artifact: Draw a card."),
    "Food": ("#dcfce7", "#166534", "{2}, {T}, Sacrifice this artifact: You gain 3 life."),
    "Treasure": ("#fef9c3", "#854d0e", "{T}, Sacrifice this artifact: Add one mana of any color."),
    "Zombie": ("#f3e8ff", "#581c87", "2/2 black Zombie creature token."),
}


@dataclass(frozen=True, slots=True)
class CardMetadata:
    name: str
    set_code: str
    mana_cost: str
    type_line: str
    oracle_text: str
    is_token: bool = False


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def render_token_svg(name: str, *, subtitle: str = "TLA Limited") -> bytes:
    bg, accent, _ = _TOKEN_STYLES.get(name, ("#e4e4e7", "#3f3f46", ""))
    safe_name = html.escape(name[:18])
    safe_sub = html.escape(subtitle[:20])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="308" viewBox="0 0 220 308">'
        f'<rect width="220" height="308" rx="16" fill="{accent}"/>'
        f'<rect x="10" y="10" width="200" height="288" rx="12" fill="{bg}" stroke="{accent}" stroke-width="4"/>'
        f'<rect x="18" y="24" width="184" height="54" rx="8" fill="{accent}"/>'
        '<text x="110" y="60" text-anchor="middle" fill="#ffffff" font-family="sans-serif" '
        f'font-size="26" font-weight="800">{safe_name}</text>'
        f'<rect x="18" y="96" width="184" height="132" rx="10" fill="#ffffff" stroke="{accent}" stroke-width="2"/>'
        f'<text x="110" y="172" text-anchor="middle" fill="{accent}" font-family="sans-serif" '
        f'font-size="22" font-weight="700">{safe_sub}</text>'
        f'<text x="110" y="272" text-anchor="middle" fill="{accent}" font-family="monospace" '
        'font-size="14" font-weight="700">TOKEN</text>'
        "</svg>"
    )
    return svg.encode("utf-8")


def _extract_image_url(payload: dict[str, object]) -> str | None:
    uris = payload.get("image_uris")
    if isinstance(uris, dict) and isinstance(uris.get("normal"), str):
        return str(uris["normal"])
    faces = payload.get("card_faces")
    if isinstance(faces, list) and faces and isinstance(faces[0], dict):
        face_uris = faces[0].get("image_uris")
        if isinstance(face_uris, dict) and isinstance(face_uris.get("normal"), str):
            return str(face_uris["normal"])
    return None


def _extract_oracle_text(payload: dict[str, object]) -> str:
    text = payload.get("oracle_text")
    if isinstance(text, str):
        return text
    faces = payload.get("card_faces")
    if isinstance(faces, list):
        parts = [str(face.get("oracle_text")) for face in faces if isinstance(face, dict) and face.get("oracle_text")]
        return "\n---\n".join(parts)
    return ""


def _download_image(client: httpx.Client, *, url: str, target: Path) -> None:
    img_resp = client.get(url, headers={"User-Agent": "JaceStudio/0.1"})
    if img_resp.status_code == _HTTP_OK:
        _atomic_write_bytes(target, img_resp.content)


def _read_cached_meta(meta_path: Path, name: str) -> CardMetadata | None:
    if not meta_path.is_file():
        return None
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(raw, dict):
        return None
    return CardMetadata(
        name=str(raw.get("name") or name),
        set_code=str(raw.get("set_code") or "TLA"),
        mana_cost=str(raw.get("mana_cost") or ""),
        type_line=str(raw.get("type_line") or ""),
        oracle_text=str(raw.get("oracle_text") or ""),
    )


class ScryfallCache:
    def __init__(self, cache_dir: Path, *, allow_network: bool = True) -> None:
        self._cache_dir = cache_dir.resolve()
        self._allow_network = allow_network
        self._locks_guard = threading.Lock()
        self._card_locks: dict[str, threading.Lock] = {}

    def _card_paths(self, name: str) -> tuple[Path, Path]:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "card"
        return self._cache_dir / f"{slug}.json", self._cache_dir / f"{slug}.jpg"

    def _lock_for(self, name: str) -> threading.Lock:
        key = name.lower()
        with self._locks_guard:
            lock = self._card_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._card_locks[key] = lock
            return lock

    def _fetch_and_cache(self, name: str) -> CardMetadata | None:
        if not self._allow_network or name in _TOKEN_STYLES:
            return None
        url = "https://api.scryfall.com/cards/named?exact=" + urllib.parse.quote(name)
        headers = {"User-Agent": "JaceStudio/0.1", "Accept": "application/json"}
        try:
            with httpx.Client(timeout=6.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code != _HTTP_OK:
                    return None
                payload = resp.json()
                if not isinstance(payload, dict):
                    return None
                meta = CardMetadata(
                    name=str(payload.get("name") or name),
                    set_code=str(payload.get("set") or "TLA").upper(),
                    mana_cost=str(payload.get("mana_cost") or ""),
                    type_line=str(payload.get("type_line") or ""),
                    oracle_text=_extract_oracle_text(payload),
                )
                meta_path, img_path = self._card_paths(name)
                _atomic_write_bytes(meta_path, (json.dumps(asdict(meta), indent=2) + "\n").encode("utf-8"))
                img_url = _extract_image_url(payload)
                if img_url is not None:
                    _download_image(client, url=img_url, target=img_path)
                return meta
        except httpx.HTTPError:
            return None

    def get_metadata(self, name: str) -> CardMetadata:
        if name in _TOKEN_STYLES:
            return CardMetadata(
                name=name,
                set_code="TOKEN",
                mana_cost="",
                type_line="Token",
                oracle_text=_TOKEN_STYLES[name][2],
                is_token=True,
            )
        meta_path, _ = self._card_paths(name)
        cached = _read_cached_meta(meta_path, name)
        if cached is not None:
            return cached
        with self._lock_for(name):
            cached = _read_cached_meta(meta_path, name)
            if cached is not None:
                return cached
            fetched = self._fetch_and_cache(name)
        if fetched is not None:
            return fetched
        return CardMetadata(name=name, set_code="TLA", mana_cost="", type_line="Card", oracle_text="")

    def get_image(self, name: str) -> tuple[bytes, str]:
        if name in _TOKEN_STYLES:
            return render_token_svg(name, subtitle="Token"), "image/svg+xml"
        _, img_path = self._card_paths(name)
        if img_path.is_file():
            return img_path.read_bytes(), "image/jpeg"
        with self._lock_for(name):
            if not img_path.is_file():
                self._fetch_and_cache(name)
        if img_path.is_file():
            return img_path.read_bytes(), "image/jpeg"
        return render_token_svg(name, subtitle="Card"), "image/svg+xml"
