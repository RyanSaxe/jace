"""Local Scryfall card image and metadata cache for Jace Studio.

Downloads normal-resolution card art and oracle text on first request, stores
them under `<cache_dir>/`, and serves a clean SVG fallback for custom tokens or
offline environments so the Tabletop never renders broken images.
"""

from __future__ import annotations

import html
import json
import re
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
_TOKEN_TEXT = {
    "Ally": "1/1 white Ally creature token.",
    "Soldier": "1/1 or 2/2 white Soldier creature token.",
    "Clue": "{2}, Sacrifice this artifact: Draw a card.",
    "Food": "{2}, {T}, Sacrifice this artifact: You gain 3 life.",
    "Treasure": "{T}, Sacrifice this artifact: Add one mana of any color.",
}


@dataclass(frozen=True, slots=True)
class CardMetadata:
    name: str
    set_code: str
    mana_cost: str
    type_line: str
    oracle_text: str
    is_token: bool = False


def render_token_svg(name: str, *, subtitle: str = "TLA Limited") -> bytes:
    safe_name = html.escape(name[:22])
    safe_sub = html.escape(subtitle[:24])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="308" viewBox="0 0 220 308">'
        '<rect width="220" height="308" rx="14" fill="#27272a"/>'
        '<rect x="10" y="10" width="200" height="288" rx="10" fill="#3f3f46" stroke="#71717a" stroke-width="2"/>'
        '<rect x="20" y="24" width="180" height="34" rx="6" fill="#18181b"/>'
        '<text x="110" y="46" text-anchor="middle" fill="#f4f4f5" font-family="sans-serif" '
        f'font-size="14" font-weight="700">{safe_name}</text>'
        '<rect x="20" y="70" width="180" height="140" rx="8" fill="#18181b" stroke="#52525b"/>'
        '<text x="110" y="144" text-anchor="middle" fill="#a1a1aa" font-family="sans-serif" '
        f'font-size="13">{safe_sub}</text>'
        '<text x="110" y="264" text-anchor="middle" fill="#d4d4d8" font-family="monospace" '
        'font-size="11">Jace Studio</text>'
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
        target.write_bytes(img_resp.content)


class ScryfallCache:
    def __init__(self, cache_dir: Path, *, allow_network: bool = True) -> None:
        self._cache_dir = cache_dir.resolve()
        self._allow_network = allow_network

    def _card_paths(self, name: str) -> tuple[Path, Path]:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "card"
        return self._cache_dir / f"{slug}.json", self._cache_dir / f"{slug}.jpg"

    def _fetch_and_cache(self, name: str) -> CardMetadata | None:
        if not self._allow_network or name in _TOKEN_TEXT:
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
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                meta_path, img_path = self._card_paths(name)
                meta_path.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
                img_url = _extract_image_url(payload)
                if img_url is not None:
                    _download_image(client, url=img_url, target=img_path)
                return meta
        except httpx.HTTPError:
            return None

    def get_metadata(self, name: str) -> CardMetadata:
        if name in _TOKEN_TEXT:
            return CardMetadata(
                name=name,
                set_code="TOKEN",
                mana_cost="",
                type_line="Token",
                oracle_text=_TOKEN_TEXT[name],
                is_token=True,
            )
        meta_path, _ = self._card_paths(name)
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            return CardMetadata(
                name=str(raw.get("name") or name),
                set_code=str(raw.get("set_code") or "TLA"),
                mana_cost=str(raw.get("mana_cost") or ""),
                type_line=str(raw.get("type_line") or ""),
                oracle_text=str(raw.get("oracle_text") or ""),
            )
        fetched = self._fetch_and_cache(name)
        if fetched is not None:
            return fetched
        return CardMetadata(name=name, set_code="TLA", mana_cost="", type_line="Card", oracle_text="")

    def get_image(self, name: str) -> tuple[bytes, str]:
        if name in _TOKEN_TEXT:
            return render_token_svg(name, subtitle="Token"), "image/svg+xml"
        _, img_path = self._card_paths(name)
        if img_path.is_file():
            return img_path.read_bytes(), "image/jpeg"
        self._fetch_and_cache(name)
        if img_path.is_file():
            return img_path.read_bytes(), "image/jpeg"
        return render_token_svg(name), "image/svg+xml"
