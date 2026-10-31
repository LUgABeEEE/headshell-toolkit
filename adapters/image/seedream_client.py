"""Seedream (Volcengine Ark) image client with injectable transport (MIT)."""
from __future__ import annotations
import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class SeedreamConfig:
    api_key: str
    base_url: str = "https://ark.cn-beijing.volces.com/api/plan/v3"
    model: str = "doubao-seedream-5.0-pro"
    timeout: int = 120
    retries: int = 2


class SeedreamError(RuntimeError):
    pass


def _urllib_transport(url: str, headers: dict, body: bytes, timeout: int) -> tuple[int, bytes]:
    """Real-network POST returning (status, body); HTTPError is converted to a status pair."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _urllib_download(url: str, timeout: int) -> bytes:
    """Real-network GET returning the raw response body."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


class SeedreamClient:
    """POST {base_url}/images/generations, then download the returned URL.

    transport(url, headers, body: bytes, timeout) -> (status: int, body: bytes)
    download(url, timeout) -> bytes
    """

    def __init__(self, config: SeedreamConfig,
                 transport: Callable[[str, dict, bytes, int], tuple[int, bytes]] | None = None,
                 download: Callable[[str, int], bytes] | None = None):
        self.config = config
        self._transport = transport
        self._download = download

    @staticmethod
    def _data_uri(path: Path) -> str:
        """Encode a local image as a base64 data URI, mime guessed from the suffix."""
        suffix = path.suffix.lower().lstrip(".") or "png"
        return f"data:image/{suffix};base64," + base64.b64encode(path.read_bytes()).decode("ascii")

    def generate(self, prompt: str, out_path: str | Path, size: str = "1024x1024",
                 refs: list[str | Path] | None = None) -> Path:
        """Generate one PNG and write it to out_path. Retries transient failures;
        on a rejected size, retries once with 1024x1024. Raises SeedreamError otherwise."""
        cfg = self.config
        transport = self._transport if self._transport is not None else _urllib_transport
        download = self._download if self._download is not None else _urllib_download

        body: dict = {
            "model": cfg.model,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "output_format": "png",
            "watermark": False,
        }
        if refs:
            try:
                images = [self._data_uri(Path(p)) for p in refs]
            except OSError as exc:
                raise SeedreamError(f"failed to read reference image: {exc}") from exc
            body["image"] = images if len(images) > 1 else images[0]

        url = f"{cfg.base_url.rstrip('/')}/images/generations"
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        payload = json.dumps(body).encode("utf-8")
        current_size = size
        size_fallback_used = False
        attempts = 0

        while True:
            attempts += 1
            try:
                status, resp = transport(url, headers, payload, cfg.timeout)
            except Exception as exc:
                if attempts <= cfg.retries:
                    continue
                raise SeedreamError(f"Seedream transport failed after {attempts} attempt(s): {exc}") from exc

            if status >= 500:
                if attempts <= cfg.retries:
                    continue
                detail = resp.decode("utf-8", "ignore")[:500]
                raise SeedreamError(f"Seedream API {status}: {detail}")

            if status >= 400:
                detail = resp.decode("utf-8", "ignore")[:500]
                if status == 400 and current_size != "1024x1024" and not size_fallback_used:
                    size_fallback_used = True
                    current_size = "1024x1024"
                    body["size"] = current_size
                    payload = json.dumps(body).encode("utf-8")
                    continue
                raise SeedreamError(f"Seedream API {status}: {detail}")

            # Success (2xx/3xx): parse the image URL and download it.
            try:
                data = json.loads(resp.decode("utf-8"))
                image_url = data["data"][0]["url"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise SeedreamError(f"unexpected Seedream response: {exc}") from exc

            try:
                image_bytes = download(image_url, cfg.timeout)
            except Exception as exc:
                raise SeedreamError(f"failed to download image: {exc}") from exc

            out = Path(out_path)
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(image_bytes)
            except OSError as exc:
                raise SeedreamError(f"failed to write image to {out}: {exc}") from exc
            return out
