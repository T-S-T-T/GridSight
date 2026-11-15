"""HTTP helpers: a retrying session and a parser for NEMweb directory listings."""

from __future__ import annotations

import logging
import re
import time
from html import unescape
from urllib.parse import unquote, urljoin

import requests

from gridsight.config import get_settings

log = logging.getLogger(__name__)
_HREF = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = get_settings().nemweb.get("user_agent", "GridSight")
    return s


def get(url: str, sess: requests.Session | None = None, **kwargs) -> requests.Response:
    """GET with exponential backoff on connection errors and 5xx/429 responses."""
    cfg = get_settings().nemweb
    retries = int(cfg.get("retries", 3))
    timeout = float(cfg.get("timeout_s", 60))
    sess = sess or session()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = sess.get(url, timeout=timeout, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{resp.status_code} for {url}", response=resp)
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last = exc
            if attempt < retries:
                wait = 2**attempt
                log.warning("GET %s failed (%s); retrying in %ss", url, exc, wait)
                time.sleep(wait)
    assert last is not None
    raise last


def list_links(url: str, sess: requests.Session | None = None) -> list[str]:
    """Absolute URLs of every link on a directory-listing page (parent links excluded)."""
    if not url.endswith("/"):
        url += "/"
    html = get(url, sess).text
    out = []
    for href in _HREF.findall(html):
        href = unescape(href)
        if href.startswith(("?", "#")) or "To Parent Directory" in href:
            continue
        absolute = urljoin(url, href)
        if len(absolute) <= len(url) and url.startswith(absolute):
            continue  # parent directory
        out.append(absolute)
    return out


def filename(url: str) -> str:
    return unquote(url.rstrip("/").rsplit("/", 1)[-1])
