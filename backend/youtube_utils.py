"""YouTube URL parsing helpers for the frontend."""

import re
from urllib.parse import urlparse, parse_qs

# 11-character video id: letters, digits, dash, underscore.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_ALLOWED_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}


class InvalidYouTubeURL(ValueError):
    """Raised when a string is not a YouTube URL we can read an id out of."""


def extract_video_id(url: str) -> str:
    """Pull the 11-char video id out of any common YouTube URL form.

    Handles watch?v=, youtu.be/, /shorts/, /embed/, /live/, and tolerates
    extra query parameters such as &list=... &index=... &t=...
    A bare 11-character id is also accepted so the field stays forgiving.
    """
    if not url or not url.strip():
        raise InvalidYouTubeURL("Please paste a YouTube URL.")

    url = url.strip()

    # Allow someone to paste just the id.
    if _VIDEO_ID_RE.match(url):
        return url

    # urlparse needs a scheme to find the host.
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host not in _ALLOWED_HOSTS:
        raise InvalidYouTubeURL(f"'{host or url}' is not a YouTube address.")

    candidate = None

    if host in ("youtu.be", "www.youtu.be"):
        # youtu.be/<id>
        parts = [p for p in parsed.path.split("/") if p]
        if parts:
            candidate = parts[0]
    else:
        # /watch?v=<id> is the common case.
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            candidate = qs["v"][0]
        else:
            # /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                candidate = parts[1]

    if not candidate:
        raise InvalidYouTubeURL("Could not find a video id in that URL.")

    if not _VIDEO_ID_RE.match(candidate):
        raise InvalidYouTubeURL(f"'{candidate}' is not a valid video id.")

    return candidate


def watch_url(video_id: str) -> str:
    """Canonical watch URL, used for the st.video() preview."""
    return f"https://www.youtube.com/watch?v={video_id}"
