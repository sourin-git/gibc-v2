"""Wikipedia-source exclusion.

Rejects a document when the normalized hostname of its source URL is wikipedia.org or any
subdomain of it. This is applied before tokenizer training and before corpus tokenization.

Scope: this reduces direct Wikipedia-source overlap with WikiText, which is built from Wikipedia
articles. It does NOT guarantee that the corpus is free of mirrored or quoted Wikipedia, WikiText
or benchmark content, and no contamination-free claim is made.
"""

from __future__ import annotations

from urllib.parse import urlsplit

EXCLUDED_DOMAIN = "wikipedia.org"


def normalized_hostname(url: str | None) -> str | None:
    """Lowercased hostname without port, userinfo or trailing dot; None if there is no host."""
    if not url:
        return None
    try:
        host = urlsplit(url.strip()).hostname  # urlsplit lowercases and drops port/userinfo
    except ValueError:
        return None
    host = host.rstrip(".") if host else ""
    return host or None


def is_wikipedia_source(url: str | None) -> bool:
    """True if the URL's host is wikipedia.org or a subdomain. URLs with no host are not excluded."""
    host = normalized_hostname(url)
    return host is not None and (host == EXCLUDED_DOMAIN or host.endswith("." + EXCLUDED_DOMAIN))
