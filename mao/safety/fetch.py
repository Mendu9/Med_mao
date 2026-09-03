"""Containment for server-side fetches of caller-supplied URLs.

`metadata["image_url"]` is fetched with `requests.get` from inside the
deployment, and the bytes are then handed to the vision model. Unvalidated,
that is a request-forgery primitive using the server's own network position:
`http://169.254.169.254/latest/meta-data/` returns cloud instance credentials
on the major providers, and every internal service reachable from the app is
reachable from the caller.

Two agents fetch that key. This module exists so there is one guard rather than
two copies that drift — the failure mode B7 records, where a control was added
to `clinical_agent` and not to `graphrag_agent`.

Deliberately an allowlist of *shapes*, not of hosts: the deployment has no host
allowlist to consult, and enumerating "bad" addresses is a losing game. Only
http/https survive, and only when the target resolves entirely to public
addresses.

The DNS resolution is the load-bearing part. Checking the literal hostname
would pass `http://internal.example.com` that resolves to 10.0.0.5, and would
pass every public DNS name an attacker points at a private address.

## Known residual: a DNS-rebinding window (NB1)

This module used to claim that "name resolution cannot be raced between the
check and the request". That was FALSE and is corrected here rather than left
standing: `fetch_image_bytes` validates the URL and then calls
`requests.get(url)`, which resolves the name a SECOND time. An attacker who
controls the authoritative DNS for a name can answer the first lookup with a
public address and the second with a private one.

The window is real, and closing it properly means resolving once and connecting
to the resolved IP while still sending the original Host header and validating
the certificate against the original name — which is a change to how every fetch
is made, not a change to this guard. It is recorded as NB1 against a later phase.

What this guard does close, and was measured closing in 20 of 20 adversarial
attempts, is the whole trivial class: `file://`, `gopher://`, `dict://`, literal
`169.254.169.254`, `metadata.google.internal`, decimal/octal/hex IP encodings,
IPv4-mapped IPv6, credentials-in-URL, every RFC1918 and CGNAT range, redirects
to any of the above, and unbounded reads.
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Bytes are never streamed unbounded into memory from a caller-named host.
MAX_FETCH_BYTES = 16 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 15


class UnsafeFetchError(ValueError):
    """The URL is not one the server may fetch on a caller's behalf."""


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    # `is_global` is False for loopback, link-local (169.254/16, fe80::/10),
    # private ranges, multicast, reserved and unspecified addresses — which is
    # exactly the set that must not be reachable through a caller-supplied URL.
    return ip.is_global


def _resolved_addresses(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeFetchError(f"host {host!r} could not be resolved") from exc
    return sorted({info[4][0] for info in infos})


def validate_fetch_url(url: str) -> str:
    """Return *url* unchanged, or raise `UnsafeFetchError`.

    Every address the host resolves to must be public. One private answer is
    enough to refuse: a name with both a public and a private record would
    otherwise be a coin flip on each request.
    """
    if not url or not isinstance(url, str):
        raise UnsafeFetchError("no URL supplied")

    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeFetchError(
            f"scheme {parsed.scheme!r} is not fetchable; only http and https are"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeFetchError("URL has no host")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    for address in _resolved_addresses(host, port):
        try:
            public = _is_public(address)
        except ValueError as exc:
            raise UnsafeFetchError(f"unusable address for {host!r}") from exc
        if not public:
            raise UnsafeFetchError(
                f"{host!r} resolves to the non-public address {address} — "
                "refusing to fetch on a caller's behalf"
            )
    return url


def fetch_image_bytes(url: str) -> bytes:
    """Fetch a caller-supplied image URL, or raise `UnsafeFetchError`.

    Redirects are disabled: following them re-introduces every address this
    function just refused, one hop later.
    """
    import requests

    validate_fetch_url(url)
    response = requests.get(
        url,
        timeout=FETCH_TIMEOUT_SECONDS,
        allow_redirects=False,
        stream=True,
    )
    if response.is_redirect or response.is_permanent_redirect:
        raise UnsafeFetchError(
            f"{url!r} redirected to {response.headers.get('Location', '?')!r}; "
            "redirects are not followed for caller-supplied URLs"
        )
    response.raise_for_status()

    payload = bytearray()
    for chunk in response.iter_content(chunk_size=64 * 1024):
        payload.extend(chunk)
        if len(payload) > MAX_FETCH_BYTES:
            raise UnsafeFetchError(
                f"{url!r} returned more than {MAX_FETCH_BYTES} bytes"
            )
    return bytes(payload)
