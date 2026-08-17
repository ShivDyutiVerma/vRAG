"""IPv6 is broken on this dev machine's network (confirmed via `curl` on 2026-08-17 — see
docs/RISKS.md "Blockers found this session"): IPv6 connections to huggingface.co reset instead of
completing, while IPv4 works cleanly. `socket.getaddrinfo` normally returns IPv6 results first, and
most HTTP clients (httpx, requests, urllib3) try them before falling back to IPv4, which makes every
request pay a slow, sometimes-fatal timeout first. Importing this module forces IPv4-only DNS
resolution for the whole process. Machine-specific workaround, not a project dependency — import it
at the top of any script that talks to the network in this environment.
"""

import socket

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_only_getaddrinfo
