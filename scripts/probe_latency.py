"""Phase 0, the most important script: measure real network physics from this machine to every
candidate provider BEFORE any architecture decision. Per AGENT_BUILD_SPEC.md §3.1 — a round trip
from an Indian server to a US-hosted API can consume the entire 200ms budget before a token is
generated, and no amount of FAISS tuning fixes that.

Two independent measurements, deliberately kept separate:
  1. Raw TCP-connect + TLS-handshake time to each provider's host — pure network physics, needs
     no API key, and is honest about what's a network cost vs. a provider processing cost.
  2. End-to-end TTFT (time-to-first-token) for a minimal real request — needs a valid API key.

Endpoints below were looked up against Sarvam's and Groq's current public API docs
(docs.sarvam.ai, console.groq.com) on 2026-08-17 rather than assumed from memory — record any
drift as a new entry in docs/DECISIONS_R.md if these ever change.

Output: P50/P95/P100 per provider per measurement, over N=30 samples, printed to stdout as a
markdown table ready to paste into docs/DECISIONS.md as ADR-003 (never invent numbers here that
this script didn't produce).

Note: this dev machine's IPv6 is broken (docs/RISKS.md), so `_netcompat` forces IPv4-only DNS
resolution here — these numbers are IPv4 physics specifically, which is also what the eventual
deployment host will use if it has working dual-stack routing.
"""

from __future__ import annotations

import argparse
import socket
import ssl
import statistics
import time
from dataclasses import dataclass, field

import _netcompat  # noqa: F401  — IPv6 is broken on this dev machine; forces IPv4 (see docs/RISKS.md)
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProbeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sarvam_api_key: str | None = None
    groq_api_key: str | None = None


@dataclass
class Provider:
    name: str
    host: str
    port: int = 443
    chat_url: str | None = None
    chat_headers: dict[str, str] = field(default_factory=dict)
    chat_model: str | None = None


def build_providers(settings: ProbeSettings) -> list[Provider]:
    providers = [
        Provider(name="sarvam", host="api.sarvam.ai"),
        Provider(name="groq", host="api.groq.com"),
    ]
    if settings.sarvam_api_key:
        sarvam = next(p for p in providers if p.name == "sarvam")
        sarvam.chat_url = "https://api.sarvam.ai/v1/chat/completions"
        sarvam.chat_headers = {"api-subscription-key": settings.sarvam_api_key}
        sarvam.chat_model = "sarvam-105b"
    if settings.groq_api_key:
        groq = next(p for p in providers if p.name == "groq")
        groq.chat_url = "https://api.groq.com/openai/v1/chat/completions"
        groq.chat_headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
        groq.chat_model = "llama-3.3-70b-versatile"
    return providers


def measure_tcp_tls_connect_ms(host: str, port: int = 443, timeout: float = 5.0) -> float | None:
    """Raw TCP connect + TLS handshake time, in ms. No API key needed — pure network physics."""
    ctx = ssl.create_default_context()
    t0 = time.perf_counter()
    try:
        with (
            socket.create_connection((host, port), timeout=timeout) as sock,
            ctx.wrap_socket(sock, server_hostname=host),
        ):
            pass
    except OSError:
        return None
    return (time.perf_counter() - t0) * 1000


def measure_chat_ttft_ms(
    client: httpx.Client, url: str, headers: dict[str, str], model: str, timeout: float = 15.0
) -> float | None:
    """Time to first streamed token for a minimal chat completion. Needs a real key."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}],
        "stream": True,
        "max_tokens": 8,
    }
    t0 = time.perf_counter()
    try:
        with client.stream(
            "POST", url, headers=headers, json=payload, timeout=timeout
        ) as response:
            if response.status_code != 200:
                return None
            for chunk in response.iter_bytes():
                if chunk:
                    return (time.perf_counter() - t0) * 1000
    except (httpx.HTTPError, OSError):
        return None
    return None


def percentiles(samples: list[float]) -> tuple[float, float, float]:
    s = sorted(samples)
    p50 = statistics.median(s)
    p95 = s[min(int(len(s) * 0.95), len(s) - 1)]
    p100 = s[-1]
    return p50, p95, p100


def run_probe(n_samples: int) -> None:
    settings = ProbeSettings()
    providers = build_providers(settings)

    print(f"# Provider latency probe — N={n_samples} samples per measurement\n")

    print("## 1. Raw TCP + TLS connect time (network physics, no API key needed)\n")
    print("| Provider | Host | P50 (ms) | P95 (ms) | P100 (ms) | Failures |")
    print("|----------|------|----------|----------|-----------|----------|")
    for provider in providers:
        samples: list[float] = []
        failures = 0
        for _ in range(n_samples):
            ms = measure_tcp_tls_connect_ms(provider.host, provider.port)
            if ms is None:
                failures += 1
            else:
                samples.append(ms)
        if samples:
            p50, p95, p100 = percentiles(samples)
            print(
                f"| {provider.name} | {provider.host} | {p50:.1f} | {p95:.1f} | "
                f"{p100:.1f} | {failures}/{n_samples} |"
            )
        else:
            print(
                f"| {provider.name} | {provider.host} | — | — | — | "
                f"{n_samples}/{n_samples} (all failed) |"
            )

    print("\n## 2. Chat completion TTFT (needs API key; skipped otherwise)\n")
    print("| Provider | Model | P50 (ms) | P95 (ms) | P100 (ms) | Failures |")
    print("|----------|-------|----------|----------|-----------|----------|")
    with httpx.Client() as client:
        for provider in providers:
            if not provider.chat_url:
                print(f"| {provider.name} | — | — | — | — | SKIPPED — no API key in .env |")
                continue
            samples = []
            failures = 0
            for _ in range(n_samples):
                ms = measure_chat_ttft_ms(
                    client, provider.chat_url, provider.chat_headers, provider.chat_model or ""
                )
                if ms is None:
                    failures += 1
                else:
                    samples.append(ms)
            if samples:
                p50, p95, p100 = percentiles(samples)
                print(
                    f"| {provider.name} | {provider.chat_model} | {p50:.1f} | {p95:.1f} | "
                    f"{p100:.1f} | {failures}/{n_samples} |"
                )
            else:
                print(
                    f"| {provider.name} | {provider.chat_model} | — | — | — | "
                    f"{n_samples}/{n_samples} (all failed) |"
                )

    print(
        "\n**Not yet covered by this script:** Sarvam STT streaming TTFT (needs real audio input, "
        "deferred to Phase 1 when a sample WAV exists) and a local llama.cpp baseline (needs a "
        "downloaded model). Both are known gaps, not silently skipped — see docs/RISKS.md."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="samples per measurement (default 30)")
    args = parser.parse_args()
    run_probe(args.n)
