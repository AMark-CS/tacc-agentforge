"""Check outbound HTTPS from a compute node (HEAD requests only; nothing is downloaded).

The first probe saw CERTIFICATE_VERIFY_FAILED from Python, which means a TLS connection was
made. This checks whether system CA certificates or a proxy change the result.
"""
import json
import os
import shutil
import ssl
import subprocess
import urllib.request
from pathlib import Path

URLS = ["https://huggingface.co", "https://hf-mirror.com", "https://pypi.org/simple/pip/",
        "https://files.pythonhosted.org", "https://github.com", "https://conda.anaconda.org"]
CA_CANDIDATES = ["/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt",
                 "/etc/ssl/cert.pem"]


def curl(url: str, extra: list[str] | None = None) -> str:
    cmd = ["curl", "-sS", "-I", "--max-time", "15", "-o", "/dev/null",
           "-w", "http=%{http_code} remote=%{remote_ip} ssl_verify=%{ssl_verify_result}",
           *(extra or []), url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + " " + out.stderr).strip()[:300]
    except Exception as exc:
        return f"ERROR {exc!r}"


def py_head(url: str, cafile: str | None) -> str:
    try:
        ctx = ssl.create_default_context(cafile=cafile)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=15, context=ctx) as r:
            return f"http={r.status}"
    except Exception as exc:
        return f"no: {exc!r}"[:300]


ca = next((c for c in CA_CANDIDATES if Path(c).exists()), None)
report = {
    "job": os.environ.get("SLURM_JOB_ID"),
    "proxy_env": {k: v for k, v in os.environ.items() if "proxy" in k.lower()},
    "ssl_cert_env": {k: v for k, v in os.environ.items() if k.startswith(("SSL_", "REQUESTS_", "CURL_", "PIP_", "CONDA_"))},
    "default_verify_paths": str(ssl.get_default_verify_paths()),
    "system_ca_bundle": ca,
    "which": {t: shutil.which(t) for t in ["curl", "wget", "git", "gcc", "g++", "pip", "conda"]},
    "curl_default": {u: curl(u) for u in URLS},
    "curl_insecure_reachability": {u: curl(u, ["-k"]) for u in URLS[:2]},
    "python_default": {u: py_head(u, None) for u in URLS[:3]},
    "python_system_ca": {u: py_head(u, ca) for u in URLS[:3]} if ca else "no system CA bundle found",
}
out = Path(os.environ["TACC_USERDIR"]) / "tacc-agentforge-results"
out.mkdir(parents=True, exist_ok=True)
path = out / f"net-check-{report['job']}.json"
path.write_text(json.dumps(report, indent=2))
print("REPORT", json.dumps(report, indent=2), flush=True)
