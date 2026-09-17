"""Download only declared safetensors. No model code is executed by this module."""
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import re
import struct
import time
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

LOG = logging.getLogger("assets")


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_. -]*\.safetensors", name):
        raise ValueError("Local LoRA name must be a plain .safetensors filename (no paths).")
    return name


def auth(url):
    host = urlparse(url).hostname
    key = "HF_TOKEN" if host == "huggingface.co" else "CIVITAI_TOKEN" if host == "civitai.com" else None
    token = os.environ.get(key, "") if key else ""
    return {"Authorization": "Bearer " + token} if token else {}


def api_json(url):
    # requests removes Authorization on cross-host redirects, including signed CDN URLs.
    try:
        response = requests.get(url, headers=auth(url), timeout=(20, 60))
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError):
        raise ValueError("Cannot read asset metadata. Check repo/version, HF access or Civitai token.") from None


def hf_spec(repo, filename, revision="main", name=None):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise ValueError("HF repo must be owner/repository.")
    path = PurePosixPath(filename)
    if path.is_absolute() or ".." in path.parts or not filename.endswith(".safetensors"):
        raise ValueError("HF filename must name a .safetensors file inside the repo.")
    info = api_json(f"https://huggingface.co/api/models/{repo}/revision/{quote(revision, safe='')}?blobs=true")
    entry = next((s for s in info.get("siblings", []) if s["rfilename"] == filename), None)
    if not entry or not entry.get("lfs", {}).get("sha256"):
        raise ValueError("HF file not found or missing SHA256 integrity metadata.")
    return {"repo": repo, "revision": info["sha"], "file": filename,
            "name": "loras/" + safe_name(name or path.name),
            "sha256": entry["lfs"]["sha256"], "size": entry["size"]}


def source_spec(source, name=None):
    """Accept an HF resolve URL, HF descriptor, Civitai version ID or version URL."""
    if isinstance(source, dict):
        if set(source) - {"repo", "file", "revision"}:
            raise ValueError("HF source accepts repo, file, revision only.")
        return hf_spec(source["repo"], source["file"], source.get("revision", "main"), name)
    value = str(source).strip()
    if value.isdigit():
        version = value
    else:
        u = urlparse(value)
        if u.scheme != "https" or u.username or u.password or u.port not in (None, 443):
            raise ValueError("Asset source must use HTTPS without credentials.")
        query = parse_qs(u.query)
        if set(query) - {"download", "modelVersionId", "type", "format", "size", "fp"}:
            raise ValueError("Do not put tokens in URLs. Use HF_TOKEN / CIVITAI_TOKEN.")
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        if u.hostname == "huggingface.co" and len(parts) >= 5 and parts[2] == "resolve":
            return hf_spec("/".join(parts[:2]), "/".join(parts[4:]), parts[3], name)
        if u.hostname != "civitai.com":
            raise ValueError("Only Hugging Face resolve URLs and civitai.com are supported.")
        if parts[:3] == ["api", "download", "models"] and len(parts) == 4:
            version = parts[3]
        elif parts and parts[0] == "models" and query.get("modelVersionId"):
            version = query["modelVersionId"][0]
        else:
            raise ValueError("Civitai needs a MODEL VERSION ID, not a model ID. Add ?modelVersionId=... .")
    if not version.isdigit():
        raise ValueError("Civitai model version ID must be numeric.")
    data = api_json("https://civitai.com/api/v1/model-versions/" + version)
    files = [f for f in data.get("files", []) if f.get("type") == "Model" and f.get("name", "").endswith(".safetensors")]
    primary = [f for f in files if f.get("primary")]
    selected = primary if len(primary) == 1 else files
    if len(selected) != 1:
        raise ValueError("Civitai version has no unambiguous safetensors model file.")
    item = selected[0]
    digest = item.get("hashes", {}).get("SHA256", "").lower()
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Civitai file is missing SHA256.")
    url = item["downloadUrl"]
    endpoint = urlparse(url)
    if endpoint.scheme != "https" or endpoint.hostname != "civitai.com" or endpoint.username or endpoint.password or endpoint.port not in (None, 443) or set(parse_qs(endpoint.query)) - {"type", "format", "size", "fp"}:
        raise ValueError("Unexpected Civitai download endpoint.")
    return {"url": url, "name": "loras/" + safe_name(name or item["name"]),
            "sha256": digest, "size": int(item.get("sizeKB", 0) * 1024),
            "civitai_version": version, "declared_base": data.get("baseModel", "")}


def asset_url(spec):
    if "repo" in spec:
        return f"https://huggingface.co/{spec['repo']}/resolve/{quote(spec['revision'], safe='')}/{quote(spec['file'], safe='/')}"
    return spec["url"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_access(spec):
    url = asset_url(spec)
    try:
        with requests.head(url, headers=auth(url), allow_redirects=True, timeout=(20, 60)) as response:
            if response.status_code in (401, 403, 404):
                repo = spec.get("repo", "Civitai version " + spec.get("civitai_version", ""))
                raise ValueError(f"Access denied or file missing: {repo} / {Path(spec['name']).name}. Check token and accept access conditions on the original model page.")
            response.raise_for_status()
    except requests.RequestException:
        raise ValueError("Cannot verify download access for " + Path(spec["name"]).name) from None


def inspect_safetensors(path, lora=False):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated safetensors file.")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 32 * 1024 * 1024:
            raise ValueError("Not a valid safetensors header.")
        header = json.loads(f.read(length))
    payload = path.stat().st_size - 8 - length
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors or any(not 0 <= v["data_offsets"][0] <= v["data_offsets"][1] <= payload for v in tensors.values()):
        raise ValueError("Invalid or incomplete safetensors tensor data.")
    metadata = header.get("__metadata__", {})
    if lora:
        if not any("lora" in k.lower() for k in tensors):
            raise ValueError("Declared LoRA does not contain LoRA tensors.")
        base = str(metadata.get("ss_base_model_version", "")).lower()
        if base and ("klein" not in base or "9b" not in base):
            raise ValueError("LoRA metadata declares a model other than Klein 9B: " + base)
        if not base:
            LOG.warning("%s: no architecture metadata; actual layer compatibility needs GPU verification.", path.name)
    return metadata


def download(spec, models):
    relative = PurePosixPath(spec["name"])
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
        raise ValueError("Invalid model destination.")
    target = models / str(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not follow user-created destination symlinks outside the mounted model tree.
    if not target.resolve().is_relative_to(models.resolve()):
        raise ValueError("Model destination escapes the model directory.")
    digest = spec["sha256"].lower()
    if target.is_file() and sha256(target) == digest:
        inspect_safetensors(target, relative.parts[0] == "loras")
        LOG.info("Verified cached %s", target.name)
        return target
    # Hash in the partial filename prevents resuming a different model revision.
    partial = target.with_name(target.name + "." + digest[:16] + ".part")
    if partial.is_symlink():
        raise ValueError("Partial download must not be a symlink.")
    url = asset_url(spec)
    LOG.info("Downloading %s (%.2f GB)", target.name, spec.get("size", 0) / 1e9)
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = auth(url)
        headers["Accept-Encoding"] = "identity"
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as r:
                if r.status_code in (401, 403, 404):
                    raise ValueError(f"Cannot download {target.name}: HTTP {r.status_code}. Check token, gated-repo access, and version.")
                if r.status_code == 416:
                    if sha256(partial) == digest:
                        inspect_safetensors(partial, relative.parts[0] == "loras")
                        partial.replace(target)
                        return target
                    partial.unlink(missing_ok=True)
                    continue
                r.raise_for_status()
                append = offset > 0 and r.status_code == 206
                if append and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an incorrect resume offset.")
                with open(partial, "ab" if append else "wb") as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if sha256(partial) != digest:
                partial.unlink(missing_ok=True)
                LOG.warning("Integrity mismatch for %s; retrying.", target.name)
                continue
            inspect_safetensors(partial, relative.parts[0] == "loras")
            partial.replace(target)
            LOG.info("Downloaded and verified %s", target.name)
            return target
        except requests.RequestException:
            # Exception strings may contain signed URLs; never log them or tokens.
            LOG.warning("Network interruption downloading %s; retry %d/4", target.name, attempt + 1)
            time.sleep(min(2 ** attempt, 8))
    raise ValueError("Download failed after retries: " + target.name)
