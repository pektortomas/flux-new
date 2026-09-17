import fcntl
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys

from assets import check_access, download, source_spec
from workflows import write_workflows

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("bootstrap")


def lora_options(item):
    strength = float(item.get("strength", 0.4))
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite.")
    modes = item.get("modes", ["text", "reference", "dual"])
    profiles = item.get("profiles", ["quality", "balanced", "fast"])
    if not isinstance(modes, list) or set(modes) - {"text", "reference", "dual"}:
        raise ValueError("LoRA modes must be text, reference, dual.")
    if not isinstance(profiles, list) or set(profiles) - {"quality", "balanced", "fast"}:
        raise ValueError("LoRA profiles must be quality, balanced, fast.")
    return {"strength": strength, "trigger": str(item.get("trigger", "")), "modes": modes, "profiles": profiles}


def configuration():
    lock = json.loads((ROOT / "config/assets.lock.json").read_text())
    profiles = list(dict.fromkeys(x.strip() for x in os.getenv("MODEL_PROFILES", "quality,fast").split(",") if x.strip()))
    if not profiles or set(profiles) - {"quality", "balanced", "fast"}:
        raise ValueError("MODEL_PROFILES accepts quality,balanced,fast.")
    encoder = os.getenv("TEXT_ENCODER_PRECISION", "fp8")
    if encoder not in ("fp8", "bf16"):
        raise ValueError("TEXT_ENCODER_PRECISION must be fp8 or bf16.")
    text_encoder = lock["text_encoder_bf16" if encoder == "bf16" else "text_encoder"]
    maja = source_spec(os.environ["MAJA_SOURCE"], "Maja.safetensors") if os.getenv("MAJA_SOURCE") else lock["maja"]
    maja_options = lora_options({"strength": os.getenv("MAJA_STRENGTH", "1.0"), "trigger": os.getenv("MAJA_TRIGGER", "maja")})
    maja_options["name"] = maja["name"].split("/", 1)[1]
    assets = [maja, text_encoder, lock["vae"]] + [lock[p] for p in profiles]
    extras = []
    if os.getenv("ENABLE_SNAPSHOT", "1") == "1":
        assets.append(lock["snapshot"])
        extras.append({"name": lock["snapshot"]["name"].split("/", 1)[1], **lora_options({
            "strength": os.getenv("SNAPSHOT_STRENGTH", "0.35"), "trigger": "amtr snapshot photo",
            "profiles": ["quality", "balanced"]})})
    if os.getenv("ENABLE_CONSISTENCY", "0") == "1":
        assets.append(lock["consistency"])
        extras.append({"name": lock["consistency"]["name"].split("/", 1)[1], **lora_options({
            "strength": os.getenv("CONSISTENCY_STRENGTH", "0.35"),
            "profiles": ["fast"], "modes": ["reference", "dual"]})})
    declared = json.loads(os.getenv("EXTRA_LORAS_JSON", "[]"))
    if not isinstance(declared, list):
        raise ValueError("EXTRA_LORAS_JSON must be a JSON array.")
    for item in declared:
        if not isinstance(item, dict) or "source" not in item:
            raise ValueError("Each extra LoRA needs a source.")
        if set(item) - {"source", "name", "strength", "trigger", "modes", "profiles"}:
            raise ValueError("Unknown extra LoRA option.")
        options = lora_options(item)
        asset = source_spec(item["source"], item.get("name"))
        assets.append(asset)
        extras.append({"name": asset["name"].split("/", 1)[1], **options})
    names = [a["name"] for a in assets]
    if len(names) != len(set(names)):
        raise ValueError("Two assets use the same local filename. Set a unique name for each extra LoRA.")
    return {"profiles": profiles, "assets": assets, "maja": maja_options, "extras": extras,
            "models": {p: lock[p]["name"].split("/", 1)[1] for p in profiles},
            "text_encoder": text_encoder["name"].split("/", 1)[1], "vae": lock["vae"]["name"].split("/", 1)[1]}


def main():
    workspace = Path(os.getenv("WORKSPACE", "/workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    with open(workspace / ".maja-bootstrap.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        config = configuration()
        models = workspace / "models"
        models.mkdir(exist_ok=True)
        # Estimate is conservative on corrupt cached files and partial downloads.
        needed = sum(a.get("size", 0) for a in config["assets"] if not (models / a["name"]).is_file())
        if shutil.disk_usage(workspace).free < needed + 2 * 1024**3:
            raise ValueError(f"Insufficient volume space: need about {needed / 1e9 + 2.2:.1f} GB free for selected assets.")
        for asset in config["assets"]:
            if not (models / asset["name"]).is_file():
                check_access(asset)
        for asset in config["assets"]:
            download(asset, models)
        for folder in ("input", "output", "temp", "user/default/workflows/maja-generated", "workflows/api"):
            (workspace / folder).mkdir(parents=True, exist_ok=True)
        write_workflows(config, workspace)
        # Only public asset identifiers and generation settings; never environment or tokens.
        (workspace / "resolved-assets.json").write_text(json.dumps(config, indent=2) + "\n")
        LOG.info("Maja and all declared assets verified. Workflows are ready.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        LOG.error("Startup stopped: %s", error)
        sys.exit(1)
