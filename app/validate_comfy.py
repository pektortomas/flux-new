"""Image-build contract test against the real ComfyUI /object_info, without weights."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from bootstrap import configuration
from workflows import build


def validate(graph, catalog):
    for ident, node in graph.api.items():
        kind = node["class_type"]
        assert kind in catalog, f"Missing core node: {kind}"
        spec = catalog[kind]
        required = spec["input"].get("required", {})
        allowed = {**required, **spec["input"].get("optional", {})}
        assert set(required) <= set(node["inputs"]), (kind, "missing required inputs")
        assert set(node["inputs"]) <= set(allowed), (kind, "unknown inputs")
        for key, value in node["inputs"].items():
            expected = allowed[key][0]
            if isinstance(value, list):
                origin = graph.api[value[0]]["class_type"]
                actual = catalog[origin]["output"][value[1]]
                assert actual == expected, (kind, key, actual, expected)
            elif isinstance(expected, list):
                assert value in expected, (kind, key, value, "not an allowed combo value")
        expected_outputs = list(spec["output"])
        actual_outputs = [x["type"] for x in graph.nodes[int(ident) - 1]["outputs"]]
        assert expected_outputs == actual_outputs, (kind, "output signature changed")


def main():
    os.environ.update(MODEL_PROFILES="quality,balanced,fast", ENABLE_SNAPSHOT="1", ENABLE_CONSISTENCY="1", EXTRA_LORAS_JSON="[]")
    config = configuration()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for asset in config["assets"]:
            f = root / "models" / asset["name"]
            f.parent.mkdir(parents=True, exist_ok=True)
            f.touch()  # Only populate loader choices; never execute these files.
        for folder in ("input", "output", "user"):
            (root / folder).mkdir()
        from PIL import Image
        for name in ("scene-reference.png", "identity-reference.png"):
            Image.new("RGB", (32, 32)).save(root / "input" / name)
        log = open(root / "server.log", "w+")
        process = subprocess.Popen([sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", "8199",
                                    "--disable-api-nodes", "--disable-all-custom-nodes", "--disable-auto-launch",
                                    "--models-directory", str(root / "models"), "--input-directory", str(root / "input"),
                                    "--output-directory", str(root / "output"), "--user-directory", str(root / "user")],
                                   cwd="/opt/ComfyUI", stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("ComfyUI CPU startup failed")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8199/object_info", timeout=5) as response:
                        catalog = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError("ComfyUI CPU startup timed out")
            count = 0
            for profile in config["profiles"]:
                for mode in ("text", "reference", "dual"):
                    for extras in (False, True):
                        validate(build(config, profile, mode, extras), catalog)
                        count += 1
            print(f"Validated {count} graph variants against real ComfyUI node schemas (CPU, no inference).")
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-12000:])
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
