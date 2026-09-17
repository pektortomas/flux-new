"""Run on the deployed Pod; records real successful inference and timing."""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--workspace", type=Path, default=Path("/workspace"))
    p.add_argument("--profile", choices=["quality", "balanced", "fast"], default="fast")
    p.add_argument("--reference", type=Path)
    p.add_argument("--identity", type=Path)
    p.add_argument("--extras", action="store_true")
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--seed", type=int, default=20260917)
    args = p.parse_args()
    if args.identity and not args.reference:
        p.error("--identity requires --reference")
    if args.runs < 1:
        p.error("--runs must be positive")
    mode = "dual" if args.identity else "reference" if args.reference else "text"
    variant = "extras" if args.extras else "identity"
    path = args.workspace / f"workflows/api/maja-{args.profile}-{mode}-{variant}.json"
    graph = json.loads(path.read_text())
    for node in graph.values():
        if node["class_type"] == "RandomNoise":
            node["inputs"]["noise_seed"] = args.seed
        if node["class_type"] == "LoadImage":
            source = args.reference if node["inputs"]["image"] == "scene-reference.png" else args.identity
            with open(source, "rb") as f:
                r = requests.post(args.url + "/upload/image", files={"image": (source.name, f)}, data={"type": "input"}, timeout=120)
            r.raise_for_status()
            uploaded = r.json()
            node["inputs"]["image"] = "/".join(x for x in (uploaded.get("subfolder"), uploaded["name"]) if x)
    stats_response = requests.get(args.url + "/system_stats", timeout=10)
    stats_response.raise_for_status()
    stats = stats_response.json()
    results = []
    for run in range(args.runs):
        q = requests.get(args.url + "/queue", timeout=10).json()
        if q.get("queue_running") or q.get("queue_pending"):
            raise SystemExit("Queue is busy. Run the benchmark when no other generation is active.")
        peak = [0]
        stop = threading.Event()
        def sample_memory():
            while not stop.is_set():
                try:
                    output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=3)
                    peak[0] = max(peak[0], int(output.decode().strip()))
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                stop.wait(0.5)
        thread = threading.Thread(target=sample_memory, daemon=True)
        thread.start()
        start = time.monotonic()
        try:
            response = requests.post(args.url + "/prompt", json={"prompt": graph}, timeout=30)
            if response.status_code != 200:
                raise RuntimeError("ComfyUI rejected the workflow: " + response.text[:4000])
            prompt_id = response.json()["prompt_id"]
            while time.monotonic() - start < args.timeout:
                r = requests.get(args.url + "/history/" + prompt_id, timeout=15)
                r.raise_for_status()
                history = r.json().get(prompt_id)
                if history:
                    if history.get("status", {}).get("status_str") == "error":
                        raise RuntimeError("Inference failed: " + json.dumps(history["status"]))
                    images = [image for output in history.get("outputs", {}).values() for image in output.get("images", [])]
                    if not images:
                        raise RuntimeError("Execution completed without saved images.")
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Timed out; generation may still be running. Inspect the ComfyUI queue.")
            result = {"run": run + 1, "seconds": round(time.monotonic() - start, 2),
                      "sampled_peak_gpu_memory_mib": peak[0] or None, "prompt_id": prompt_id, "images": images}
            results.append(result)
            print(json.dumps(result), flush=True)
        finally:
            stop.set()
            thread.join(timeout=5)
    report = args.workspace / "output" / f"benchmark-{args.profile}-{mode}-{variant}-{int(time.time())}.json"
    report.write_text(json.dumps({"profile": args.profile, "mode": mode, "extras": args.extras, "seed": args.seed,
                                 "system": stats, "runs": results,
                                 "notes": "Times include HTTP polling. First run may include model loading. GPU memory sampled at 0.5 s; not a precise allocation peak. Images still require visual review."}, indent=2))
    print("Report:", report)


if __name__ == "__main__":
    main()
