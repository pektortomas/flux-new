import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import assets
import bootstrap
from workflows import build, write_workflows


def tensor_file(base="flux2_klein_9b"):
    header = json.dumps({"__metadata__": {"ss_base_model_version": base},
                         "diffusion_model.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\0" * 4


class Response:
    def __init__(self, content, status=200, headers=None):
        self.content, self.status_code, self.headers = content, status, headers or {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def raise_for_status(self):
        pass
    def iter_content(self, size):
        yield self.content


class AssetTests(unittest.TestCase):
    def test_reject_ambiguous_civitai_model(self):
        with self.assertRaisesRegex(ValueError, "VERSION"):
            assets.source_spec("https://civitai.com/models/123/example")

    def test_reject_token_urls_and_untrusted_hosts(self):
        for url in ("http://huggingface.co/a/b/resolve/main/a.safetensors", "https://evil.test/a", "https://civitai.com/api/download/models/42?token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                assets.source_spec(url)

    def test_hf_revision_and_filename(self):
        with patch.object(assets, "hf_spec", return_value={}) as fn:
            assets.source_spec("https://huggingface.co/me/repo/resolve/abcdef/sub/a%20b.safetensors", "local.safetensors")
            fn.assert_called_once_with("me/repo", "sub/a b.safetensors", "abcdef", "local.safetensors")

    def test_civitai_version_selection(self):
        data = {"baseModel": "Flux.2 Klein 9B", "files": [{"type": "Model", "name": "sample.safetensors", "primary": True, "sizeKB": 1, "hashes": {"SHA256": "a" * 64}, "downloadUrl": "https://civitai.com/api/download/models/42?type=Model&format=SafeTensor&fp=bf16"}]}
        with patch.object(assets, "api_json", return_value=data) as fn:
            a = assets.source_spec("https://civitai.com/models/12/name?modelVersionId=42")
            self.assertEqual(a["civitai_version"], "42")
            self.assertTrue(fn.call_args.args[0].endswith("/42"))

    def test_token_host_scoping(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf-secret", "CIVITAI_TOKEN": "c-secret"}):
            self.assertEqual(assets.auth("https://huggingface.co/a")["Authorization"], "Bearer hf-secret")
            self.assertEqual(assets.auth("https://cdn.example/a"), {})
            self.assertEqual(assets.auth("https://huggingface.co.evil/a"), {})

    def test_reject_path_traversal(self):
        for name in ("../x.safetensors", "/x.safetensors", "x.pt", "a/b.safetensors"):
            with self.assertRaises(ValueError):
                assets.safe_name(name)

    def test_incompatible_and_corrupt_lora(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "x.safetensors"
            p.write_bytes(tensor_file("flux1"))
            with self.assertRaisesRegex(ValueError, "other than Klein"):
                assets.inspect_safetensors(p, True)
            p.write_bytes(tensor_file()[:-2])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                assets.inspect_safetensors(p, True)

    def test_resume_then_cache_without_network(self):
        body = tensor_file()
        digest = hashlib.sha256(body).hexdigest()
        spec = {"url": "https://civitai.com/api/download/models/42", "name": "loras/test.safetensors", "sha256": digest}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "loras").mkdir()
            partial = root / f"loras/test.safetensors.{digest[:16]}.part"
            partial.write_bytes(body[:20])
            with patch.object(assets.requests, "get", return_value=Response(body[20:], 206, {"Content-Range": f"bytes 20-{len(body)-1}/{len(body)}"})) as get:
                result = assets.download(spec, root)
                self.assertEqual(result.read_bytes(), body)
                self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=20-")
            with patch.object(assets.requests, "get", side_effect=AssertionError("cache hit must not download")):
                assets.download(spec, root)

    def test_server_ignores_range_replaces_partial(self):
        body = tensor_file()
        digest = hashlib.sha256(body).hexdigest()
        spec = {"url": "https://civitai.com/api/download/models/42", "name": "loras/test.safetensors", "sha256": digest}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "loras").mkdir()
            (root / f"loras/test.safetensors.{digest[:16]}.part").write_bytes(body[:20])
            with patch.object(assets.requests, "get", return_value=Response(body)):
                self.assertEqual(assets.download(spec, root).read_bytes(), body)

    def test_bad_hash_never_installed(self):
        spec = {"url": "https://civitai.com/api/download/models/42", "name": "loras/test.safetensors", "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as folder, patch.object(assets.requests, "get", return_value=Response(tensor_file())):
            with self.assertRaisesRegex(ValueError, "failed after retries"):
                assets.download(spec, Path(folder))
            self.assertFalse((Path(folder) / spec["name"]).exists())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"MODEL_PROFILES": "quality,balanced,fast", "ENABLE_CONSISTENCY": "1"}, clear=True):
            self.config = bootstrap.configuration()

    def test_all_workflows_have_maja_connected_to_sampler(self):
        for profile in self.config["profiles"]:
            for mode in ("text", "reference", "dual"):
                for extras in (False, True):
                    g = build(self.config, profile, mode, extras)
                    nodes = g.api
                    guider = next(x for x in nodes.values() if x["class_type"] == "CFGGuider")
                    current = guider["inputs"]["model"][0]
                    chain = []
                    while nodes[current]["class_type"] == "LoraLoaderModelOnly":
                        chain.append(nodes[current]["inputs"]["lora_name"])
                        current = nodes[current]["inputs"]["model"][0]
                    self.assertIn("Maja.safetensors", chain)
                    self.assertEqual(nodes[current]["class_type"], "UNETLoader")
                    self.assertEqual(guider["inputs"]["cfg"], 1.0 if profile == "fast" else 4.0 if profile == "quality" else 5.0)
                    refs = [n for n in nodes.values() if n["class_type"] == "ReferenceLatent"]
                    self.assertEqual(len(refs), {"text": 0, "reference": 2, "dual": 4}[mode])

    def test_ui_links_match_api_and_outputs(self):
        for profile in self.config["profiles"]:
            for mode in ("text", "reference", "dual"):
                g = build(self.config, profile, mode, True)
                for link_id, source, slot, dest, dest_slot, typ in g.links:
                    origin = g.nodes[source - 1]
                    target = g.nodes[dest - 1]
                    self.assertIn(link_id, origin["outputs"][slot]["links"])
                    self.assertEqual(target["inputs"][dest_slot]["link"], link_id)
                    key = target["inputs"][dest_slot]["name"]
                    self.assertEqual(g.api[str(dest)]["inputs"][key], [str(source), slot])
                    self.assertLess(source, dest)

    def test_profile_scopes_and_trigger(self):
        g = build(self.config, "quality", "text", True)
        prompt = next(n["inputs"]["text"] for n in g.api.values() if n["class_type"] == "CLIPTextEncode")
        self.assertIn("maja", prompt)
        self.assertIn("amtr snapshot photo", prompt)
        self.assertFalse(any("consistency" in str(n) for n in g.api.values()))
        fast = build(self.config, "fast", "reference", True)
        self.assertTrue(any("consistency" in str(n) for n in fast.api.values()))
        self.assertFalse(any("snapshot-v9" in str(n) for n in fast.api.values()))

    def test_generated_files_and_personal_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            personal = root / "user/default/workflows/my-edits.json"
            personal.parent.mkdir(parents=True)
            personal.write_text("{}")
            write_workflows(self.config, root)
            self.assertEqual(len(list((root / "workflows/api").glob("*.json"))), 17)
            self.assertTrue(personal.exists())

    def test_duplicate_names_fail_before_download(self):
        with patch.dict(os.environ, {"EXTRA_LORAS_JSON": '[{"source":"42"}]'}, clear=True), patch.object(bootstrap, "source_spec", return_value={"name": "loras/Maja.safetensors"}):
            with self.assertRaisesRegex(ValueError, "same local filename"):
                bootstrap.configuration()


if __name__ == "__main__":
    unittest.main()
