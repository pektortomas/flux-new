"""Core-node graphs following Comfy-Org's Klein Euler/Flux2Scheduler recipes."""
import json
from pathlib import Path

PROFILES = {"quality": (50, 4.0), "balanced": (20, 5.0), "fast": (4, 1.0)}
SCENE = ("A casual smartphone photo of {trigger}, an adult woman sitting at a small café table. "
         "A friend seated directly opposite takes the photo from eye level, about one metre away. "
         "Framed from the waist up with the table edge visible. One forearm rests on the table, "
         "shoulders relaxed, looking at the friend with a slight smile. Soft daylight from a window "
         "on the left, natural skin texture, ordinary café interior still recognizable in the background. "
         "Everyday colors and natural exposure.")
EDIT = ("Replace the adult woman in image 1 with {trigger}. Preserve the camera position, "
        "perspective, framing, body pose, clothing and environment of image 1. Match the person "
        "to the existing light and shadows. Keep natural skin texture and the ordinary smartphone "
        "photography appearance.")
DUAL = ("Use image 1 for the scene, camera position, framing, body pose, clothing and lighting. "
        "Replace the adult woman with {trigger}, using image 2 for her facial identity. Preserve "
        "her facial proportions. Match her lighting to image 1. A natural everyday smartphone photo.")


class Graph:
    def __init__(self):
        self.api = {}
        self.nodes = []
        self.links = []

    def node(self, kind, title, values, outputs, pos, widgets=None):
        """Connected values are tuples (source_id, slot); literals are widgets."""
        ident = str(len(self.nodes) + 1)
        inputs = []
        api_inputs = {}
        for key, value in values.items():
            if isinstance(value, tuple):
                source, slot = value
                origin = self.nodes[int(source) - 1]
                socket_type = origin["outputs"][slot]["type"]
                link = len(self.links) + 1
                item = {"name": key, "type": socket_type, "link": link}
                if key in ("width", "height"):
                    item["widget"] = {"name": key}
                inputs.append(item)
                origin["outputs"][slot]["links"].append(link)
                self.links.append([link, int(source), slot, int(ident), len(inputs) - 1, socket_type])
                api_inputs[key] = [source, slot]
            else:
                api_inputs[key] = value
        values_ui = widgets if widgets is not None else [v for v in values.values() if not isinstance(v, tuple)]
        self.nodes.append({"id": int(ident), "type": kind, "title": title, "pos": list(pos),
                           "size": [340, 290 if kind == "CLIPTextEncode" else 180],
                           "flags": {}, "order": len(self.nodes), "mode": 0,
                           "inputs": inputs,
                           "outputs": [{"name": name, "type": typ, "links": [], "slot_index": i}
                                       for i, (name, typ) in enumerate(outputs)],
                           "properties": {"Node name for S&R": kind}, "widgets_values": values_ui})
        self.api[ident] = {"class_type": kind, "inputs": api_inputs, "_meta": {"title": title}}
        return ident, 0

    def ui(self):
        return {"last_node_id": len(self.nodes), "last_link_id": len(self.links), "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {}, "extra": {"ds": {"scale": 0.55, "offset": [60, 60]}}, "version": 0.4}


def build(config, profile, mode="text", enhanced=False):
    g = Graph()
    steps, cfg = PROFILES[profile]
    model = g.node("UNETLoader", profile + " — Klein 9B", {"unet_name": config["models"][profile], "weight_dtype": "default"}, [("MODEL", "MODEL")], (0, 0))
    clip = g.node("CLIPLoader", "Qwen3 8B — flux2", {"clip_name": config["text_encoder"], "type": "flux2", "device": "default"}, [("CLIP", "CLIP")], (0, 400))
    vae = g.node("VAELoader", "Standard FLUX.2 VAE", {"vae_name": config["vae"]}, [("VAE", "VAE")], (0, 650))
    loras = [config["maja"]]
    if enhanced:
        loras += [x for x in config["extras"] if mode in x["modes"] and profile in x["profiles"]]
    for index, lora in enumerate(loras):
        model = g.node("LoraLoaderModelOnly", "Maja identity" if index == 0 else "Optional: " + lora["name"],
                       {"model": model, "lora_name": lora["name"], "strength_model": lora["strength"]}, [("MODEL", "MODEL")], (400, index * 220))
    prompt = {"text": SCENE, "reference": EDIT, "dual": DUAL}[mode].format(trigger=config["maja"]["trigger"])
    triggers = [x["trigger"] for x in loras[1:] if x["trigger"]]
    if triggers:
        prompt += " " + ", ".join(triggers)
    positive = g.node("CLIPTextEncode", "Describe the real camera situation", {"clip": clip, "text": prompt}, [("CONDITIONING", "CONDITIONING")], (800, 0))
    if profile == "fast":
        negative = g.node("ConditioningZeroOut", "Distilled CFG 1", {"conditioning": positive}, [("CONDITIONING", "CONDITIONING")], (800, 350))
    else:
        negative = g.node("CLIPTextEncode", "Empty negative — base CFG", {"clip": clip, "text": ""}, [("CONDITIONING", "CONDITIONING")], (800, 350))
    width, height = 896, 1120
    for index in range({"text": 0, "reference": 1, "dual": 2}[mode]):
        name = "scene-reference.png" if index == 0 else "identity-reference.png"
        picture = g.node("LoadImage", "UPLOAD scene / pose reference" if index == 0 else "UPLOAD Maja identity reference",
                         {"image": name}, [("IMAGE", "IMAGE"), ("MASK", "MASK")], (0 + index * 400, 1000), [name, "image"])
        scaled = g.node("ImageScaleToTotalPixels", "Reference ~1 MP", {"image": picture, "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}, [("IMAGE", "IMAGE")], (800 + index * 400, 1000))
        if index == 0:
            size = g.node("GetImageSize", "Keep scene aspect ratio", {"image": scaled}, [("width", "INT"), ("height", "INT"), ("batch_size", "INT")], (1200, 1250))
            width, height = size, (size[0], 1)
        encoded = g.node("VAEEncode", "Encode reference " + str(index + 1), {"pixels": scaled, "vae": vae}, [("LATENT", "LATENT")], (1600 + index * 400, 1000))
        positive = g.node("ReferenceLatent", "Reference " + str(index + 1) + " → positive", {"conditioning": positive, "latent": encoded}, [("CONDITIONING", "CONDITIONING")], (1200 + index * 400, 0))
        negative = g.node("ReferenceLatent", "Reference " + str(index + 1) + " → negative", {"conditioning": negative, "latent": encoded}, [("CONDITIONING", "CONDITIONING")], (1200 + index * 400, 350))
    latent = g.node("EmptyFlux2LatentImage", "Output — 4:5 or reference aspect", {"width": width, "height": height, "batch_size": 1}, [("LATENT", "LATENT")], (2000, 750), [896, 1120, 1])
    scheduler = g.node("Flux2Scheduler", "Native Klein schedule", {"steps": steps, "width": width, "height": height}, [("SIGMAS", "SIGMAS")], (2000, 500), [steps, 896, 1120])
    guider = g.node("CFGGuider", "Base CFG" if profile != "fast" else "Distilled CFG = 1", {"model": model, "positive": positive, "negative": negative, "cfg": cfg}, [("GUIDER", "GUIDER")], (2000, 0))
    sampler = g.node("KSamplerSelect", "Official Euler baseline", {"sampler_name": "euler"}, [("SAMPLER", "SAMPLER")], (2000, 250))
    noise = g.node("RandomNoise", "Fixed seed for comparisons", {"noise_seed": 20260917}, [("NOISE", "NOISE")], (2400, 700), [20260917, "fixed"])
    sampled = g.node("SamplerCustomAdvanced", "Generate", {"noise": noise, "guider": guider, "sampler": sampler, "sigmas": scheduler, "latent_image": latent}, [("output", "LATENT"), ("denoised_output", "LATENT")], (2400, 0))
    decoded = g.node("VAEDecode", "Decode once — no face restorer", {"samples": sampled, "vae": vae}, [("IMAGE", "IMAGE")], (2800, 0))
    g.node("SaveImage", "Save PNG + workflow metadata", {"images": decoded, "filename_prefix": f"maja/{profile}-{mode}-{'extras' if enhanced else 'identity'}"}, [], (3200, 0))
    return g


def write_workflows(config, workspace):
    ui = workspace / "user/default/workflows/maja-generated"
    api = workspace / "workflows/api"
    ui.mkdir(parents=True, exist_ok=True)
    api.mkdir(parents=True, exist_ok=True)
    # This dedicated generated folder is owned by startup. Save personal edits elsewhere.
    for folder in (ui, api):
        for old in folder.glob("maja-*.json"):
            old.unlink()
    for profile in config["profiles"]:
        for mode in ("text", "reference", "dual"):
            variants = [False]
            if any(mode in x["modes"] and profile in x["profiles"] for x in config["extras"]):
                variants.append(True)
            for enhanced in variants:
                graph = build(config, profile, mode, enhanced)
                name = f"maja-{profile}-{mode}-{'extras' if enhanced else 'identity'}.json"
                (ui / name).write_text(json.dumps(graph.ui(), indent=2) + "\n")
                (api / name).write_text(json.dumps(graph.api, indent=2) + "\n")
