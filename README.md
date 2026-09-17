# Maja — FLUX.2 klein 9B na RunPodu

Docker projekt pro GPU Pod s ComfyUI, RTX 5090 a povinnou identitní LoRA `Maja.safetensors`, trigger `maja`. Modely se stahují při startu na persistentní volume. Tokeny ani váhy nejsou součástí image.

**Stav dodávky:** připravený zdroj pro sestavení image, připnuté závislosti a workflow. Lokální testy a skutečné stažení Maja prošly. Docker image zatím nebyl sestaven/publikován a inference na RTX 5090 nebyla provedena. Podrobnosti v [ověření](docs/VERIFICATION.md). Názvy profilů popisují konfiguraci, nikoli naměřené pořadí vizuální kvality.

## Co je připravené

- PyTorch 2.10.0, CUDA 12.8, linux/amd64; Docker základ připnutý digestem.
- ComfyUI v0.36.0 připnuté na commit, vyřešený Python lock pro Linux/Python 3.12.
- Povinná Maja; při chybě stažení/integrity se ComfyUI nespustí s chybějící LoRA.
- Nativní uzly ComfyUI, bez instalace custom nodes nebo aktualizací při startu.
- Text-to-image, jedna kompoziční reference a dvě reference (scéna + identita).
- Oddělené workflow pouze s Majou a s přidaným stylem. Každé workflow obsahuje Maja LoRA.
- Volitelné LoRA z Hugging Face a Civitai přes JSON env; kontrola SHA-256, obnovitelné stahování, ověření známé architektury.
- Standardní FLUX.2 VAE a PNG s metadaty. Bez automatického face restoreru, generativního upscaleru nebo sharpening řetězce.

## Profily

| `MODEL_PROFILES` | Generátor | Kroky / CFG | Účel |
|---|---|---|---|
| `quality` | Klein **base 9B BF16** | 50 / 4 | Výchozí plná přesnost generátoru; pomalejší |
| `balanced` | Klein **base 9B FP8** | 20 / 5 | Úspornější postup podle ComfyUI šablony |
| `fast` | Klein **distilled 9B FP8** | 4 / 1 | Rychlé hledání scén a kompozice |

Výchozí `quality,fast`. Textový enkodér je Qwen3 8B FP8 mixed, společný pro profily. `TEXT_ENCODER_PRECISION=bf16` stáhne jeho BF16 variantu; na 32GB kartě počítej s větším přesouváním mezi RAM a VRAM. Automatická správa paměti ComfyUI zůstává zapnutá. Žádný `--gpu-only` ani `--highvram`.

Textové workflow generuje 896 × 1120 (4:5), batch 1. Referenční workflow zachová poměr stran první reference a zmenší ji přibližně na 1 MP s rozměry dělitelnými 16. Pro feed tedy použij kompoziční referenci předem oříznutou na 4:5.

## Sestavení a publikování

Nejjednodušší bez lokálního Dockeru:

1. Cílový repozitář je [pektortomas/flux-new](https://github.com/pektortomas/flux-new). Projekt patří do jeho kořene.
2. V Docker Hubu vytvoř repository `flux-new` a access token s oprávněním zápisu do tohoto repository.
3. GitHub → repository Settings → Secrets and variables → Actions: nastav **secret** `DOCKERHUB_TOKEN`. Účet `pektor` je přednastavený; volitelná **variable** `DOCKERHUB_USERNAME` ho může změnit. Volitelná variable `DOCKERHUB_REPOSITORY` mění jméno image, jinak je `flux-new`.
4. V Actions spusť **Build and publish RunPod image**. Workflow se samo nespustí pouhým commitem.
5. Po úspěšném dokončení použij `docker.io/pektor/flux-new:2026-09-17-1`.
6. Pokud je Docker Hub repository private, přidej v RunPodu registry credentials pro čtení. Image neobsahuje HF token ani modely.

Build ověřuje Python závislosti, jednotkové testy, CPU start ComfyUI a všechny graph varianty proti skutečnému `/object_info`. **Tyto kontroly nenahrazují GPU generování.**

Na Linux stroji s Dockerem a dostatkem disku lze místo Actions spustit:

```bash
docker buildx build --platform linux/amd64 \
  -t docker.io/pektor/flux-new:2026-09-17-1 --push .
```

Počítej alespoň s 30 GB volného prostoru pro build a jeho cache. Při dalších změnách použij nový release tag a v RunPodu ideálně výsledný digest.

## Nasazení do RunPod GPU Pod

| Položka | Hodnota |
|---|---|
| GPU | 1× RTX 5090, 32 GB VRAM |
| RAM hosta | doporučeno alespoň 64 GB; více pomůže při více referencích/offloadu |
| CUDA compatibility filtr | 12.8 nebo novější |
| Container image | publikovaný Docker Hub tag nebo digest |
| Container disk | 30 GB |
| Persistent volume | 100 GB, mount `/workspace`; pro přenos mezi Pody network volume |
| HTTP port | `8188` |
| Entrypoint / start command | ponechat prázdné, použít nastavení image |

Výchozí váhy potřebují přibližně 36.8 GB místa, bez výstupů. Přidání `balanced` přidá 9.6 GB. Změna textového enkodéru na BF16 znamená dalších 16.4 GB stahování; stará varianta zůstane na volume, dokud ji sám neodstraníš.

Před prvním startem přijmi podmínky přístupu na Hugging Face pro vybrané BFL repozitáře:

- [Base 9B BF16](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) pro `quality`.
- [9B FP8 distilled](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) pro `fast`.
- [Base 9B FP8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) pro `balanced`.

Jde o gated repozitáře s licencí BFL pro model, nikoli Apache 2.0. Použij vlastní HF read token s přístupem k těmto repozitářům a ulož ho do RunPod secrets/env. Podmínky komerčního provozu modelu a použití jeho výstupů posuzuj podle konkrétní licence BFL.

Minimální env:

```dotenv
HF_TOKEN=tvuj_read_token_vlozeny_pouze_v_RunPodu
MODEL_PROFILES=quality,fast
MAJA_TRIGGER=maja
MAJA_STRENGTH=1.0
ENABLE_SNAPSHOT=1
SNAPSHOT_STRENGTH=0.35
EXTRA_LORAS_JSON=[]
```

Konkrétní template skeleton je v [config/runpod-template.json](config/runpod-template.json). Obsahuje cílový tag `pektor/flux-new:2026-09-17-1`, který bude existovat až po úspěšném buildu. Nahraď placeholder HF tokenu skutečným RunPod secretem.

Po startu proběhne CUDA výpočet, kontrola přístupu a volného místa, stažení, SHA-256 ověření a vytvoření workflow. První download může trvat déle; port se otevře až po dokončení. Další start soubory znovu hashově ověří, ale správné soubory znovu nestahuje.

V RunPod Connect otevři ComfyUI na portu 8188. Ve workflow browseru vyber `maja-generated`. Tento kontejner sám nepřidává login; přístupový bod chraň podle zvoleného způsobu připojení a nesdílej přístup k ovládání GPU.

## Doporučený první záběr

1. Otevři `maja-quality-reference-identity.json`.
2. V `LoadImage` nahraj skutečnou referenci s uvěřitelnou pózou a úhlem. Placeholder obrázky nejsou součástí běhu.
3. Prompt už žádá zachovat kameru, pózu a světlo a použít `maja`. Uprav oblečení/scénu podle potřeby.
4. Porovnej stejný seed s `maja-quality-reference-extras.json`, kde je navíc snapshot LoRA na 0.35.
5. Vyber vizuálně lepší variantu. Pokud styl mění tvář nebo přidává přílišný kontrast, použij identity workflow nebo sniž sílu stylu.

Pro dvě reference otevři `dual`: první je scéna/póza, druhá obličej Maji. Nativní reference není přesný OpenPose nebo Depth ControlNet; kamera ani identita nejsou matematicky uzamčené. Jde o výrazně konkrétnější vodítko než samotný text.

Snapshot 0.35 je konzervativní výchozí pokus, ne GPU ověřené optimum. Nevyměňuj base/distilled checkpoint pouze v loaderu bez změny kroků a CFG; použij připravený profil.

## Další LoRA při deployi

`Maja.safetensors` se bere z připnuté verze tvého repozitáře automaticky. `MAJA_SOURCE` může změnit zdroj identitní LoRA; chybějící/poškozenou LoRA nelze obejít přepínačem. Výchozí trigger je `maja`.

`EXTRA_LORAS_JSON` je JSON pole. Příklad HF LoRA, kterou už máš v repozitáři, jako alternativního stylového testu:

```json
[
  {
    "source": {
      "repo": "TommyPekk/maja_v2",
      "file": "V3_flux_klein.safetensors",
      "revision": "b7e7c1c21c20b065aa0d1abeada05c4eb5b0237f"
    },
    "name": "instapic-klein.safetensors",
    "strength": 0.3,
    "trigger": "Instapic_Klein",
    "profiles": ["quality", "balanced"],
    "modes": ["text", "reference", "dual"]
  }
]
```

Pro samostatné porovnání této LoRA nastav `ENABLE_SNAPSHOT=0`. Jinak extras workflow aplikuje **všechny** LoRA povolené pro jeho profil/režim postupně. Identity workflow zůstává pouze s Majou.

Pro Civitai použij `source` jako číselné **modelVersionId** (například `"123456"` pouze jako ukázka formátu), URL `https://civitai.com/api/download/models/MODEL_VERSION_ID` nebo stránku s `?modelVersionId=...`. Samotné ID modelu nestačí. Přidej `CIVITAI_TOKEN` do secrets/env. Vybere se jednoznačný primary model soubor v safetensors formátu; nejasný výběr skončí chybou. Podporované source URL jsou pouze Hugging Face a Civitai.

Povolená pole: `source`, `name`, `strength`, `trigger`, `profiles`, `modes`. `profiles` je podmnožina `quality,balanced,fast`; `modes` podmnožina `text,reference,dual`. Bez těchto omezení se LoRA použije v extras workflow všech profilů/režimů. U jmen se speciálními znaky nastav jednoduché lokální `name`.

`ENABLE_CONSISTENCY=1` navíc stáhne připnutou dx8152 Consistency V2. Aplikuje se pouze ve `fast` referenčních extras workflow s výchozí silou 0.35 (`CONSISTENCY_STRENGTH`). Autor uvádí zlepšení při editaci; kombinace s Majou není vizuálně ověřená. Ve výchozím nastavení je vypnutá.

Soubor `amateurSnapshotPhotoSTYLE_v160FINALFINAL.safetensors` v tvém repozitáři **nepoužívej s Kleinem**: jeho metadata deklarují FLUX.1 a model `flux-1-dev/lora`. Startup takovou známou nekompatibilitu odmítne. Chybějící metadata však sama o sobě kompatibilitu nepotvrzují; finální kontrola vrstev nastane až v ComfyUI.

## Ověření na GPU

V terminálu Podu po startu:

```bash
python /opt/maja/app/smoke.py --profile fast --runs 2
python /opt/maja/app/smoke.py --profile quality --runs 2
python /opt/maja/app/smoke.py --profile quality --reference /workspace/input/tvoje-scena.png --extras --runs 2
```

Skript opravdu generuje obrázky, kontroluje dokončení a uloží JSON report do `/workspace/output`. Uvádí čas včetně pollingu a vzorkovanou spotřebu VRAM; první průchod může načítat modely. Porovnávej i druhý průchod. Při timeoutu zkontroluj frontu, úloha může dál běžet. V logu nesmí být neaplikované klíče Maja LoRA, NaN ani CUDA chyby. Následuje vizuální kontrola kamery, pózy, podoby a pleti; úspěšný HTTP požadavek realismus neprokazuje.

## Soubory a změny

- `/workspace/models`: váhy; zůstanou při použití persistent volume.
- `/workspace/output`: fotky a benchmark reporty.
- `/workspace/input`: reference.
- `/workspace/user`: ComfyUI nastavení a uživatelské workflow.
- `/workspace/user/default/workflows/maja-generated`: generované workflow, při startu se obnovují. Vlastní úpravy ukládej mimo tento podadresář.
- `/workspace/workflows/api`: API verze grafů.
- `/workspace/resolved-assets.json`: přesné revize/hash a nastavení, bez tokenů.

Závislosti ani ComfyUI neaktualizuj uvnitř běžícího kontejneru. Pro upgrade vytvoř nový image a zopakuj CPU a GPU smoke test. Veškerá východiska a kandidáti jsou v [průzkumu](docs/RESEARCH.md).
