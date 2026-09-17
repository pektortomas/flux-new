# Výběr konfigurace a zjištění

Průzkum: 17. 9. 2026. Priorita: uvěřitelná mobilní fotografie a identita Maji na RTX 5090, omezení ruční instalace a nejasných kombinací. „Ověřeno ve zdroji“ zde znamená dokumentaci, metadata či kód. Neznamená to, že jsme danou kombinaci vizuálně otestovali na GPU.

## 1. Přesný model a jeho nastavení

Klein 9B není FLUX.1 ani velký FLUX.2 dev. Má samostatný Qwen3 8B textový enkodér a FLUX.2 VAE. LoRA pro FLUX.1, Klein 4B nebo velký FLUX.2 dev nejsou automaticky přenosné. Podporuje generování i editaci s referencemi. Náš loader používá `CLIPLoader(type=flux2)` a `LoraLoaderModelOnly`, nikoli FLUX.1 dual-CLIP/T5 sestavu.

[Oficiální modelový přehled a inference kód BFL](https://github.com/black-forest-labs/flux2), [ComfyUI návod](https://docs.comfy.org/tutorials/flux/flux-2-klein).

Base je undistilled; distilled má odlišný rozpočet kroků a guidance. BFL příklad base používá 50 kroků a guidance 4.0; distilled používá 4 a 1.0. Tyto hodnoty přebíráme pro `quality` a `fast`. ComfyUI šablona base používá 20 kroků, CFG 5, Euler a `Flux2Scheduler`; z ní vychází `balanced`. Jde o dvě doložené výchozí konfigurace, nikoli tvrzení, že 50 kroků vyhraje v každé scéně.

[BFL base příklad](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B), [BFL distilled příklad](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), [připnutá ComfyUI base šablona](https://github.com/Comfy-Org/workflow_templates/blob/840a0ab4c1e0994cf4c549b02752a2d875c2bcb0/templates/image_flux2_klein_image_edit_9b_base.json).

Base/distilled mohou sdílet architekturu adaptérů, ale metadata Maji uvádějí jen `flux2_klein_9b`, nikoli jednoznačně přesný tréninkový checkpoint. Zda má Maja stejnou kvalitu na obou, musí potvrdit srovnání. Proto nelze slibovat, že přenos mezi oběma variantami zachová stejnou tvář nebo kompozici.

## 2. Co skutečně řeší nepřirozené úhly

Realism LoRA mění naučený vzhled, ale neurčuje přesnou polohu kamery. Hlavní cesta pro tento problém je skutečná kompoziční reference a Klein image editing. Používáme `VAEEncode → ReferenceLatent` pro pozitivní i negativní větev a nativní Flux2 scheduler. Se dvěma referencemi je první scéna/póza a druhá identita.

Tento postup kopíruje strukturu oficiálních referenčních grafů. Není to tvrdé geometrické omezení jako explicitní pose/depth conditioning. Silná reference může přenášet i původní vzhled nebo obličej, proto je nutné kontrolovat podobu. Nevkládáme FLUX.1 ControlNet jen proto, že v názvu obsahuje „Flux“.

[Oficiální příklady jednoho a více referenčních vstupů](https://github.com/Comfy-Org/workflow_templates/blob/840a0ab4c1e0994cf4c549b02752a2d875c2bcb0/templates/image_flux2_klein_image_edit_9b_distilled.json), [implementace ReferenceLatent](https://github.com/Comfy-Org/ComfyUI/blob/ee71d5c4993f29086b27fde1629a945ae48425bf/comfy_extras/nodes_edit_model.py).

Prompt popisuje fotografa, výšku kamery, vzdálenost, výřez, činnost a existující světlo. Nežádá současně širokoúhlé selfie, teleobjektivový portrét a filmové světlo. To je návrh promptu pro tento konkrétní účel, nikoli fyzikální záruka modelu.

## 3. LoRA: konkrétní důkazy a rozhodnutí

| Kandidát | Zjištění | Rozhodnutí |
|---|---|---|
| TommyPekk/Maja | Stažena celá, SHA-256 ověřeno; hlavička `flux2_klein_9b`; AI Toolkit 0.8.2, step 3000; LoRA projekce hidden size 4096 | Povinná, trigger `maja` potvrzený uživatelem, síla 1.0 jako start |
| SmartphoneSnapshotPhotoReality v9 v témže repu | Hlavička Klein 9B; trigger `amtr snapshot photo` doložen metadaty; jméno označuje base 9B | Stažena při startu, pouze v samostatných base extras workflow, start 0.35 |
| V3_flux_klein v témže repu | Hlavička Klein 9B; output `fluxtrait`; tag `Instapic_Klein` | Připraven příklad env jako alternativa, nikoli druhá automaticky vrstvená stylová LoRA |
| amateurSnapshotPhotoSTYLE_v160FINALFINAL | Hlavička `flux1`, architektura `flux-1-dev/lora`, text-encoder LoRA klíče | Vyloučena pro nekompatibilitu |
| dx8152 Consistency V2 | Autor označuje Klein 9B image-to-image; popisuje opravu barev, přesycení a přehnaných detailů ve V2 | Volitelná `ENABLE_CONSISTENCY=1`, pouze fast reference/dual, start 0.35 |
| SOLRICKS Realistic Detail | Model card uvádí Klein 9B, trigger `srx_detail`, doporučení 0.6–0.8; sám upozorňuje na přílišnou pleťovou texturu při vysoké síle | Bez automatického zařazení: zde primárně řešíme kompozici a tento styl může být kontraproduktivní |

Metadata uživatelova repozitáře jsme četli přímo z hlaviček safetensors (včetně tensor shapes), nikoli z názvu samotného. Jméno „best“ u jiného Maja checkpointu není důvod nahradit soubor, který uživatel výslovně určil. Originální autorství/licence přenahraných stylových souborů není tímto ověřeno; u těchto souborů dokládáme technická metadata.

[Připnutý repozitář s Majou a styly](https://huggingface.co/TommyPekk/maja_v2/tree/b7e7c1c21c20b065aa0d1abeada05c4eb5b0237f), [dx8152 model card](https://huggingface.co/dx8152/Flux2-Klein-9B-Consistency), [SOLRICKS model card](https://huggingface.co/SOLRICKS/Flux2-Klein-9B-Realistic-Detail).

U SOLRICKS navíc text model card jmenuje jiný soubor než skutečný seznam souborů. To ukazuje, proč downloader kontroluje metadata konkrétního souboru místo skládání odhadnutého URL. Síly 0.35 a 0.3 v tomto projektu jsou naše počáteční hodnoty pro kombinování s identitou; nejsou výsledkem měření ani univerzálním doporučením autora.

Civitai kandidáti pro následné ověření: [Phone Photography](https://civitai.com/models/2537408/phone-photography-2000-2025-klein-9b), [Ultra-Real Klein 9B](https://civitai.com/models/2462105/ultra-real-klein-9b), [Realism Engine Klein](https://civitai.com/models/2374977/realism-engine-klein). Původní stránky/API byly v tomto prostředí nedostupné (403 / chyba otevření); nebyla ověřena konkrétní verze, licence, soubor ani kvalita. Proto je nepřidáváme do locku ani neoznačujeme za nejlepší. Uživatel může dodat verzi a autorizovaný Civitai token při deployi.

Neexistuje zde důkaz, že tři realism LoRA dohromady budou lepší než jedna. Projekt umožňuje stack, ale výchozí base extras má jen Maju a jeden fotografický styl. Referenční identity workflow je srovnávací základ.

## 4. RTX 5090, paměť a výkon

RTX 5090 je Blackwell/sm_120. PyTorch podpora vyžaduje CUDA 12.8+ sestavení. Používáme ověřeně existující oficiální `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`, linux/amd64, Python 3.12; digest je v Dockerfile. Preflight provede skutečné BF16 CUDA násobení ještě před stahováním.

[PyTorch maintainer o Blackwell podpoře](https://discuss.pytorch.org/t/pytorch-support-for-nvidia-rtx-5090/221612), [oficiální Docker tag](https://hub.docker.com/layers/pytorch/pytorch/2.10.0-cuda12.8-cudnn9-runtime/images).

FP8 mixed enkodér a automatický offload jsou kompromis pro 32 GB. BF16 generátor má ~18.16 GB na disku, FP8 base ~9.57 GB, FP8 distilled ~9.43 GB, FP8 Qwen ~8.66 GB. Součet velikostí souborů **není** peak VRAM: reference, CFG, aktivace a LoRA patche vyžadují další paměť. Úsporu kvantizace ani BF16 nepřekládáme do nepodloženého příslibu kvality.

Výchozí: PyTorch SDPA, batch 1, přibližně 1 MP, 2 GB rezerva, vypnuté živé latent previews a compiler. Bez dodatečně kompilovaného SageAttention/FlashAttention, bez TeaCache a aproximací denoisingu. To omezuje počet proměnných při prvním spuštění; vyšší rychlost lze posoudit až po GPU benchmarku. Nový malý VAE decoder není použit, aby baseline používal standardní dekodér.

Neuvádíme odhad sekund za snímek jako naměřený výsledek. Marketingové hodnoty pro 4B nebo jiný server nelze přenést na 9B s Majou a více referencemi. Pro kontrolu máme `app/smoke.py`, dva průchody a záznam skutečného času/paměti. Při porovnání rozdílných checkpointů ani stejný seed nezaručí stejnou scénu.

## 5. RunPod a reproducibilita

RunPod může spustit vlastní container z registry a mountovat persistent volume do `/workspace`. Entrypoint image se nesmí v template přepsat příkazem `sleep infinity`. Pro přenos modelů mezi novými Pody použij network volume; container disk není trvalé úložiště. Docker Compose není potřeba ani součástí dodávky.

[RunPod custom template](https://docs.runpod.io/pods/templates/create-custom-template), [správa Podů](https://docs.runpod.io/pods/manage-pods), [typy nasazení](https://docs.runpod.io/pods/overview).

Připnutý ComfyUI commit, Docker digest, Python lock a HF revize omezují změny mezi starty. SHA-256 kontrola váhy detekuje neúplný či zaměněný download. Extra HF URL s `main` se při startu vyřeší na konkrétní commit, který se uloží v `resolved-assets.json`; pro opakovatelnost použij tento commit v env. Civitai se zadává konkrétní modelVersionId a ověřuje se file hash.

[Hugging Face revision koncept](https://huggingface.co/docs/huggingface_hub/guides/download), [Civitai API](https://github.com/civitai/civitai/wiki/REST-API-Reference).

## 6. Co zbývá pro tvrzení „ověřeno na 5090“

Úspěšný Linux Docker build včetně CPU/schema testů, přístup k gated BFL vahám, start na skutečné RTX 5090, generování text/reference/dual, kontrola ComfyUI logu na neaplikované LoRA vrstvy a vizuální porovnání Maja vs Maja+snapshot na stejných scénách. Až potom lze vybrat nejlepší výchozí sílu a uvést reálnou rychlost. Žádný z těchto GPU výsledků zatím nepředstíráme.
