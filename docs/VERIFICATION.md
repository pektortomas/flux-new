# Ověření — 17. 9. 2026

## Provedeno lokálně

- Vyhledání a porovnání BFL model cards, oficiálních ComfyUI šablon a implementací uzlů.
- Ověření existence Docker tagu a linux/amd64 digestu přes Docker registry metadata; Python 3.12 potvrzen v image config historii.
- Vyřešení 90 Python závislostí pro Linux x86_64/Python 3.12, Torch 2.10.0+cu128; výsledek `requirements.lock`.
- 15 automatických testů prošlo: povinná Maja ve všech grafech, base/distilled nastavení, reference v obou conditioning větvích, propojení UI/API grafů, scope stylových LoRA, ochrana uživatelských workflow, duplicitní jména, URL a credential scope, nekompatibilní FLUX.1 metadata, neúplný safetensors, hash mismatch, navázání downloadu i server ignorující Range.
- Python syntax compilation a shell syntax check prošly.
- Skutečně stažený `Maja.safetensors`: **165704368 bytes**, SHA-256 `98eae432ea407880537a33f82df8dbb6cf153c84baf3c389103795f1331274da`. Hlavička potvrzuje `flux2_klein_9b`.
- Runtime HF metadata resolver vyřešil tento soubor na commit `b7e7c1c21c20b065aa0d1abeada05c4eb5b0237f` a stejný SHA-256.
- Přečteny hlavičky dalších LoRA ve stejném repozitáři; odhalena nekompatibilní FLUX.1 varianta.
- Ukázkové UI/API JSON grafy v `workflows/` vygenerovány ze stejného kódu jako při startu.

## Připraveno jako povinná součást buildu, zatím nespuštěno

- `pip check` uvnitř linux/amd64 kontejneru.
- CPU start skutečného ComfyUI.
- `validate_comfy.py`: 18 kombinací grafů proti živému `/object_info` přesných instalovaných uzlů. Používá pouze placeholdery pro seznamy souborů; váhy se nenačítají a neprobíhá generování.

## Zatím neověřeno

- Úspěšné sestavení a publikování Docker image. Lokálně neběží Docker daemon a zbývá přibližně 4 GB disku; samotný komprimovaný base image má ~4.43 GB.
- Autentizované stahování gated BFL vah: vyžaduje uživatelův HF token a přijaté podmínky přístupu. V tomto prostředí žádný takový token nebyl použit.
- Autentizovaný Civitai download: veřejné pokusy o získání metadat konkrétních kandidátů vrátily 403. API výběr a downloader mají lokální testy s testovacími odpověďmi; to není ověření živého Civitai účtu.
- CUDA start tohoto kontejneru na RTX 5090, reálná VRAM, latence, loading všech LoRA vrstev, text/reference/dual inference a vizuální kvalita.
- Optimální síla snapshot LoRA a podobnost Maji na base vs distilled.

## Dokončení na RunPodu

1. Úspěšný image build v Actions.
2. Nastavit HF token a spustit Pod podle README.
3. Spustit `smoke.py` pro oba výchozí profily, nejdřív text, potom reference/dual.
4. Zkontrolovat log bez neaplikovaných LoRA klíčů, NaN a CUDA chyb.
5. Vyhodnotit alespoň dvě scény × tři seedy pro identity vs snapshot. Hodnotit polohu kamery, přirozenost pózy, podobu a konzistenci osvětlení samostatně.

Teprve po těchto bodech lze tuto konkrétní image nazvat ověřenou na RTX 5090. Žádné naměřené GPU výsledky zatím nejsou součástí dodávky.
