#!/usr/bin/env python3
"""Boltz-2 / Boltz-CP protein-metabolite co-folding campaign runner.

Pipeline phases (each is a subcommand, or run the whole thing with `all`):
    subset  - carve a size-balanced subset out of the full sequence/metabolite TSVs
    setup   - build the workdir tree, fetch repos, create venvs, download weights
    msa     - pull the ColabFold .a3m MSAs from the LiteFold HF dataset
    plan    - turn every (protein, metabolite) pair into a Boltz YAML + batched queue
    run     - execute batches (locally across GPUs, or a single batch for SLURM)
    submit  - emit + submit SLURM array jobs (one single-GPU task per batch)
    push    - upload the organized outputs to a HF dataset

Only Python's standard library is needed for plan/run/submit; huggingface_hub is
imported lazily inside setup/msa/push so worker tasks stay dependency-free.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BOLTZ_CP_GIT = "https://github.com/NVIDIA-BioNeMo/boltz-cp.git"
MSA_DATASET = "LiteFold/human-proteome-wide-msa"
DEFAULT_ORG = "LiteFold"
DEFAULT_DATASET = "cofolding-test"

# Length tiers (residues). supra = too large for a single GPU -> Boltz-CP.
TIER_BOUNDS = {"small": 250, "mid": 700, "big": 1500}
TIER_ORDER = ["small", "mid", "big", "supra"]

# Per-GPU inference batch sizes by tier (pairs folded in one Boltz process).
TIER_BATCH = {"small": 8, "mid": 4, "big": 2, "supra": 1}


# --------------------------------------------------------------------------- #
# Colored logging
# --------------------------------------------------------------------------- #
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


_COLOR = _supports_color()
_C = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}


def _paint(text: str, *styles: str) -> str:
    if not _COLOR:
        return text
    return "".join(_C[s] for s in styles) + text + _C["reset"]


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _emit(tag: str, color: str, msg: str) -> None:
    print(f"{_paint(_ts(), 'dim')} {_paint(tag, color, 'bold')} {msg}", flush=True)


def info(msg: str) -> None: _emit("INFO", "cyan", msg)
def ok(msg: str) -> None: _emit("OK  ", "green", msg)
def warn(msg: str) -> None: _emit("WARN", "yellow", msg)
def err(msg: str) -> None: _emit("ERR ", "red", msg)


def section(title: str) -> None:
    line = "-" * max(8, 60 - len(title))
    print(f"\n{_paint('==', 'magenta', 'bold')} {_paint(title, 'magenta', 'bold')} {_paint(line, 'magenta')}", flush=True)


def die(msg: str) -> "NoReturn":  # type: ignore[valid-type]
    err(msg)
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_tsv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
        return (reader.fieldnames or []), rows


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text())


def clean_seq(raw: str) -> str:
    return re.sub(r"\s+", "", (raw or "").strip()).upper()


def tier_of(length: int) -> str:
    if length <= TIER_BOUNDS["small"]:
        return "small"
    if length <= TIER_BOUNDS["mid"]:
        return "mid"
    if length <= TIER_BOUNDS["big"]:
        return "big"
    return "supra"


def parse_status(spec: str) -> set[str] | None:
    """core|non|all|comma-list -> set of statuses to keep (None = keep all)."""
    spec = (spec or "all").strip().lower()
    if spec == "all":
        return None
    return {s.strip() for s in spec.split(",") if s.strip()}


def norm_met_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"\([^)]*\)", "", s)
    return re.sub(r"\s+", "", s)


def parse_names(spec: str | None) -> list[str] | None:
    if spec is None:
        return None
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return parts or None


def load_metabolite_catalog(met_rows: list[dict]) -> list[dict]:
    catalog = []
    for idx, row in enumerate(met_rows):
        name = (row.get("updated_name") or "").strip()
        smiles = (row.get("updated_smiles") or "").strip()
        st = (row.get("status") or "").strip()
        if not name or not smiles:
            continue
        catalog.append({"id": idx, "name": name, "smiles": smiles, "status": st})
    return catalog


def select_named_metabolites(catalog: list[dict], names: list[str],
                             wanted_status: set[str] | None) -> list[dict]:
    selected, used = [], set()
    for query in names:
        qn = norm_met_name(query)
        hits = [m for m in catalog if norm_met_name(m["name"]) == qn]
        if wanted_status is not None:
            hits = [m for m in hits if m["status"] in wanted_status]
        if not hits:
            close = [m["name"] for m in catalog
                     if qn in norm_met_name(m["name"]) or norm_met_name(m["name"]).startswith(qn)]
            hint = f" (close: {', '.join(close[:8])})" if close else ""
            die(f"no metabolite matching {query!r}{hint}")
        for m in hits:
            if m["id"] not in used:
                selected.append(m)
                used.add(m["id"])
    return selected


def heavy_atoms(smiles: str) -> int:
    """Rough non-hydrogen atom count, good enough for size-ordering ligands."""
    bracket = re.findall(r"\[([^\]]+)\]", smiles)
    plain = re.sub(r"\[[^\]]+\]", "", smiles)
    total = 0
    for token in bracket:
        m = re.match(r"(?:\d+)?([A-Z][a-z]?|[bcnops])", token)
        if m and m.group(1) != "H":
            total += 1
    return total + len(re.findall(r"Br|Cl|[A-Z][a-z]?|[bcnops]", plain))


def spread(items: list, count: int) -> list:
    """Evenly sample `count` items across a sorted list (keeps the variety)."""
    if count <= 0 or count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    idx = dict.fromkeys(round(i * (len(items) - 1) / (count - 1)) for i in range(count))
    return [items[i] for i in idx]


def slug(text: str, limit: int = 40) -> str:
    return (re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-") or "x")[:limit]


def gpu_ids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


def run_cmd(cmd: list[str], *, env: dict | None = None, log: Path | None = None) -> int:
    info(_paint("$ " + " ".join(cmd), "dim"))
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as fh:
            fh.write(f"\n[{now_iso()}] $ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        return proc.returncode
    return subprocess.run(cmd, env=env).returncode


# --------------------------------------------------------------------------- #
# Workdir layout
# --------------------------------------------------------------------------- #
class Work:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.repos = self.root / "repos"
        self.checkpoints = self.root / "checkpoints"
        self.msa_cache = self.root / "msa_cache"
        self.sequences = self.root / "sequences.tsv"
        self.metabolites = self.root / "metabolites.tsv"
        self.boltz_run_cache = self.root / "boltz_run_cache"
        self.outputs = self.root / "outputs"
        self.configs = self.root / "configs"
        self.queue = self.root / "queue"
        self.env = self.root / "env"
        self.logs = self.root / "logs"
        self.slurm = self.root / "slurm"

    def make_tree(self) -> None:
        for d in (self.repos, self.checkpoints, self.msa_cache, self.boltz_run_cache,
                  self.outputs, self.configs, self.queue, self.env, self.logs, self.slurm):
            d.mkdir(parents=True, exist_ok=True)

    # Tool locations inside the workdir.
    @property
    def boltz_bin(self) -> Path: return self.env / "boltz" / "bin" / "boltz"
    @property
    def boltz_py(self) -> Path: return self.env / "boltz" / "bin" / "python"
    @property
    def boltz_cp_py(self) -> Path: return self.env / "boltz-cp" / "bin" / "python"
    @property
    def boltz_cp_repo(self) -> Path: return self.repos / "boltz-cp"
    @property
    def boltz_cp_main(self) -> Path: return self.boltz_cp_repo / "src" / "boltz" / "distributed" / "main.py"

    def cp_available(self) -> bool:
        return self.boltz_cp_py.exists() and self.boltz_cp_main.exists()


# --------------------------------------------------------------------------- #
# subset: build size-balanced test inputs
# --------------------------------------------------------------------------- #
def select_proteins(rows: list[dict], per_tier: dict[str, int]) -> list[dict]:
    buckets: dict[str, list[dict]] = {t: [] for t in TIER_ORDER}
    for row in rows:
        seq = clean_seq(row.get("seq", ""))
        acc = (row.get("acc") or "").strip()
        if not acc or not seq or not re.fullmatch(r"[A-Z]+", seq):
            continue
        buckets[tier_of(len(seq))].append(row)
    chosen: list[dict] = []
    for tier in TIER_ORDER:
        ranked = sorted(buckets[tier], key=lambda r: (len(clean_seq(r["seq"])), r["acc"]))
        picked = spread(ranked, per_tier.get(tier, 0))
        if per_tier.get(tier, 0) and not picked:
            warn(f"no proteins available in tier '{tier}'")
        chosen.extend(picked)
    return chosen


def select_metabolites(rows: list[dict], count: int, status: str,
                       names: list[str] | None = None) -> list[dict]:
    catalog = load_metabolite_catalog(rows)
    wanted = parse_status(status)
    if names:
        picked = select_named_metabolites(catalog, names, wanted)
        by_id = {m["id"]: rows[m["id"]] for m in picked}
        return [by_id[m["id"]] for m in picked]
    pool = [r for r in rows if (r.get("status") or "").strip() == status
            and (r.get("updated_name") or "").strip() and (r.get("updated_smiles") or "").strip()]
    ranked = sorted(pool, key=lambda r: (heavy_atoms(r["updated_smiles"]), r["updated_name"]))
    return spread(ranked, count)


def cmd_subset(args: argparse.Namespace) -> None:
    section("Building test subset")
    seq_fields, seq_rows = read_tsv(Path(args.sequences))
    met_fields, met_rows = read_tsv(Path(args.metabolites))

    per_tier = {"small": args.small, "mid": args.mid, "big": args.big, "supra": args.supra}
    proteins = select_proteins(seq_rows, per_tier)
    metabolites = select_metabolites(met_rows, args.metabolites_count, args.status,
                                     parse_names(args.names))

    write_tsv(Path(args.out_sequences), seq_fields, proteins)
    write_tsv(Path(args.out_metabolites), met_fields, metabolites)

    by_tier = {t: sum(1 for p in proteins if tier_of(len(clean_seq(p["seq"]))) == t) for t in TIER_ORDER}
    ok(f"proteins: {len(proteins)}  ({', '.join(f'{t}={by_tier[t]}' for t in TIER_ORDER)})")
    for p in proteins:
        info(f"  {p['acc']:<12} {p.get('gene',''):<12} len={len(clean_seq(p['seq'])):>5}  [{tier_of(len(clean_seq(p['seq'])))}]")
    ok(f"metabolites ({args.status}): {len(metabolites)}")
    for i, m in enumerate(metabolites):
        info(f"  id={i}  {m['updated_name']:<28} heavy~{heavy_atoms(m['updated_smiles']):>3}")
    ok(f"wrote {args.out_sequences} and {args.out_metabolites}")


# --------------------------------------------------------------------------- #
# setup: repos, venvs, weights
# --------------------------------------------------------------------------- #
def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def _make_venv(venv: Path, log: Path) -> Path:
    py = _venv_python(venv)
    if not py.exists():
        run_cmd([sys.executable, "-m", "venv", str(venv)], log=log)
    run_cmd([str(py), "-m", "pip", "install", "-q", "-U", "pip", "setuptools", "wheel"], log=log)
    return py


def setup_boltz(work: Work, source: str | None) -> None:
    section("Installing Boltz-2 (serial)")
    if work.boltz_bin.exists():
        ok("boltz already installed")
        return
    py = _make_venv(work.env / "boltz", work.logs / "setup_boltz.log")
    target = f"{source}[cuda]" if source else "boltz[cuda]"
    rc = run_cmd([str(py), "-m", "pip", "install", "-q", target], log=work.logs / "setup_boltz.log")
    if rc or not work.boltz_bin.exists():
        die("Boltz install failed; see logs/setup_boltz.log")
    ok("boltz installed")


def setup_boltz_cp(work: Work, source: str | None, force_serial: bool) -> None:
    section("Installing Boltz-CP (context-parallel)")
    if force_serial:
        warn("--force-serial set: skipping Boltz-CP (supra tier will run on a single GPU)")
        return
    log = work.logs / "setup_boltz_cp.log"
    if not work.boltz_cp_main.exists():
        src = Path(source) if source else None
        if src and src.exists():
            info(f"copying Boltz-CP from {src}")
            shutil.copytree(src, work.boltz_cp_repo, dirs_exist_ok=True)
        else:
            run_cmd(["git", "clone", "--depth", "1", BOLTZ_CP_GIT, str(work.boltz_cp_repo)], log=log)
    if not work.boltz_cp_main.exists():
        warn("Boltz-CP checkout missing; supra tier will fall back to serial")
        return
    if work.boltz_cp_py.exists():
        ok("boltz-cp venv already present")
        return
    py = _make_venv(work.env / "boltz-cp", log)
    # torch must be present before the no-build-isolation editable install.
    run_cmd([str(py), "-m", "pip", "install", "-q", "torch"], log=log)
    rc = run_cmd([str(py), "-m", "pip", "install", "-q", "--no-build-isolation",
                  "-e", f"{work.boltz_cp_repo}[cuda]"], log=log)
    if rc:
        warn("Boltz-CP [cuda] install failed; retrying without the cuda extra")
        rc = run_cmd([str(py), "-m", "pip", "install", "-q", "--no-build-isolation",
                      "-e", str(work.boltz_cp_repo)], log=log)
    if rc:
        warn("Boltz-CP install failed; supra tier will fall back to serial. See logs/setup_boltz_cp.log")
        shutil.rmtree(work.env / "boltz-cp", ignore_errors=True)
    else:
        ok("boltz-cp installed")


def download_weights(work: Work) -> None:
    section("Downloading Boltz-2 weights + CCD molecules")
    marker = work.checkpoints / "boltz2_conf.ckpt"
    if marker.exists() and (work.checkpoints / "mols").exists():
        ok("weights already present")
        return
    snippet = (
        "from pathlib import Path;"
        "from boltz.main import download_boltz2;"
        f"download_boltz2(Path(r'{work.checkpoints}'))"
    )
    rc = run_cmd([str(work.boltz_py), "-c", snippet], log=work.logs / "weights.log")
    if rc or not marker.exists():
        die("weight download failed; see logs/weights.log")
    ok("weights ready in checkpoints/")


def cmd_setup(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    work.make_tree()
    setup_boltz(work, args.boltz_src)
    setup_boltz_cp(work, args.boltz_cp_src, args.force_serial)
    download_weights(work)
    ok(f"setup complete: {work.root}")


# --------------------------------------------------------------------------- #
# msa: download ColabFold a3m MSAs from the HF dataset
# --------------------------------------------------------------------------- #
def _enable_hf_transfer() -> None:
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    except Exception:
        os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)


def download_msa_full(work: Work, token: str | None) -> None:
    section("Downloading full ColabFold a3m mirror (~105 GB)")
    from huggingface_hub import snapshot_download
    _enable_hf_transfer()
    snapshot_download(
        repo_id=MSA_DATASET, repo_type="dataset", local_dir=str(work.msa_cache),
        allow_patterns=["a3m/**", "manifest.tsv", "manifest.jsonl"],
        max_workers=16, token=token,
    )
    ok("a3m mirror downloaded")


def download_msa_targeted(work: Work, accessions: list[str], token: str | None) -> None:
    section(f"Downloading a3m MSAs for {len(accessions)} selected proteins")
    from huggingface_hub import hf_hub_download
    _enable_hf_transfer()
    manifest = hf_hub_download(
        repo_id=MSA_DATASET, repo_type="dataset", filename="manifest.tsv",
        local_dir=str(work.msa_cache), token=token,
    )
    _, rows = read_tsv(Path(manifest))
    path_by_acc = {r["acc"]: r["a3m_path"] for r in rows if r.get("a3m_path")}
    missing = 0
    for acc in accessions:
        rel = path_by_acc.get(acc)
        if not rel:
            warn(f"no a3m listed for {acc}")
            missing += 1
            continue
        hf_hub_download(repo_id=MSA_DATASET, repo_type="dataset", filename=rel,
                        local_dir=str(work.msa_cache), token=token)
    ok(f"downloaded {len(accessions) - missing} a3m files ({missing} missing)")


def cmd_msa(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    work.msa_cache.mkdir(parents=True, exist_ok=True)
    if args.mode == "full":
        download_msa_full(work, args.hf_token or os.environ.get("HF_TOKEN"))
    else:
        _, rows = read_tsv(Path(args.sequences))
        accs = [(r.get("acc") or "").strip() for r in rows if (r.get("acc") or "").strip()]
        download_msa_targeted(work, accs, args.hf_token or os.environ.get("HF_TOKEN"))


def find_a3m(work: Work, acc: str) -> Path | None:
    direct = work.msa_cache / "a3m"
    if direct.exists():
        hits = list(direct.glob(f"*/{acc}.a3m")) + list(direct.glob(f"{acc}.a3m"))
        if hits:
            return hits[0]
    hits = list(work.msa_cache.rglob(f"{acc}.a3m"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# plan: YAML configs + batched queue
# --------------------------------------------------------------------------- #
def write_yaml(path: Path, sequence: str, smiles: str, msa_path: str) -> None:
    q = lambda s: json.dumps(s, ensure_ascii=False)
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {q(sequence)}",
        f"      msa: {q(msa_path)}",
        "  - ligand:",
        "      id: B",
        f"      smiles: {q(smiles)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_plan(work: Work, batch_size_override: int | None, cp_size: int,
               force_serial: bool, allow_missing_msa: bool, status: str = "all",
               names: str | None = None) -> dict:
    section("Planning co-folding jobs")
    _, seq_rows = read_tsv(work.sequences)
    _, met_rows = read_tsv(work.metabolites)

    catalog = load_metabolite_catalog(met_rows)
    write_json(work.outputs / "metabolites_index.json", catalog)

    wanted = parse_status(status)
    name_list = parse_names(names)
    if name_list:
        metabolites = select_named_metabolites(catalog, name_list, wanted)
    elif wanted is None:
        metabolites = catalog
    else:
        metabolites = [m for m in catalog if m["status"] in wanted]
    if not metabolites:
        die(f"no metabolites match --status {status!r}"
            + (f" --names {names!r}" if names else ""))
    info(f"metabolites selected: {len(metabolites)}  "
         f"(status={status}, names={names or 'all'}, order={'given' if name_list else 'tsv'})")
    for m in metabolites:
        info(f"  id={m['id']:<4} {m['name']:<28} [{m['status']}]")

    proteins, skipped = [], []
    for row in seq_rows:
        acc = (row.get("acc") or "").strip()
        seq = clean_seq(row.get("seq", ""))
        if not acc or not seq or not re.fullmatch(r"[A-Z]+", seq):
            skipped.append({"acc": acc, "reason": "invalid sequence"})
            continue
        a3m = find_a3m(work, acc)
        if a3m is None and not allow_missing_msa:
            skipped.append({"acc": acc, "reason": "missing a3m"})
            warn(f"skipping {acc}: no a3m in msa_cache")
            continue
        proteins.append({
            "acc": acc, "gene": (row.get("gene") or "").strip(),
            "entry": (row.get("entry") or "").strip(), "seq": seq, "length": len(seq),
            "tier": tier_of(len(seq)),
            "msa": str(a3m) if a3m else str(work.msa_cache / "a3m" / f"{acc}.a3m"),
        })

    use_cp = work.cp_available() and not force_serial and len(gpu_ids()) >= cp_size
    if not use_cp and any(p["tier"] == "supra" for p in proteins):
        warn("Boltz-CP unavailable or <%d GPUs: supra proteins routed to serial (may OOM)" % cp_size)

    proteins_sorted = sorted(proteins, key=lambda x: (TIER_ORDER.index(x["tier"]), x["length"], x["acc"]))
    jobs: list[dict] = []
    for m in metabolites:
        for p in proteins_sorted:
            route = "cp" if (p["tier"] == "supra" and use_cp) else "serial"
            job_id = f"{slug(p['acc'], 20)}__m{m['id']:02d}"
            yaml_path = work.configs / p["tier"] / f"{job_id}.yaml"
            jobs.append({
                "job_id": job_id, "acc": p["acc"], "gene": p["gene"], "entry": p["entry"],
                "length": p["length"], "tier": p["tier"], "route": route,
                "metabolite_id": m["id"], "metabolite_name": m["name"],
                "smiles": m["smiles"], "status": m["status"], "msa": p["msa"],
                "input_yaml": str(yaml_path),
                "cif_out": str(work.outputs / p["acc"] / f"structure_with_metabolite_id_{m['id']}.cif"),
                "scores_out": str(work.outputs / p["acc"] / f"scores_with_metabolite_id_{m['id']}.json"),
                "_seq": p["seq"],
            })

    # One metabolite at a time, then by tier, so ATP can finish before GTP etc.
    batches: list[dict] = []
    seq_no = 0
    for m in metabolites:
        for tier in TIER_ORDER:
            tier_jobs = [j for j in jobs if j["metabolite_id"] == m["id"] and j["tier"] == tier]
            size = batch_size_override or TIER_BATCH[tier]
            if tier == "supra":
                size = 1
            for start in range(0, len(tier_jobs), size):
                seq_no += 1
                chunk = tier_jobs[start:start + size]
                batch_id = f"batch_{seq_no:05d}_{slug(m['name'], 16)}_{tier}"
                config_dir = work.configs / tier / batch_id
                config_dir.mkdir(parents=True, exist_ok=True)
                for j in chunk:
                    dest = config_dir / f"{j['job_id']}.yaml"
                    write_yaml(dest, j["_seq"], j["smiles"], j["msa"])
                    j["input_yaml"] = str(dest)
                    j["batch_id"] = batch_id
                batches.append({
                    "batch_id": batch_id, "tier": tier, "route": chunk[0]["route"],
                    "metabolite_id": m["id"], "metabolite_name": m["name"],
                    "resources": cp_size if chunk[0]["route"] == "cp" else 1,
                    "config_dir": str(config_dir),
                    "out_dir": str(work.boltz_run_cache / batch_id),
                    "job_ids": [j["job_id"] for j in chunk],
                })

    for j in jobs:
        j.pop("_seq", None)
    plan = {
        "created_at": now_iso(), "workdir": str(work.root),
        "proteins": len(proteins), "metabolites": len(metabolites),
        "metabolite_names": [m["name"] for m in metabolites],
        "jobs": len(jobs), "batches": len(batches), "skipped": len(skipped),
        "cp_size": cp_size, "use_cp": use_cp, "tier_bounds": TIER_BOUNDS,
        "status": status, "names": names,
    }
    write_json(work.queue / "jobs.json", jobs)
    write_json(work.queue / "batches.json", batches)
    write_json(work.queue / "skipped.json", skipped)
    write_json(work.queue / "plan.json", plan)

    by_tier = {t: sum(1 for b in batches if b["tier"] == t) for t in TIER_ORDER}
    ok(f"{len(jobs)} jobs -> {len(batches)} batches ({', '.join(f'{t}={by_tier[t]}' for t in TIER_ORDER)})")
    if skipped:
        warn(f"{len(skipped)} proteins skipped (see queue/skipped.json)")
    return plan


def cmd_plan(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    work.make_tree()
    shutil.copyfile(args.sequences, work.sequences)
    shutil.copyfile(args.metabolites, work.metabolites)
    build_plan(work, args.batch_size, args.cp_size, args.force_serial, args.allow_missing_msa,
               args.status, args.names)


# --------------------------------------------------------------------------- #
# run: execute batches
# --------------------------------------------------------------------------- #
def _boltz_common_flags(args) -> list[str]:
    return [
        "--recycling_steps", str(args.recycling_steps),
        "--sampling_steps", str(args.sampling_steps),
        "--diffusion_samples", str(args.diffusion_samples),
        "--max_msa_seqs", str(args.max_msa_seqs),
        "--output_format", "mmcif",
    ]


def batch_command(work: Work, batch: dict, args) -> list[str]:
    config_dir = batch["config_dir"]
    out_dir = batch["out_dir"]
    if batch["route"] == "cp":
        cp = int(batch["resources"])
        return [
            str(work.boltz_cp_py), "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={cp}", str(work.boltz_cp_main), "predict", config_dir,
            "--input_format", "config_files", "--out_dir", out_dir,
            "--cache", str(work.checkpoints), "--size_dp", "1", "--size_cp", str(cp),
            "--accelerator", "gpu", "--local_batch_size", "1",
        ] + _boltz_common_flags(args)
    return [
        str(work.boltz_bin), "predict", config_dir, "--out_dir", out_dir,
        "--cache", str(work.checkpoints), "--model", "boltz2",
        "--accelerator", "gpu", "--devices", "1", "--num_workers", str(args.num_workers),
    ] + _boltz_common_flags(args)


def _prediction_dir(work: Work, batch: dict, job_id: str) -> Path | None:
    root = Path(batch["out_dir"]) / f"boltz_results_{batch['batch_id']}"
    for sub in ("predictions", "predictions_dp0_cp0"):
        cand = root / sub / job_id
        if cand.is_dir():
            return cand
    return None


def collect_job(work: Work, batch: dict, job: dict) -> bool:
    pred = _prediction_dir(work, batch, job["job_id"])
    if pred is None:
        return False
    cifs = sorted(pred.glob(f"{job['job_id']}_model_*.cif"))
    if not cifs:
        return False
    best = next((c for c in cifs if c.name.endswith("_model_0.cif")), cifs[0])
    cif_out = Path(job["cif_out"])
    cif_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, cif_out)

    conf_path = pred / f"confidence_{job['job_id']}_model_0.json"
    confidence = load_json(conf_path) if conf_path.exists() else {}
    keys = ["confidence_score", "iptm", "ptm", "ligand_iptm", "protein_iptm",
            "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde"]
    scores = {
        "protein": {k: job[k] for k in ("acc", "gene", "entry", "length", "tier")},
        "metabolite": {"id": job["metabolite_id"], "name": job["metabolite_name"],
                       "smiles": job["smiles"], "status": job["status"]},
        "model": {"rank": 0, "route": batch["route"], "structure_file": cif_out.name},
        "metrics": {k: confidence.get(k) for k in keys},
        "boltz_confidence": confidence,
        "created_at": now_iso(),
    }
    write_json(Path(job["scores_out"]), scores)
    return True


def job_done(job: dict) -> bool:
    return Path(job["cif_out"]).exists() and Path(job["scores_out"]).exists()


def run_batch(work: Work, batch: dict, jobs_by_id: dict, args, gpus: list[int] | None) -> bool:
    jobs = [jobs_by_id[j] for j in batch["job_ids"]]
    if all(job_done(j) for j in jobs):
        ok(f"{batch['batch_id']}: already complete, skipping")
        return True

    env = os.environ.copy()
    env["BOLTZ_CACHE"] = str(work.checkpoints)
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    log = work.logs / f"{batch['batch_id']}.log"
    info(f"{batch['batch_id']}: route={batch['route']} jobs={len(jobs)} gpus={gpus if gpus is not None else 'inherited'}")
    rc = run_cmd(batch_command(work, batch, args), env=env, log=log)

    collected = sum(collect_job(work, batch, j) for j in jobs)
    if rc:
        warn(f"{batch['batch_id']}: boltz exit={rc}, collected {collected}/{len(jobs)} (see {log.name})")
    else:
        ok(f"{batch['batch_id']}: collected {collected}/{len(jobs)}")
    return rc == 0 and collected == len(jobs)


def orchestrate_local(work: Work, batches: list[dict], jobs_by_id: dict, args, gpus: list[int]) -> None:
    section(f"Running locally on GPUs {gpus}")
    serial = [b for b in batches if b["route"] == "serial"]
    cp = [b for b in batches if b["route"] == "cp"]

    # Serial batches: one per GPU, filled greedily.
    pending = list(serial)
    active: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    free = list(gpus)

    def launch(gpu: int, batch: dict):
        jobs = [jobs_by_id[j] for j in batch["job_ids"]]
        if all(job_done(j) for j in jobs):
            ok(f"{batch['batch_id']}: already complete, skipping")
            return None
        env = os.environ.copy()
        env["BOLTZ_CACHE"] = str(work.checkpoints)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = work.logs / f"{batch['batch_id']}.log"
        fh = log.open("a")
        fh.write(f"\n[{now_iso()}] $ {' '.join(batch_command(work, batch, args))}\n")
        fh.flush()
        info(f"START {batch['batch_id']} on GPU {gpu} ({len(batch['job_ids'])} jobs)")
        proc = subprocess.Popen(batch_command(work, batch, args), env=env, stdout=fh, stderr=subprocess.STDOUT)
        return (proc, batch, fh)

    while pending or active:
        while free and pending:
            gpu = free.pop(0)
            batch = pending.pop(0)
            handle = launch(gpu, batch)
            if handle is None:
                free.append(gpu)
            else:
                active[gpu] = handle
        for gpu, (proc, batch, fh) in list(active.items()):
            if proc.poll() is None:
                continue
            fh.close()
            jobs = [jobs_by_id[j] for j in batch["job_ids"]]
            collected = sum(collect_job(work, batch, j) for j in jobs)
            (ok if proc.returncode == 0 and collected == len(jobs) else warn)(
                f"END   {batch['batch_id']} exit={proc.returncode} collected={collected}/{len(jobs)}")
            del active[gpu]
            free.append(gpu)
        if active:
            time.sleep(args.poll_seconds)

    for batch in cp:  # context-parallel: one at a time, all GPUs.
        run_batch(work, batch, jobs_by_id, args, gpus)


def cmd_run(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    batches = load_json(work.queue / "batches.json")
    jobs_by_id = {j["job_id"]: j for j in load_json(work.queue / "jobs.json")}

    # Single-batch mode (SLURM task): inherit the task's allocated GPUs.
    if args.batch_id or args.batch_index is not None:
        if args.batch_id:
            batch = next((b for b in batches if b["batch_id"] == args.batch_id), None)
            if batch is None:
                die(f"unknown batch id: {args.batch_id}")
        else:
            if not 0 <= args.batch_index < len(batches):
                die(f"batch index out of range: {args.batch_index}")
            batch = batches[args.batch_index]
        run_batch(work, batch, jobs_by_id, args, None)
        return

    gpus = gpu_ids() if args.gpus == "auto" else [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        die("no GPUs found; pass --gpus or run on a GPU node")
    orchestrate_local(work, batches, jobs_by_id, args, gpus)
    report(work)


def report(work: Work) -> None:
    jobs = load_json(work.queue / "jobs.json")
    done = sum(1 for j in jobs if job_done(j))
    section("Summary")
    (ok if done == len(jobs) else warn)(f"{done}/{len(jobs)} pairs complete -> {work.outputs}")


# --------------------------------------------------------------------------- #
# submit: SLURM array jobs
# --------------------------------------------------------------------------- #
SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --requeue
#SBATCH --array=0-{last}{throttle}
#SBATCH --output={logs}/slurm-%A_%a.out
{account}
set -euo pipefail
BATCH_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "{list_file}")
echo "[$(date -u +%FT%TZ)] task $SLURM_ARRAY_TASK_ID -> $BATCH_ID"
exec "{bootstrap}" run --workdir "{workdir}" --batch-id "$BATCH_ID"
"""


def _write_array(work: Work, name: str, batch_ids: list[str], gpus: int, args) -> Path | None:
    if not batch_ids:
        return None
    list_file = work.slurm / f"{name}_batches.txt"
    list_file.write_text("\n".join(batch_ids) + "\n")
    throttle = f"%{args.slurm_throttle}" if args.slurm_throttle else ""
    account = f"#SBATCH --account={args.slurm_account}" if args.slurm_account else ""
    script = SLURM_TEMPLATE.format(
        job_name=f"cofold-{name}", qos=args.slurm_qos, gpus=gpus, cpus=args.slurm_cpus,
        time_limit=args.slurm_time, last=len(batch_ids) - 1, throttle=throttle,
        logs=str(work.logs), account=account, list_file=str(list_file),
        bootstrap=args.bootstrap, workdir=str(work.root),
    )
    path = work.slurm / f"submit_{name}.sbatch"
    path.write_text(script)
    ok(f"wrote {path} ({len(batch_ids)} tasks, gpu:{gpus})")
    return path


def cmd_submit(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    batches = load_json(work.queue / "batches.json")
    serial = [b["batch_id"] for b in batches if b["route"] == "serial"]
    cp = [b for b in batches if b["route"] == "cp"]
    cp_size = cp[0]["resources"] if cp else 4

    section("Generating SLURM array scripts")
    scripts = [
        _write_array(work, "serial", serial, 1, args),
        _write_array(work, "cp", [b["batch_id"] for b in cp], cp_size, args),
    ]
    if args.submit:
        for script in scripts:
            if script:
                run_cmd(["sbatch", str(script)])
    else:
        info("dry run: pass --submit to sbatch these scripts")


# --------------------------------------------------------------------------- #
# push: upload outputs to a HF dataset
# --------------------------------------------------------------------------- #
def cmd_push(args: argparse.Namespace) -> None:
    section("Pushing outputs to Hugging Face")
    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        die("no HF token: pass --hf-token or set HF_TOKEN")
    from huggingface_hub import HfApi
    work = Work(Path(args.workdir))
    repo_id = f"{args.org}/{args.dataset}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(work.outputs), path_in_repo="outputs",
                      repo_id=repo_id, repo_type="dataset")
    for extra in (work.sequences, work.metabolites, work.queue / "plan.json"):
        if extra.exists():
            api.upload_file(path_or_fileobj=str(extra), path_in_repo=extra.name,
                            repo_id=repo_id, repo_type="dataset")
    ok(f"pushed to https://huggingface.co/datasets/{repo_id}")


# --------------------------------------------------------------------------- #
# all: full pipeline
# --------------------------------------------------------------------------- #
def cmd_all(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    work.make_tree()
    if not args.skip_setup:
        setup_boltz(work, args.boltz_src)
        setup_boltz_cp(work, args.boltz_cp_src, args.force_serial)
        download_weights(work)
    if not args.skip_msa:
        if args.msa_mode == "full":
            download_msa_full(work, args.hf_token or os.environ.get("HF_TOKEN"))
        else:
            _, rows = read_tsv(Path(args.sequences))
            accs = [(r.get("acc") or "").strip() for r in rows if (r.get("acc") or "").strip()]
            download_msa_targeted(work, accs, args.hf_token or os.environ.get("HF_TOKEN"))
    shutil.copyfile(args.sequences, work.sequences)
    shutil.copyfile(args.metabolites, work.metabolites)
    build_plan(work, args.batch_size, args.cp_size, args.force_serial, allow_missing_msa=False,
               status=args.status, names=args.names)

    batches = load_json(work.queue / "batches.json")
    jobs_by_id = {j["job_id"]: j for j in load_json(work.queue / "jobs.json")}
    gpus = gpu_ids() if args.gpus == "auto" else [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        die("no GPUs found for local run")
    orchestrate_local(work, batches, jobs_by_id, args, gpus)
    report(work)
    if not args.skip_push and (args.hf_token or os.environ.get("HF_TOKEN")):
        cmd_push(args)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _add_met_select(p: argparse.ArgumentParser, status_default: str) -> None:
    p.add_argument("--status", default=status_default,
                   help="metabolites to fold: core | non | all | comma-list")
    p.add_argument("--names", default=None,
                   help="comma-separated metabolite names, run in this order "
                        "(ATP matches ATP(4-), not dATP). e.g. ATP,GTP,NAD+,NADH")


def _add_boltz_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recycling-steps", type=int, default=3, dest="recycling_steps")
    p.add_argument("--sampling-steps", type=int, default=200, dest="sampling_steps")
    p.add_argument("--diffusion-samples", type=int, default=1, dest="diffusion_samples")
    p.add_argument("--max-msa-seqs", type=int, default=8192, dest="max_msa_seqs")
    p.add_argument("--num-workers", type=int, default=2, dest="num_workers")
    p.add_argument("--poll-seconds", type=float, default=3.0, dest="poll_seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Boltz-2/Boltz-CP protein-metabolite co-folding campaign")
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("subset", help="carve a size-balanced subset from full TSVs")
    sp.add_argument("--sequences", required=True)
    sp.add_argument("--metabolites", required=True)
    sp.add_argument("--out-sequences", default="sequences_subset.tsv")
    sp.add_argument("--out-metabolites", default="metabolites_subset.tsv")
    sp.add_argument("--small", type=int, default=15)
    sp.add_argument("--mid", type=int, default=15)
    sp.add_argument("--big", type=int, default=12)
    sp.add_argument("--supra", type=int, default=8)
    sp.add_argument("--metabolites-count", type=int, default=5)
    _add_met_select(sp, "core")
    sp.set_defaults(func=cmd_subset)

    sp = sub.add_parser("setup", help="repos, venvs, weights")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--boltz-src", default=None, help="local boltz source dir (else PyPI)")
    sp.add_argument("--boltz-cp-src", default=None, help="local boltz-cp source dir (else git clone)")
    sp.add_argument("--force-serial", action="store_true")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("msa", help="download ColabFold a3m MSAs")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--mode", choices=["full", "targeted"], default="full")
    sp.add_argument("--sequences", default=None, help="needed for targeted mode")
    sp.add_argument("--hf-token", default=None)
    sp.set_defaults(func=cmd_msa)

    sp = sub.add_parser("plan", help="build YAML configs + batched queue")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--sequences", required=True)
    sp.add_argument("--metabolites", required=True)
    sp.add_argument("--batch-size", type=int, default=None, help="override per-GPU batch size")
    sp.add_argument("--cp-size", type=int, default=4)
    _add_met_select(sp, "all")
    sp.add_argument("--force-serial", action="store_true")
    sp.add_argument("--allow-missing-msa", action="store_true")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="execute batches")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--gpus", default="auto")
    sp.add_argument("--batch-id", default=None, help="run a single batch by id (SLURM task)")
    sp.add_argument("--batch-index", type=int, default=None, help="run a single batch by index")
    _add_boltz_run_flags(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("submit", help="generate/submit SLURM array jobs")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--bootstrap", required=True, help="path to run_cofold.sh on the cluster")
    sp.add_argument("--submit", action="store_true", help="actually call sbatch")
    sp.add_argument("--slurm-qos", default="low")
    sp.add_argument("--slurm-account", default=None)
    sp.add_argument("--slurm-cpus", type=int, default=8)
    sp.add_argument("--slurm-time", default="04:00:00")
    sp.add_argument("--slurm-throttle", type=int, default=0, help="max concurrent array tasks (0=unlimited)")
    sp.set_defaults(func=cmd_submit)

    sp = sub.add_parser("push", help="upload outputs to a HF dataset")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--org", default=DEFAULT_ORG)
    sp.add_argument("--dataset", default=DEFAULT_DATASET)
    sp.add_argument("--hf-token", default=None)
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("all", help="setup -> msa -> plan -> run -> push")
    sp.add_argument("--workdir", required=True)
    sp.add_argument("--sequences", required=True)
    sp.add_argument("--metabolites", required=True)
    sp.add_argument("--gpus", default="auto")
    sp.add_argument("--msa-mode", choices=["full", "targeted"], default="full")
    sp.add_argument("--batch-size", type=int, default=None)
    sp.add_argument("--cp-size", type=int, default=4)
    _add_met_select(sp, "all")
    sp.add_argument("--force-serial", action="store_true")
    sp.add_argument("--boltz-src", default=None)
    sp.add_argument("--boltz-cp-src", default=None)
    sp.add_argument("--org", default=DEFAULT_ORG)
    sp.add_argument("--dataset", default=DEFAULT_DATASET)
    sp.add_argument("--hf-token", default=None)
    sp.add_argument("--skip-setup", action="store_true")
    sp.add_argument("--skip-msa", action="store_true")
    sp.add_argument("--skip-push", action="store_true")
    _add_boltz_run_flags(sp)
    sp.set_defaults(func=cmd_all)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
