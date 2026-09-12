#!/usr/bin/env python3
"""Rough GPU-hour / cost benchmark for the protein–metabolite campaign.

No MSA download. Boltz-2 single-sequence mode (`msa: empty`) so fold quality
is irrelevant — we only measure wall time, then scale to the full TSV.

Length buckets match run_cofold.py: small≤250, mid≤700, big≤1500, supra>1500.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cofold import (  # noqa: E402
    TIER_BATCH,
    TIER_ORDER,
    Work,
    clean_seq,
    download_weights,
    gpu_ids,
    info,
    ok,
    section,
    setup_boltz,
    tier_of,
)

# RunPod A100 80GB SXM Secure Cloud (this box). Override with --gpu-price.
DEFAULT_GPU_PRICE = 1.49


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = []
        for row in csv.DictReader(fh, delimiter="\t"):
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
        return rows


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def pick_protein(rows: list[dict], tier: str, which: str) -> dict:
    pool = []
    for row in rows:
        seq = clean_seq(row.get("seq", ""))
        acc = (row.get("acc") or "").strip()
        if not acc or not seq:
            continue
        if tier_of(len(seq)) != tier:
            continue
        pool.append({
            "acc": acc,
            "gene": (row.get("gene") or "").strip(),
            "seq": seq,
            "length": len(seq),
            "tier": tier,
        })
    if not pool:
        raise SystemExit(f"no proteins in tier {tier}")
    pool.sort(key=lambda p: (p["length"], p["acc"]))
    if which == "short":
        return pool[0]
    if which == "long":
        return pool[-1]
    return pool[len(pool) // 2]


def pick_metabolite(rows: list[dict], name: str | None) -> dict:
    pool = [r for r in rows if (r.get("updated_name") or "").strip() and (r.get("updated_smiles") or "").strip()]
    if name:
        for r in pool:
            if r["updated_name"] == name:
                return {"name": r["updated_name"], "smiles": r["updated_smiles"], "status": r.get("status", "")}
        raise SystemExit(f"metabolite {name!r} not found")
    for r in pool:
        if r["updated_name"] == "Citrate":
            return {"name": r["updated_name"], "smiles": r["updated_smiles"], "status": r.get("status", "")}
    return {"name": pool[0]["updated_name"], "smiles": pool[0]["updated_smiles"], "status": pool[0].get("status", "")}


def campaign_counts(seq_rows: list[dict], met_rows: list[dict]) -> dict:
    proteins = {t: 0 for t in TIER_ORDER}
    for row in seq_rows:
        seq = clean_seq(row.get("seq", ""))
        if seq and (row.get("acc") or "").strip():
            proteins[tier_of(len(seq))] += 1
    mets = {"core": 0, "non": 0}
    for row in met_rows:
        if not (row.get("updated_name") or "").strip() or not (row.get("updated_smiles") or "").strip():
            continue
        st = (row.get("status") or "").strip()
        if st in mets:
            mets[st] += 1
    mets["all"] = mets["core"] + mets["non"]
    return {"proteins": proteins, "proteins_total": sum(proteins.values()), "metabolites": mets}


def write_pair_yaml(path: Path, sequence: str, smiles: str) -> None:
    seq = json.dumps(sequence, ensure_ascii=False)
    smi = json.dumps(smiles, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq}\n"
        "      msa: empty\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: {smi}\n"
    )


def boltz_cmd(work: Work, config: Path, out_dir: Path, args) -> list[str]:
    return [
        str(work.boltz_bin), "predict", str(config),
        "--out_dir", str(out_dir),
        "--cache", str(work.checkpoints),
        "--model", "boltz2",
        "--accelerator", "gpu",
        "--devices", "1",
        "--num_workers", str(args.num_workers),
        "--recycling_steps", str(args.recycling_steps),
        "--sampling_steps", str(args.sampling_steps),
        "--diffusion_samples", str(args.diffusion_samples),
        "--max_msa_seqs", "1",
        "--output_format", "mmcif",
    ]


def launch(cmd: list[str], gpu: int, log: Path, work: Work) -> subprocess.Popen:
    env = os.environ.copy()
    env["BOLTZ_CACHE"] = str(work.checkpoints)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    fh = log.open("w")
    fh.write(f"[{now_iso()}] GPU {gpu}\n$ {' '.join(cmd)}\n\n")
    fh.flush()
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    proc._bench_log = fh  # type: ignore[attr-defined]
    proc._bench_t0 = time.perf_counter()  # type: ignore[attr-defined]
    return proc


def reap(proc: subprocess.Popen) -> dict:
    rc = proc.wait()
    elapsed = time.perf_counter() - proc._bench_t0  # type: ignore[attr-defined]
    proc._bench_log.close()  # type: ignore[attr-defined]
    return {"returncode": rc, "seconds": elapsed, "ok": rc == 0}


def wait_all(handles: dict[str, subprocess.Popen]) -> dict[str, dict]:
    out = {}
    pending = dict(handles)
    while pending:
        done = [k for k, p in pending.items() if p.poll() is not None]
        if not done:
            time.sleep(2)
            continue
        for k in done:
            out[k] = reap(pending.pop(k))
            tag = "OK" if out[k]["ok"] else "FAIL"
            info(f"  {tag} {k}: {out[k]['seconds']:.1f}s (exit={out[k]['returncode']})")
    return out


def fmt_hours(seconds: float) -> str:
    h = seconds / 3600.0
    if h >= 100:
        return f"{h:,.0f} h"
    if h >= 10:
        return f"{h:,.1f} h"
    return f"{h:,.2f} h"


def fmt_money(x: float) -> str:
    if x >= 1000:
        return f"${x:,.0f}"
    return f"${x:,.2f}"


def parse_loader_times(log: Path) -> dict:
    """Cumulative Lightning predict-bar times: first item and last item."""
    text = Path(log).read_text(errors="replace") if Path(log).exists() else ""
    hits = []
    for m in re.finditer(
        r"Predicting DataLoader 0:\s+\d+%\|.*?\|\s+(\d+)/(\d+)\s+\[(\d+):(\d+)<",
        text,
    ):
        hits.append((int(m.group(1)), int(m.group(2)), int(m.group(3)) * 60 + int(m.group(4))))
    if not hits:
        return {}
    by_done = {done: sec for done, _total, sec in hits}
    n = hits[-1][1]
    first = by_done.get(1)
    total = by_done.get(n)
    extra = (total - first) / max(1, n - 1) if first is not None and total is not None and n > 1 else None
    return {"n": n, "first_s": first, "total_s": total, "extra_s": extra}


def estimate_tier(n_pairs: int, batch_size: int, t_first: float, t_extra: float) -> dict:
    """Each campaign batch is a new Boltz process: 1st pair pays t_first, rest t_extra."""
    n_batches = math.ceil(n_pairs / batch_size) if n_pairs else 0
    last = n_pairs - (n_batches - 1) * batch_size if n_batches else 0
    full = n_batches - (1 if last != batch_size and n_batches else 0)
    if n_batches and last == batch_size:
        full = n_batches
        last = 0

    def batch_sec(n: int) -> float:
        if n <= 0:
            return 0.0
        return t_first + max(0, n - 1) * t_extra

    gpu_s = full * batch_sec(batch_size) + (batch_sec(last) if last else 0.0)
    return {
        "pairs": n_pairs,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "t_first_s": t_first,
        "t_extra_s": t_extra,
        "gpu_seconds": gpu_s,
        "gpu_hours": gpu_s / 3600.0,
        "sec_per_pair": (gpu_s / n_pairs) if n_pairs else 0.0,
    }


def derive_rates(timings: dict, t_one_warm: float | None) -> tuple[dict, dict, float, list[str]]:
    """Steady-state t_first / t_extra per tier from wall clocks + Lightning bars."""
    notes = []
    infer = {}
    for t, rec in timings.items():
        parsed = parse_loader_times(rec.get("log", ""))
        rec["infer"] = parsed
        infer[t] = parsed

    small_bar = infer.get("small") or {}
    infer_first_small = small_bar.get("first_s") or 23.0
    infer_extra_small = small_bar.get("extra_s") or 10.0
    cold_small = timings["small"]["seconds"]
    n_small = timings["small"].get("n_in_process") or 2
    # Cold 2-job wall = startup_cold + first_infer + extra. Don't subtract a later warm 1-job.
    startup_cold = cold_small - (small_bar.get("total_s") or infer_first_small + infer_extra_small)

    if t_one_warm:
        startup_warm = max(15.0, t_one_warm - infer_first_small)
    else:
        startup_warm = max(15.0, startup_cold * 0.35)
    notes.append(
        f"Process startup is ~{startup_cold:.0f}s on a cold 4-GPU simultaneous launch "
        f"(checkpoint load contention) and ~{startup_warm:.0f}s once weights are in page cache "
        f"(measured warm small 1-job = {t_one_warm:.0f}s)."
        if t_one_warm else
        f"Process startup estimated at {startup_warm:.0f}s (warm) / {startup_cold:.0f}s (cold)."
    )
    notes.append(
        f"In-process extra pair time for small is {infer_extra_small:.0f}s "
        f"(Lightning bar {infer_first_small:.0f}s then +{infer_extra_small:.0f}s), not wall_2job - wall_1job."
    )

    t_first, t_extra = {}, {}
    for t in TIER_ORDER:
        rec = timings[t]
        bar = infer.get(t) or {}
        infer_first = bar.get("first_s")
        infer_extra = bar.get("extra_s")
        if infer_first is None:
            infer_first = max(5.0, rec["seconds"] * 0.2)
            notes.append(f"no Lightning bar for {t}; guessed infer={infer_first:.0f}s")
        # New process, warm machine: startup + first infer (JIT for this length).
        t_first[t] = startup_warm + infer_first
        if t == "small":
            t_extra[t] = infer_extra_small
            rec["t_first_s"] = t_first[t]
            rec["t_extra_s"] = t_extra[t]
            continue
        # Same-process extra: drop the ~JIT gap seen on small (first - extra).
        jit = max(0.0, infer_first_small - infer_extra_small)
        t_extra[t] = max(infer_extra_small, infer_first - jit) if infer_extra is None else infer_extra
        rec["t_first_s"] = t_first[t]
        rec["t_extra_s"] = t_extra[t]
    return t_first, t_extra, startup_warm, notes


def build_estimates(counts: dict, t_first: dict, t_extra: dict, price: float, n_gpus: int) -> dict:
    out = {}
    for label, n_met in (("core metabolites", counts["metabolites"]["core"]),
                         ("all metabolites", counts["metabolites"]["all"])):
        by_tier = {}
        for t in TIER_ORDER:
            n_pairs = counts["proteins"][t] * n_met
            by_tier[t] = estimate_tier(n_pairs, TIER_BATCH[t], t_first[t], t_extra[t])
        gpu_s = sum(v["gpu_seconds"] for v in by_tier.values())
        n_pairs = sum(v["pairs"] for v in by_tier.values())
        gpu_h = gpu_s / 3600.0
        out[label] = {
            "n_metabolites": n_met,
            "n_pairs": n_pairs,
            "gpu_seconds": gpu_s,
            "gpu_hours": gpu_h,
            "sec_per_pair": gpu_s / n_pairs if n_pairs else 0.0,
            "by_tier": by_tier,
            "cost_usd": gpu_h * price,
            "wall_hours": {str(n): gpu_h / n for n in (1, 4, 8, 32, 64) if n},
            "wall_hours_1gpu": gpu_h,
            "wall_hours_4gpu": gpu_h / max(1, n_gpus),
        }
    return out


def print_report(report: dict) -> None:
    section("Benchmark results")
    hw = report["hardware"]
    print(f"  GPU: {hw['name']} x {hw['n_gpus_seen']}   price=${report['gpu_price_per_hour']}/GPU-h")
    print(f"  Boltz settings: recycling={report['recycling_steps']}  sampling={report['sampling_steps']}  "
          f"diffusion_samples={report['diffusion_samples']}  msa=empty")
    print()
    print("  Timed protein–metabolite pairs (Citrate ligand)")
    print(f"    {'tier':<8}{'L':>6}  {'acc':<12}{'cold wall':>12}{'infer':>10}{'t_first':>10}{'t_extra':>10}")
    for t in TIER_ORDER:
        r = report["timings"].get(t)
        if not r:
            continue
        inf = (r.get("infer") or {}).get("first_s")
        inf_s = f"{inf:.0f}s" if inf else "?"
        print(f"    {t:<8}{r['length']:>6}  {r['acc']:<12}{r['seconds']:>11.1f}s{inf_s:>10}"
              f"{r.get('t_first_s', 0):>9.0f}s{r.get('t_extra_s', 0):>9.0f}s")
    print("    t_first = warm process startup + first infer; t_extra = next pair in the same Boltz process.")
    print()
    if report.get("parallel"):
        print("  Parallelization (identical small pair, one new process per GPU, lockstep start)")
        one = next((r for r in report["parallel"] if r["n_jobs"] == 1), None)
        for row in report["parallel"]:
            ideal = (one["wall_s"] if one else row["wall_s"])
            eff = (ideal / row["wall_s"] * 100) if row["wall_s"] else 0
            print(f"    {row['n_gpus']} GPU  n={row['n_jobs']}  wall={row['wall_s']:.1f}s  "
                  f"pairs/h/GPU={row['pairs_per_hour_per_gpu']:.1f}  scaling={eff:.0f}% of 1-GPU rate")
        print("    Lockstep starts contend on CPU/disk. A long queue staggers, so scaling approaches linear.")
        print()

    counts = report["campaign"]
    print(f"  Campaign size: {counts['proteins_total']} proteins  "
          f"(small={counts['proteins']['small']}, mid={counts['proteins']['mid']}, "
          f"big={counts['proteins']['big']}, supra={counts['proteins']['supra']})")
    print(f"  Metabolites: core={counts['metabolites']['core']}  non={counts['metabolites']['non']}  "
          f"all={counts['metabolites']['all']}")
    print(f"  Campaign batch sizes: {TIER_BATCH}")
    print()

    price = report["gpu_price_per_hour"]
    for label, block in report["estimates"].items():
        print(f"  === {label}  ({block['n_metabolites']} metabolites, {block['n_pairs']:,} pairs) ===")
        print(f"    {'tier':<8}{'pairs':>12}{'s/pair':>10}{'GPU-h':>12}{'@4 GPU':>12}{'@32 GPU':>12}{'cost':>12}")
        for t in TIER_ORDER:
            e = block["by_tier"][t]
            print(f"    {t:<8}{e['pairs']:>12,}{e['sec_per_pair']:>9.1f}s{e['gpu_hours']:>12,.0f}"
                  f"{fmt_hours(e['gpu_hours']/4*3600):>12}{fmt_hours(e['gpu_hours']/32*3600):>12}"
                  f"{fmt_money(e['gpu_hours'] * price):>12}")
        tot = block["gpu_hours"]
        print(f"    {'TOTAL':<8}{block['n_pairs']:>12,}{block['sec_per_pair']:>9.1f}s{tot:>12,.0f}"
              f"{fmt_hours(tot/4*3600):>12}{fmt_hours(tot/32*3600):>12}"
              f"{fmt_money(tot * price):>12}")
        print(f"    wall if the queue stays full:  "
              + "  ".join(f"{n} GPU={fmt_hours(tot/n*3600)}" for n in (1, 4, 8, 32, 64)))
        print()
    print("  Notes")
    for n in report["notes"]:
        print(f"    - {n}")


def cmd_bench(args: argparse.Namespace) -> None:
    work = Work(Path(args.workdir))
    work.make_tree()
    if not args.skip_setup:
        setup_boltz(work, None)
        download_weights(work)

    if not work.boltz_bin.exists():
        raise SystemExit("boltz is not installed; run without --skip-setup")

    gpus = gpu_ids() if args.gpus == "auto" else [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise SystemExit("no GPUs found")
    info(f"GPUs: {gpus}")

    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        smi = "unknown"

    seq_rows = read_rows(Path(args.sequences))
    met_rows = read_rows(Path(args.metabolites))
    counts = campaign_counts(seq_rows, met_rows)
    ligand = pick_metabolite(met_rows, args.metabolite)
    info(f"ligand: {ligand['name']}  smiles={ligand['smiles'][:40]}...")

    proteins = {
        "small": pick_protein(seq_rows, "small", "median"),
        "mid": pick_protein(seq_rows, "mid", "median"),
        "big": pick_protein(seq_rows, "big", "median"),
        "supra": pick_protein(seq_rows, "supra", "short"),
    }
    for t, p in proteins.items():
        info(f"  sample {t}: {p['acc']} {p['gene']} L={p['length']}")

    bench_root = work.root / "bench"
    if bench_root.exists():
        import shutil
        shutil.rmtree(bench_root)
    bench_root.mkdir(parents=True)

    # --- Phase 1: one pair per bucket, 2-job small batch on GPU0 to get extra-in-batch time ---
    section("Phase 1: per-bucket timings (parallel across GPUs)")
    jobs = {}
    # small: two yaml files in one directory -> one Boltz process, two folds
    small_dir = bench_root / "small_batch2"
    write_pair_yaml(small_dir / "small_a.yaml", proteins["small"]["seq"], ligand["smiles"])
    write_pair_yaml(small_dir / "small_b.yaml", proteins["small"]["seq"], ligand["smiles"])
    jobs["small"] = {
        "config": small_dir,
        "out": bench_root / "out_small",
        "n": 2,
        "protein": proteins["small"],
        "gpu": gpus[0],
    }
    rest_gpus = gpus[1:] + gpus[:1]
    for i, t in enumerate(["mid", "big", "supra"]):
        d = bench_root / t
        write_pair_yaml(d / f"{t}.yaml", proteins[t]["seq"], ligand["smiles"])
        jobs[t] = {
            "config": d,
            "out": bench_root / f"out_{t}",
            "n": 1,
            "protein": proteins[t],
            "gpu": rest_gpus[i % len(gpus)],
        }

    handles = {}
    t_phase1 = time.perf_counter()
    for name, job in jobs.items():
        log = work.logs / f"bench_{name}.log"
        cmd = boltz_cmd(work, job["config"], job["out"], args)
        info(f"START {name} on GPU {job['gpu']}  n={job['n']}  L={job['protein']['length']}")
        handles[name] = launch(cmd, job["gpu"], log, work)
    phase1 = wait_all(handles)
    phase1_wall = time.perf_counter() - t_phase1

    timings = {}
    t_extra_small = None
    for name, job in jobs.items():
        rec = phase1[name]
        per = rec["seconds"] / job["n"]
        proto = {k: v for k, v in job["protein"].items() if k != "seq"}
        timings[name] = {
            **proto,
            "seconds": rec["seconds"],
            "n_in_process": job["n"],
            "sec_per_pair_in_process": per,
            "ok": rec["ok"],
            "returncode": rec["returncode"],
            "gpu": job["gpu"],
            "log": str(work.logs / f"bench_{name}.log"),
        }
        if name == "small" and rec["ok"] and job["n"] == 2:
            # overhead unknown yet; filled after the 1-job small run in phase 2
            timings[name]["batch2_s"] = rec["seconds"]

    # --- Phase 2: parallelization scaling with the small pair ---
    section("Phase 2: parallelization scaling (small pair)")
    small = proteins["small"]
    parallel = []
    # 1-job single process (needed to split small batch2 into first vs extra)
    one_dir = bench_root / "small_one"
    write_pair_yaml(one_dir / "one.yaml", small["seq"], ligand["smiles"])
    t0 = time.perf_counter()
    p = launch(boltz_cmd(work, one_dir, bench_root / "out_small_one", args), gpus[0],
               work.logs / "bench_small_one.log", work)
    r1 = reap(p)
    t_one = r1["seconds"]
    info(f"  1 GPU / 1 job: {t_one:.1f}s ok={r1['ok']}")
    parallel.append({
        "n_gpus": 1, "n_jobs": 1, "wall_s": t_one,
        "gpu_seconds": t_one, "ok": r1["ok"],
        "pairs_per_hour_per_gpu": (3600.0 / t_one) if t_one else 0.0,
    })

    # N GPUs × 1 job each
    for n in ([2, 4] if len(gpus) >= 2 else []):
        if n > len(gpus):
            continue
        hs = {}
        t0 = time.perf_counter()
        for i in range(n):
            d = bench_root / f"par_{n}_{i}"
            write_pair_yaml(d / "p.yaml", small["seq"], ligand["smiles"])
            hs[f"p{i}"] = launch(
                boltz_cmd(work, d, bench_root / f"out_par_{n}_{i}", args),
                gpus[i], work.logs / f"bench_par_{n}_{i}.log", work,
            )
        recs = wait_all(hs)
        wall = time.perf_counter() - t0
        gpu_s = sum(v["seconds"] for v in recs.values())
        ok_all = all(v["ok"] for v in recs.values())
        parallel.append({
            "n_gpus": n, "n_jobs": n, "wall_s": wall,
            "gpu_seconds": gpu_s, "ok": ok_all,
            "pairs_per_hour_per_gpu": (n * 3600.0 / wall) / n if wall else 0.0,
        })
        info(f"  {n} GPU / {n} jobs: wall={wall:.1f}s  gpu-s={gpu_s:.1f}  ok={ok_all}")

    write_and_print_report(
        args, timings, parallel, counts, ligand, smi, gpus, phase1_wall, t_one,
    )


def write_and_print_report(args, timings, parallel, counts, ligand, smi, gpus, phase1_wall, t_one_warm):
    t_first, t_extra, startup_warm, derived_notes = derive_rates(timings, t_one_warm)
    notes = [
        "No MSAs downloaded; Boltz single-sequence mode (faster than ColabFold MSA, slightly optimistic vs production).",
        "Affinity head is off (same as the campaign YAML).",
        f"Timed against {ligand['name']}; ligand size is a second-order effect vs protein length.",
        "supra sample is the shortest protein in that bucket (1501 aa). It fit on one A100 80GB (~27 GB). Median supra is 1888 aa — treat supra as a lower bound.",
        "Campaign batches are independent Boltz processes (1 GPU each for small/mid/big). Wall clock ≈ GPU-hours / N_GPUs once the queue is staggered.",
        "Supra in production is meant for Boltz-CP across 4 GPUs. This bench used 1 GPU; if CP is used, supra GPU-hours scale by ~4.",
        f"Price default is RunPod A100 80GB SXM Secure Cloud ${args.gpu_price}/h. Pass --gpu-price to change.",
        f"Steady-state model: warm startup {startup_warm:.0f}s + Lightning infer time, with campaign batch sizes {TIER_BATCH}.",
        *derived_notes,
    ]
    estimates = build_estimates(counts, t_first, t_extra, args.gpu_price, max(1, len(gpus) if gpus else 4))
    for t, rec in timings.items():
        rec.pop("seq", None)
    report = {
        "created_at": now_iso(),
        "hardware": {"name": smi, "gpus": gpus, "n_gpus_seen": len(gpus) if gpus else 0},
        "gpu_price_per_hour": args.gpu_price,
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "ligand": ligand,
        "campaign": counts,
        "timings": timings,
        "parallel": parallel,
        "t_first_s": t_first,
        "t_extra_s": t_extra,
        "startup_warm_s": startup_warm,
        "phase1_wall_s": phase1_wall,
        "estimates": estimates,
        "notes": notes,
    }
    out_path = Path(args.out)
    write_json(out_path, report)
    print_report(report)
    ok(f"wrote {out_path}")


def cmd_recompute(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.out).read_text())
    write_and_print_report(
        args,
        report["timings"],
        report.get("parallel") or [],
        report["campaign"],
        report["ligand"],
        report["hardware"]["name"],
        report["hardware"].get("gpus") or [],
        report.get("phase1_wall_s") or 0.0,
        next((r["wall_s"] for r in report.get("parallel") or [] if r.get("n_jobs") == 1), None),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Boltz-2 co-folding time/cost benchmark (no MSA)")
    p.add_argument("--workdir", default=str(Path(__file__).resolve().parent / "bench_runs"))
    p.add_argument("--sequences", default=str(Path(__file__).resolve().parent / "sequences.tsv"))
    p.add_argument("--metabolites", default=str(Path(__file__).resolve().parent / "metabolites.tsv"))
    p.add_argument("--out", default=str(Path(__file__).resolve().parent / "bench_report.json"))
    p.add_argument("--gpus", default="auto")
    p.add_argument("--gpu-price", type=float, default=DEFAULT_GPU_PRICE,
                   help="USD per GPU-hour (default: RunPod A100 80GB SXM Secure)")
    p.add_argument("--metabolite", default=None, help="metabolite name (default: Citrate)")
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--recompute-only", action="store_true",
                   help="rebuild the cost report from existing logs/JSON, no GPU work")
    p.add_argument("--recycling-steps", type=int, default=3, dest="recycling_steps")
    p.add_argument("--sampling-steps", type=int, default=200, dest="sampling_steps")
    p.add_argument("--diffusion-samples", type=int, default=1, dest="diffusion_samples")
    p.add_argument("--num-workers", type=int, default=2, dest="num_workers")
    args = p.parse_args()
    if args.recompute_only:
        cmd_recompute(args)
    else:
        cmd_bench(args)


if __name__ == "__main__":
    main()
