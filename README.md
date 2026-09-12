# Protein–metabolite co-folding

Fold each protein in `sequences.tsv` together with metabolites from `metabolites.tsv` using Boltz-2 (Boltz-CP for the really large ones). Structure + confidence only — no binding affinity.

You need a GPU box, the four files in this folder (`run_cofold.sh`, `run_cofold.py`, `sequences.tsv`, `metabolites.tsv`), and a Hugging Face token so we can pull MSAs and (optionally) push results.

```bash
export HF_TOKEN=...
chmod +x run_cofold.sh
```

Everything installs itself into `--workdir` the first time you run `setup`.

## Getting started

One-time setup (Boltz, weights). Then download MSAs, plan the queue, run it.

```bash
./run_cofold.sh setup --workdir ./runs
./run_cofold.sh msa   --workdir ./runs --mode targeted --sequences sequences.tsv
```

Use `--mode full` on `msa` if you want the whole ColabFold mirror instead of only the proteins in the TSV.

### Pick which metabolites to fold

`plan` decides the work. `run` just executes whatever is in that workdir.

**A handful of ligands first** (order is the run order — ATP finishes before GTP, so you can start looking at results early):

```bash
./run_cofold.sh plan --workdir ./runs \
  --sequences sequences.tsv --metabolites metabolites.tsv \
  --names "ATP,GTP,NAD+,NADH"

./run_cofold.sh run --workdir ./runs
```

`--names` is fuzzy on charge suffixes: `ATP` matches `ATP(4-)`, not `dATP`. Add more names in the same list whenever you want.

**Every core metabolite:**

```bash
./run_cofold.sh plan --workdir ./runs \
  --sequences sequences.tsv --metabolites metabolites.tsv \
  --status core

./run_cofold.sh run --workdir ./runs
```

`--status non` is the rest. `--status all` is everything.

You can do the four names first, then re-plan with `--status core` in the same `--workdir`. Pairs that already have outputs are skipped.

## What you get

Per protein / metabolite pair, two files:

```
runs/outputs/<ACC>/structure_with_metabolite_id_<k>.cif
runs/outputs/<ACC>/scores_with_metabolite_id_<k>.json
```

`<k>` is the row index in `metabolites.tsv` (stable, even if you only planned a subset). The name mapping lives in `runs/outputs/metabolites_index.json`.

For the priority four:

| name | k |
|---|---|
| ATP | 8 |
| GTP | 42 |
| NAD+ | 82 |
| NADH | 83 |

## How jobs are batched

Proteins are grouped by length: small ≤250, mid ≤700, big ≤1500, supra >1500. On each GPU we fold `8 / 4 / 2 / 1` pairs at a time. The queue is metabolite-by-metabolite, so one ligand is filled out before the next.

Supra proteins try Boltz-CP across 4 GPUs; if that is not installed they fall back to a single GPU (`--force-serial` to skip CP entirely).

## Other useful bits

```bash
# size-balanced test slice of the TSVs
./run_cofold.sh subset --sequences sequences.tsv --metabolites metabolites.tsv \
  --names "ATP,GTP" --out-sequences sequences_subset.tsv --out-metabolites metabolites_subset.tsv

# SLURM array (one GPU per batch, qos=low, requeue-safe)
./run_cofold.sh submit --workdir ./runs --bootstrap "$PWD/run_cofold.sh" --submit \
  --slurm-account YOURACCT --slurm-throttle 32

# upload outputs/
./run_cofold.sh push --workdir ./runs --org LiteFold --dataset cofolding-prod
```

Or chain it: `./run_cofold.sh all --workdir ./runs --sequences sequences.tsv --metabolites metabolites.tsv --status core`.

Rough time/cost estimates (no MSA download, empty-MSA timing only) are in `bench_cost.py` / `bench_report.json`.
