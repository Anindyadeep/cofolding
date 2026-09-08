# Protein–Metabolite Co-folding Campaign

## Task

Co-fold the structure of each protein with a given set of metabolites (protein + ligand
complexes) using **Boltz-2**, and **Boltz-CP** (NVIDIA's context-parallel fork) for proteins
too large to fit on a single GPU. We are **not** computing binding affinity — structure +
confidence only.

Given two inputs:

- `sequences.tsv` — proteins (`acc`, `entry`, `gene`, `seq`, `length`).
- `metabolites.tsv` — ligands (`updated_name`, `updated_smiles`, `status` = `core` | `non`).

...the runner does everything end to end on a machine that starts with **nothing installed**:

1. Sets up Boltz-2 and Boltz-CP (repos, venvs, model weights) automatically.
2. Downloads the precomputed ColabFold MSAs from the HF dataset
   [`LiteFold/human-proteome-wide-msa`](https://huggingface.co/datasets/LiteFold/human-proteome-wide-msa)
   (one `.a3m` per accession — the format Boltz consumes directly).
3. Builds one Boltz YAML per (protein, metabolite) pair, sorted/batched so GPUs are used well.
4. Runs the folds in a fail-safe, resumable, batched manner.
5. Organizes the important outputs into a clean tree and pushes them to a HF dataset.

Every phase is idempotent: finished pairs are skipped, so a pre-empted job (e.g. SLURM
`qos=low`) simply continues where it stopped when requeued.

## Design

Single readable script `run_cofold.py` (stdlib only for plan/run/submit; `huggingface_hub`
imported lazily for setup/msa/push) plus a `run_cofold.sh` bootstrap that creates the
orchestrator venv. Subcommands (or `all` to chain them):

| phase | what it does |
|---|---|
| `subset` | carve a size-balanced test set out of the full TSVs |
| `setup` | build the workdir tree, fetch repos, create the two venvs, download weights |
| `msa` | pull the `.a3m` MSAs from the HF dataset (`--mode full` mirror, or `targeted`) |
| `plan` | one YAML per pair + a batched, tiered queue |
| `run` | execute batches — locally across GPUs, or a single batch for a SLURM task |
| `submit` | emit + submit SLURM array jobs (one single-GPU task per batch, `--qos low --requeue`) |
| `push` | upload the organized `outputs/` to a HF dataset |
| `all` | `setup → msa → plan → run → push` |

**Size tiers & routing** (by residue length): `small ≤250`, `mid ≤700`, `big ≤1500`,
`supra >1500`. small/mid/big run as **single-GPU Boltz-2 jobs**, batched `8 / 4 / 2` pairs per
GPU respectively. `supra` proteins are routed to **Boltz-CP** (`size_cp=4`, one fold at a time
across all 4 GPUs); if Boltz-CP is unavailable or `<4` GPUs, they fall back to serial
(configurable with `--force-serial`).

**Parallelism model (fail-safe):** work is chunked into per-GPU **batches** (`batch_size`
inferences written together). Locally, a scheduler fills each visible GPU with one serial batch
at a time; CP batches use the whole node. On a cluster, `submit` writes a SLURM **array** (one
requeue-safe single-GPU task per batch) so jobs schedule easily under `qos=low` and survive
pre-emption. Idempotency comes from checking each pair's final output files before running.

**No affinity:** the generated YAML deliberately omits the `properties:` block:

```yaml
version: 1
sequences:
  - protein: { id: A, sequence: <SEQ>, msa: <abs path to .a3m> }
  - ligand:  { id: B, smiles: <SMILES> }
```

**Configurable knobs:** `--status core|non|all` (which metabolites to fold), `--msa-mode`,
`--batch-size`, `--cp-size`, `--force-serial`, Boltz params (`--recycling-steps`,
`--sampling-steps`, `--diffusion-samples`, `--max-msa-seqs`), and SLURM params
(`--slurm-qos`, `--slurm-account`, `--slurm-time`, `--slurm-throttle`).

## Expected output

Workdir tree (`--workdir`), matching `pp.md`:

```
<workdir>/
  repos/boltz-cp/          # Boltz-CP checkout
  checkpoints/             # Boltz-2 weights + CCD mols (BOLTZ_CACHE)
  msa_cache/a3m/shard-*/*.a3m
  sequences.tsv            # copy of the input used for this run
  metabolites.tsv          # copy of the input used for this run
  boltz_run_cache/<batch>/ # raw Boltz output
  outputs/
    <ACC>/
      structure_with_metabolite_id_<k>.cif   # co-folded structure (best model), pLDDT in B-factor
      scores_with_metabolite_id_<k>.json      # confidence metrics + metadata
    metabolites_index.json                    # k -> {name, smiles, status}
  configs/  queue/  env/  logs/  slurm/        # internal scaffolding
```

The **two files that matter per pair** live in `outputs/<ACC>/`:

- `structure_with_metabolite_id_<k>.cif` — protein (chain A) + metabolite (chain B) complex.
- `scores_with_metabolite_id_<k>.json` — Boltz confidence: `iptm`, `ptm`, `ligand_iptm`,
  `protein_iptm`, `complex_plddt`, `confidence_score`, … plus protein/metabolite metadata.

`<k>` is the `metabolite_id` from `metabolites_index.json` (stable = row index in the
metabolites TSV, so it stays consistent regardless of the `--status` filter).

## Running

Production files needed: **`run_cofold.py`, `run_cofold.sh`, `sequences.tsv`, `metabolites.tsv`**
(everything else is fetched). HF token via `HF_TOKEN` env, not a file.

```bash
export HF_TOKEN=...

# whole pipeline, only the core metabolites:
./run_cofold.sh all \
  --workdir ./runs \
  --sequences sequences.tsv \
  --metabolites metabolites.tsv \
  --status core \
  --org LiteFold --dataset cofolding-prod

# non-core only:  --status non      all metabolites:  --status all

# SLURM (qos=low, pre-emptible): stage once, then submit the array
./run_cofold.sh setup --workdir ./runs
./run_cofold.sh msa   --workdir ./runs
./run_cofold.sh plan  --workdir ./runs --sequences sequences.tsv --metabolites metabolites.tsv --status core
./run_cofold.sh submit --workdir ./runs --bootstrap "$PWD/run_cofold.sh" --submit \
  --slurm-account YOURACCT --slurm-throttle 32
```
