import glob, os, torch
if int(os.environ.get("SLURM_PROCID", "0")) != 0:
    raise SystemExit(0)  # only rank 0 does the (read-only) probe

BASE = "/iopsstor/scratch/cscs/course_00252/cler/_research/results/checkpoints"
CKPTS = {
    "CLER-V full (value, +7.34M)": f"{BASE}/cler-gdn-hidden-routeval-full-350m-llama2-fwe1b-muon-ckpt-2453393",
    "CLER-H full (error, +7.34M)": f"{BASE}/cler-gdn-hidden-route-350m-llama2-fwe1b-muon-ckpt-2437524",
    "CLER-V r=64 (value, +1.38M)": f"{BASE}/cler-gdn-hidden-routeval-r64-350m-llama2-fwe1b-muon-ckpt-2453394",
    "CLER-H r=64 (error, +1.38M)": f"{BASE}/cler-gdn-hidden-route-rank64-350m-llama2-fwe1b-muon-ckpt-2441918",
}

def effective_proj(model, idx_keys):
    """Return the effective projection matrix W_eff (output = W_eff @ input) for one routed layer."""
    if len(idx_keys) == 1:                       # full-rank: single Linear weight [out, in]
        return model[idx_keys[0]].float()
    # low-rank Sequential: down [r, in] (small out-dim), up [out, r] (large out-dim)
    mats = [model[k].float() for k in idx_keys]
    down = min(mats, key=lambda t: t.shape[0])   # [r, in]
    up = max(mats, key=lambda t: t.shape[0])     # [out, r]
    return up @ down

for name, base in CKPTS.items():
    last = sorted(glob.glob(f"{base}/iter_*"))[-1]
    sd = torch.load(f"{last}/mp_rank_00/model_optim_rng.pt", map_location="cpu", weights_only=False)
    model = sd.get("model", sd)
    proj_keys = [k for k in model if "cler_hidden_proj" in k]
    # group keys by layer index (the token right after 'cler_hidden_proj.')
    layers = {}
    for k in proj_keys:
        parts = k.split("cler_hidden_proj.")[1].split(".")
        layers.setdefault(parts[0], []).append(k)
    fro, opn = [], []
    for idx, keys in sorted(layers.items(), key=lambda x: int(x[0])):
        W = effective_proj(model, keys)
        fro.append(torch.linalg.norm(W).item())                  # Frobenius norm of effective proj
        opn.append(torch.linalg.svdvals(W)[0].item())            # operator (spectral) norm
    import statistics as st
    print(f"{name:30s} | layers={len(fro):2d} | "
          f"||W_eff||_F mean={st.mean(fro):.4f} (min {min(fro):.4f}, max {max(fro):.4f}) | "
          f"op-norm mean={st.mean(opn):.4f}")
