#!/usr/bin/env python3
"""
Group PCA (IncrementalPCA) on ALL subjects + BOTH tasks (compL1 + compLn) together.

Input layout (Anvil):
  /anvil/scratch/x-onarayanamud/PreprosessedData/sub-XX/preprocessed/
      sub-XX_task-compL1_allruns_bold_preproc.nii.gz
      sub-XX_task-compLn_allruns_bold_preproc.nii.gz

Goal:
- Learn ONE shared set of K PCA spatial components ("networks") from the combined dataset.
- Project each run (subject x task) onto those components -> PC timecourses for later HMM/features/classification.

Outputs (requested):
  /home/x-onarayanamud/NewPCA/results/result1

Saves:
- manifest_runs_K*.csv (all included runs)
- mask_from_first_run.nii.gz (mask built from first run mean!=0)
- group_pca_components_K*.nii.gz (4D NIfTI, last dim = K)
- group_pca_components_K*.npy
- explained variance arrays + explained_variance_summary.csv
- PCA model pickle
- timecourses for each run (T x K) as .npy and .csv
- timecourses_index_K*.csv

Author: Onila Rasanjala (adapted)
"""

import os
import gc
import json
import argparse
import pickle
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import IncrementalPCA


# -----------------------------
# Helpers
# -----------------------------

def list_subjects(base_root: str) -> List[str]:
    return sorted(d for d in os.listdir(base_root) if d.startswith("sub-"))


def find_all_runs(base_root: str, tasks: List[str]) -> List[Dict]:
    """
    Return list of dicts:
      {"subject": "sub-01", "task": "compL1", "file": "/path/to/file.nii.gz"}
    """
    runs = []
    subs = list_subjects(base_root)
    for sub in subs:
        sub_dir = os.path.join(base_root, sub, "preprocessed")
        for task in tasks:
            fname = os.path.join(sub_dir, f"{sub}_task-{task}_allruns_bold_preproc.nii.gz")
            if os.path.exists(fname):
                runs.append({"subject": sub, "task": task, "file": fname})
            else:
                print(f"[WARN] Missing: {fname}")

    runs = sorted(runs, key=lambda x: (x["subject"], x["task"]))
    print(f"[INFO] Found {len(runs)} runs total across tasks={tasks}.")
    return runs


def build_mask_from_reference(ref_file: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Mask = non-zero voxels in the mean image across time for the reference run.
    This assumes non-brain voxels are zeros after preprocessing.
    """
    print(f"[INFO] Building mask from reference: {ref_file}")
    img = nib.load(ref_file)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got {data.shape} for {ref_file}")

    mean_img = data.mean(axis=-1)
    mask = mean_img != 0

    del data, mean_img
    gc.collect()

    print(f"[INFO] Mask shape {mask.shape}, voxels in mask: {int(mask.sum())}")
    return mask, img.affine, img.header


def load_run_matrix(fname: str, mask: np.ndarray, center: bool = True, zscore: bool = False) -> np.ndarray:
    """
    Load 4D fMRI -> apply mask -> return X (T x Vmask).
    center: subtract mean across time per voxel
    zscore: divide by std across time per voxel (optional)
    """
    img = nib.load(fname)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D image, got {data.shape} for {fname}")

    vox = data[mask, :]  # (V, T)
    X = vox.T            # (T, V)

    del data, vox
    gc.collect()

    if center:
        X -= X.mean(axis=0, keepdims=True)

    if zscore:
        std = X.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        X = X / std

    return X


def save_mask(mask: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, out_file: str) -> None:
    img = nib.Nifti1Image(mask.astype(np.uint8), affine, header)
    nib.save(img, out_file)
    print(f"[INFO] Saved mask: {out_file}")


def save_components_as_nifti(components: np.ndarray, mask: np.ndarray,
                            affine: np.ndarray, header: nib.Nifti1Header,
                            out_file: str) -> None:
    """
    components: (K x Vmask) -> 4D NIfTI (X,Y,Z,K)
    """
    K, V = components.shape
    if V != int(mask.sum()):
        raise ValueError(f"components V={V} != mask voxels {int(mask.sum())}")

    vol_shape = mask.shape
    comp_4d = np.zeros(vol_shape + (K,), dtype=np.float32)
    for k in range(K):
        vol = np.zeros(vol_shape, dtype=np.float32)
        vol[mask] = components[k, :]
        comp_4d[..., k] = vol

    img = nib.Nifti1Image(comp_4d, affine, header)
    nib.save(img, out_file)
    print(f"[INFO] Saved components NIfTI: {out_file}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Group PCA (IncrementalPCA) on all subs + compL1+compLn combined.")
    parser.add_argument("--base_root", type=str, default="/anvil/scratch/x-onarayanamud/PreprosessedData")
    parser.add_argument("--out_root", type=str, default="/home/x-onarayanamud/NewPCA/results/result1")
    parser.add_argument("--tasks", type=str, default="compL1,compLn",
                        help="Comma-separated tasks to include (default: compL1,compLn)")
    parser.add_argument("--n_components", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--center", action="store_true")
    parser.add_argument("--no-center", dest="center", action="store_false")
    parser.set_defaults(center=True)
    parser.add_argument("--zscore", action="store_true",
                        help="Optional z-score per voxel across time (after centering).")

    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    tc_dir = os.path.join(args.out_root, "timecourses")
    os.makedirs(tc_dir, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        tasks = ["compL1", "compLn"]

    print("==== Group PCA (ALL subjects + compL1+compLn combined) ====")
    print(f"base_root: {args.base_root}")
    print(f"out_root:  {args.out_root}")
    print(f"tasks:     {tasks}")
    print(f"K:         {args.n_components}")
    print(f"batch:     {args.batch_size}")
    print(f"center:    {args.center}")
    print(f"zscore:    {args.zscore}")

    # 1) Collect all runs (sub x task)
    runs = find_all_runs(args.base_root, tasks)
    if len(runs) == 0:
        raise RuntimeError("No runs found. Check base_root and file naming.")

    manifest = pd.DataFrame(runs)
    manifest_path = os.path.join(args.out_root, f"manifest_runs_K{args.n_components}.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"[INFO] Saved manifest: {manifest_path}")

    # 2) Build mask from first run
    mask, affine, header = build_mask_from_reference(runs[0]["file"])
    mask_path = os.path.join(args.out_root, "mask_from_first_run.nii.gz")
    save_mask(mask, affine, header, mask_path)

    # 3) Fit IncrementalPCA streaming run-by-run (PASS 1)
    ipca = IncrementalPCA(n_components=args.n_components, batch_size=args.batch_size)

    total_tp = 0
    print("[INFO] PASS 1: partial_fit over all runs...")
    for i, r in enumerate(runs, 1):
        f = r["file"]
        print(f"  [{i}/{len(runs)}] {r['subject']} {r['task']} -> {os.path.basename(f)}")
        X = load_run_matrix(f, mask, center=args.center, zscore=args.zscore)
        total_tp += X.shape[0]
        ipca.partial_fit(X)
        del X
        gc.collect()

    print("[INFO] Fit complete.")
    print("[INFO] Explained variance ratio (first 10):", ipca.explained_variance_ratio_[:10])

    prefix = os.path.join(args.out_root, f"group_pca_K{args.n_components}")

    # 4) Save model + arrays
    with open(prefix + "_model.pkl", "wb") as f:
        pickle.dump(ipca, f)

    np.save(prefix + "_components.npy", ipca.components_.astype(np.float32))
    np.save(prefix + "_explained_variance.npy", ipca.explained_variance_.astype(np.float32))
    np.save(prefix + "_explained_variance_ratio.npy", ipca.explained_variance_ratio_.astype(np.float32))
    np.save(prefix + "_singular_values.npy", ipca.singular_values_.astype(np.float32))

    ev = pd.DataFrame({
        "component": np.arange(1, args.n_components + 1),
        "explained_variance": ipca.explained_variance_,
        "explained_variance_ratio": ipca.explained_variance_ratio_,
        "singular_value": ipca.singular_values_,
    })
    ev.to_csv(prefix + "_explained_variance_summary.csv", index=False)

    # 5) Save spatial component maps (NIfTI)
    comp_nii = prefix + "_components.nii.gz"
    save_components_as_nifti(ipca.components_, mask, affine, header, comp_nii)

    # 6) PASS 2: transform each run -> timecourses
    print("[INFO] PASS 2: transform each run -> timecourses...")
    tc_index_rows = []
    for i, r in enumerate(runs, 1):
        f = r["file"]
        sub = r["subject"]
        task = r["task"]
        print(f"  [{i}/{len(runs)}] Transform {sub} {task}")

        X = load_run_matrix(f, mask, center=args.center, zscore=args.zscore)
        S = ipca.transform(X).astype(np.float32)  # (T x K)

        stem = f"{sub}_task-{task}_PCsK{args.n_components}"
        npy_path = os.path.join(tc_dir, stem + ".npy")
        csv_path = os.path.join(tc_dir, stem + ".csv")

        np.save(npy_path, S)
        pd.DataFrame(S, columns=[f"PC{j:02d}" for j in range(1, args.n_components + 1)]).to_csv(csv_path, index=False)

        tc_index_rows.append({
            "subject": sub,
            "task": task,
            "file": f,
            "n_timepoints": int(S.shape[0]),
            "timecourses_npy": npy_path,
            "timecourses_csv": csv_path,
        })

        del X, S
        gc.collect()

    tc_index = pd.DataFrame(tc_index_rows)
    tc_index_path = os.path.join(args.out_root, f"timecourses_index_K{args.n_components}.csv")
    tc_index.to_csv(tc_index_path, index=False)
    print(f"[INFO] Saved timecourse index: {tc_index_path}")

    # 7) Save settings
    settings = {
        "base_root": args.base_root,
        "out_root": args.out_root,
        "tasks": tasks,
        "n_components": args.n_components,
        "batch_size": args.batch_size,
        "center": args.center,
        "zscore": args.zscore,
        "n_runs": len(runs),
        "total_timepoints_fit": int(total_tp),
        "mask_voxels": int(mask.sum()),
    }
    with open(prefix + "_run_settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    print("\nAll done ✅")
    print(f"[DONE] Components NIfTI: {comp_nii}")
    print(f"[DONE] Timecourses dir: {tc_dir}")


if __name__ == "__main__":
    main()
