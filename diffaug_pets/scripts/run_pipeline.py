from __future__ import annotations
import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from ..config import CFG, finalize_cfg
from ..logging_utils import silence_hf_warnings
from ..utils.seed import get_device, seed_everything, cleanup, set_tf32_if_cuda

from ..data.pets import (
    load_raw_pets,
    build_species_by_breed,
    species_token_for_class,
    stratified_split,
    build_transforms,
    PetsSubset,
    PetsMattingSubset,
    make_loader,
)
from ..models.classifier import Classifier
from ..train.baseline import train_one_epoch, evaluate_with_cm, select_target_classes, build_conf_map
from ..diffusion.pipes import load_pipes
from ..aug.augmenter import Augmenter
from ..synth.manifests import (
    load_or_make,
    generate_manifest_inpaint_bg,
    generate_manifest_small_edit,
    generate_manifest_text2img,
)
from ..metrics.clip_filter import FilterCfg, filter_manifest
from ..train.step_matched import run_step_matched


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--work_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--top_n_targets", type=int, default=None)
    p.add_argument("--run_generate", action="store_true", help="Generate synthetic images (diffusion).")
    p.add_argument("--run_filter", action="store_true", help="Filter manifests with CLIP+pHash.")
    p.add_argument("--run_train", action="store_true", help="Run step-matched training comparison.")
    p.add_argument("--all", action="store_true", help="Run generate+filter+train (baseline is always computed).")
    return p.parse_args()


def main():
    args = parse_args()
    silence_hf_warnings()

    cfg = finalize_cfg(CFG())
    if args.data_root: cfg.data_root = args.data_root
    if args.work_dir: cfg.work_dir = args.work_dir
    if args.epochs: cfg.epochs = int(args.epochs)
    if args.top_n_targets: cfg.top_n_targets = int(args.top_n_targets)

    device = get_device()
    set_tf32_if_cuda(device)
    print("DEVICE:", device)

    seed_everything(cfg.seed)

    data_root = Path(cfg.data_root)
    work = Path(cfg.work_dir); work.mkdir(parents=True, exist_ok=True)
    syn_root = work / "synthetic_targeted"; syn_root.mkdir(parents=True, exist_ok=True)
    man_dir = work / "manifests"; man_dir.mkdir(parents=True, exist_ok=True)
    out_dir = work / "results"; out_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    ds_train, ds_test = load_raw_pets(data_root)
    class_names = ds_train.classes
    K = len(class_names)
    cfg.class_names = class_names  # attach for convenience downstream

    breed2species = build_species_by_breed(data_root)
    def species_token_fn(cls_name: str) -> str:
        return species_token_for_class(cls_name, breed2species)

    labels = [y for _, (y, _) in ds_train]
    train_idx, val_idx = stratified_split(labels=torch.tensor(labels).numpy(), seed=cfg.seed, train_frac=0.85)

    train_tf, val_tf = build_transforms(cfg.img_size)

    train_ds = PetsSubset(ds_train, train_idx, tf=train_tf)
    val_ds = PetsSubset(ds_train, val_idx, tf=val_tf)
    test_ds = PetsSubset(ds_test, list(range(len(ds_test))), tf=val_tf)
    mat_ds = PetsMattingSubset(ds_train, train_idx)

    train_dl = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers)
    val_dl = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers)

    print("train/val/test:", len(train_ds), len(val_ds), len(test_ds))

    # --- Baseline classifier ---
    baseline = Classifier(K, backbone=cfg.model_name).to(device)
    opt = torch.optim.AdamW(baseline.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    for ep in range(int(cfg.epochs)):
        tr = train_one_epoch(baseline, train_dl, opt, loss_fn, device)
        va = evaluate_with_cm(baseline, val_dl, K, device)
        print(f"Baseline Epoch {ep+1} | train_loss={tr:.4f} | val_acc={va['acc']:.3f} | val_f1={va['f1_macro']:.3f}")

    va = evaluate_with_cm(baseline, val_dl, K, device)
    print("VAL baseline:", {k: va[k] for k in ["loss", "acc", "f1_macro"]})

    target_classes, errors, err_rate = select_target_classes(va["cm"], top_n=cfg.top_n_targets)
    target_set = set(target_classes)
    conf_map = build_conf_map(va["cm"], per_class=3)

    print("\nTarget classes:")
    for i in target_classes:
        print(f"{i:3d} | {class_names[i]:22s} | errors={int(errors[i])} | err_rate={err_rate[i]:.3f}")

    do_all = bool(args.all)
    do_generate = bool(args.run_generate or do_all)
    do_filter = bool(args.run_filter or do_all)
    do_train = bool(args.run_train or do_all)

    # --- Generate ---
    if do_generate:
        pipes = load_pipes(cfg.t2i_model, cfg.i2i_model, cfg.inpaint_model, device)
        augmenter = Augmenter(cfg=cfg, device=device, class_names=class_names, conf_map=conf_map, species_token_fn=species_token_fn, pipes=pipes)

        df_bg = load_or_make(man_dir / "gen_inpaint_bg.csv", lambda: generate_manifest_inpaint_bg(
            augmenter, mat_ds, class_names, target_classes, syn_root,
            base_seed=cfg.seed, n_seed_images_per_class=cfg.n_seed_images_per_class, n_variations_per_seed=cfg.n_variations_per_seed
        ))
        df_se = load_or_make(man_dir / "gen_small_edit.csv", lambda: generate_manifest_small_edit(
            augmenter, mat_ds, class_names, target_classes, syn_root,
            base_seed=cfg.seed, n_seed_images_per_class=cfg.n_seed_images_per_class, n_variations_per_seed=cfg.n_variations_per_seed
        ))
        df_t2 = load_or_make(man_dir / "gen_text2img.csv", lambda: generate_manifest_text2img(
            augmenter, class_names, target_classes, syn_root,
            base_seed=cfg.seed, n_text2img_per_class=cfg.n_text2img_per_class
        ))
        print("Generated:", {"inpaint_bg": len(df_bg), "small_edit": len(df_se), "text2img": len(df_t2)})
        cleanup(device)
    else:
        df_bg = pd.read_csv(man_dir / "gen_inpaint_bg.csv") if (man_dir / "gen_inpaint_bg.csv").exists() else pd.DataFrame()
        df_se = pd.read_csv(man_dir / "gen_small_edit.csv") if (man_dir / "gen_small_edit.csv").exists() else pd.DataFrame()
        df_t2 = pd.read_csv(man_dir / "gen_text2img.csv") if (man_dir / "gen_text2img.csv").exists() else pd.DataFrame()

    # --- Filter ---
    if do_filter:
        fcfg = FilterCfg(
            min_std=cfg.min_std,
            phash_hamming_thr=cfg.phash_hamming_thr,
            clip_margin_thr=cfg.clip_margin_thr,
            clip_own_min=cfg.clip_own_min,
            keep_per_class=cfg.keep_per_class,
        )

        def _filter_or_load(df, name: str):
            p = man_dir / f"filt_{name}.csv"
            if p.exists():
                return pd.read_csv(p)
            out = filter_manifest(
                df=df,
                device=device,
                class_names=class_names,
                conf_map=conf_map,
                species_token_fn=species_token_fn,
                clip_model_name=cfg.clip_model_name,
                fcfg=fcfg,
                batch_size=16,
            )
            out.to_csv(p, index=False)
            return out

        df_bg_f = _filter_or_load(df_bg, "inpaint_bg")
        df_se_f = _filter_or_load(df_se, "small_edit")
        df_t2_f = _filter_or_load(df_t2, "text2img")
        print("Filtered:", {"inpaint_bg": len(df_bg_f), "small_edit": len(df_se_f), "text2img": len(df_t2_f)})
        cleanup(device)
    else:
        df_bg_f = pd.read_csv(man_dir / "filt_inpaint_bg.csv") if (man_dir / "filt_inpaint_bg.csv").exists() else pd.DataFrame()
        df_se_f = pd.read_csv(man_dir / "filt_small_edit.csv") if (man_dir / "filt_small_edit.csv").exists() else pd.DataFrame()
        df_t2_f = pd.read_csv(man_dir / "filt_text2img.csv") if (man_dir / "filt_text2img.csv").exists() else pd.DataFrame()

    # --- Train step-matched ---
    if do_train:
        # model ctor for fresh init each run
        def model_ctor():
            return Classifier(K, backbone=cfg.model_name)

        results = []

        def _one(method: str, syn_df: pd.DataFrame, title: str):
            seed_everything(cfg.seed)
            teR, teM, steps, n_syn, n_mix = run_step_matched(
                model_ctor=model_ctor,
                train_real_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                train_tf=train_tf,
                syn_df=syn_df,
                cfg=cfg,
                target_set=target_set,
                device=device,
                title=title,
            )
            results.append({
                "method": method, "steps": steps, "n_syn": n_syn, "n_mix": n_mix,
                **{f"real_{k}": v for k, v in teR.items()},
                **{f"mix_{k}": v for k, v in teM.items()},
            })

        _one("inpaint_bg", df_bg_f, "INPAINT_BG")
        _one("small_edit", df_se_f, "SMALL_EDIT")
        _one("text2img", df_t2_f, "TEXT2IMG")

        df_res = pd.DataFrame(results)
        df_res.to_csv(out_dir / "step_matched_results.csv", index=False)
        print("Saved:", out_dir / "step_matched_results.csv")
        print("Synthetic folder:", syn_root)

    print("DONE.")


if __name__ == "__main__":
    main()
