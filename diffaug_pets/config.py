from dataclasses import dataclass

@dataclass
class CFG:
    seed: int = 42

    data_root: str = "/content/data"
    work_dir: str = "/content/work"
    img_size: int = 224
    batch_size: int = 64
    num_workers: int = 0  # Colab-friendly

    epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    model_name: str = "resnet18"

    t2i_model: str = "runwayml/stable-diffusion-v1-5"
    i2i_model: str = "runwayml/stable-diffusion-v1-5"
    inpaint_model: str = "runwayml/stable-diffusion-inpainting"

    gen_size: int = 384  # will be made %8
    guidance: float = 7.5
    steps_inpaint: int = 24
    steps_img2img: int = 16
    steps_t2i: int = 28

    strength_bg_inpaint: float = 0.75
    strength_small_edits: float = 0.22

    top_n_targets: int = 10
    n_seed_images_per_class: int = 2
    n_variations_per_seed: int = 3
    n_text2img_per_class: int = 6

    clip_model_name: str = "openai/clip-vit-base-patch16"
    min_std: float = 8.0
    phash_hamming_thr: int = 6
    clip_margin_thr: float = 0.05
    clip_own_min: float = 0.23
    keep_per_class: int = 40

def finalize_cfg(cfg: CFG) -> CFG:
    cfg.gen_size = int(cfg.gen_size) - (int(cfg.gen_size) % 8)
    return cfg
