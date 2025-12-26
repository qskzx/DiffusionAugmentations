import warnings

def silence_hf_warnings():
    warnings.filterwarnings("ignore", message=r".*Flax classes are deprecated.*")
    warnings.filterwarnings("ignore", message=r".*enable_vae_slicing.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*enable_vae_tiling.*deprecated.*")
    warnings.filterwarnings("ignore", message=r".*HF_TOKEN.*does not exist.*")
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        from diffusers.utils import logging as diff_log
        diff_log.set_verbosity_error()
    except Exception:
        pass
    try:
        from transformers.utils import logging as tr_log
        tr_log.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_log
        hub_log.set_verbosity_error()
    except Exception:
        pass
