from __future__ import annotations
import random
from typing import Optional, Dict, List, Callable

NEG_BASE = (
    "human, person, people, face, hands, body, skin, "
    "text, watermark, logo, signature, "
    "low quality, blurry, out of focus, distorted, deformed, bad anatomy, "
    "extra limbs, extra fingers, mutated, "
    "cartoon, painting, illustration, toy"
)

SCENES = [
    "in a cozy living room", "in a sunny garden", "on a grassy field",
    "in a snowy park", "in a modern apartment", "in a forest"
]

BG_LIST = [
    "snowy forest background, winter light",
    "beach background, bright sunlight",
    "city street background, evening neon lights",
    "mountain landscape background, golden hour",
    "indoor studio background, softbox lighting",
    "sunny garden background, shallow depth of field",
]


def class_prompt(class_name: str, species_token_fn: Callable[[str], str], scene: Optional[str] = None) -> str:
    breed = class_name.replace("_", " ")
    sp = species_token_fn(class_name)
    scene = scene or random.choice(SCENES)
    return (
        f"a high-quality realistic photo of a {breed} {sp} {scene}, "
        f"full body, sharp focus, natural lighting, realistic fur"
    )


def neg_with_confusables(class_idx: int, class_names: List[str], conf_map: Dict[int, List[int]]) -> str:
    bad = [NEG_BASE]
    confs = conf_map.get(int(class_idx), [])
    if confs:
        bad.append(", ".join([f"not a {class_names[j].replace('_',' ')}" for j in confs]))
    return ", ".join(bad)


def safe_inpaint_prompt(class_name: str, species_token_fn: Callable[[str], str], bg: str) -> str:
    cls = class_name.replace("_", " ")
    sp = species_token_fn(class_name)
    return f"a photo of a {sp} {cls}, same animal, same pose, {bg}, realistic, high quality, background replaced"
