"""Mapping between the canonical capture rig (18 bones) and common target rigs.

The add-on's own semantic matcher lives in ``addon.bone_maps``; this module
provides the same knowledge to the standalone/core pipeline and to the backend,
plus ready-made maps for popular rigs (Mixamo, Rigify, UE4/UE5, Unity Humanoid).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

# Canonical capture-rig bone names (match addon mocap.BONES).
CANONICAL_BONES: List[str] = [
    "Hips", "Spine", "Neck", "Head",
    "Shoulder.L", "UpperArm.L", "ForeArm.L", "Hand.L",
    "Shoulder.R", "UpperArm.R", "ForeArm.R", "Hand.R",
    "Thigh.L", "Shin.L", "Foot.L",
    "Thigh.R", "Shin.R", "Foot.R",
]

# Semantic slots -> candidate substrings, used for fuzzy matching.
SEMANTIC: Dict[str, List[str]] = {
    "Hips": ["hip", "pelvis", "root"],
    "Spine": ["spine", "spine1", "chest", "abdomen"],
    "Neck": ["neck"],
    "Head": ["head"],
    "Shoulder.L": ["shoulder.l", "clavicle.l", "shoulderleft", "l_shoulder", "shoulderl"],
    "UpperArm.L": ["upperarm.l", "uparm.l", "arm.l", "l_upperarm", "leftarm"],
    "ForeArm.L": ["forearm.l", "lowerarm.l", "l_forearm", "leftforearm"],
    "Hand.L": ["hand.l", "l_hand", "lefthand", "wrist.l"],
    "Shoulder.R": ["shoulder.r", "clavicle.r", "shoulderright", "r_shoulder", "shoulderr"],
    "UpperArm.R": ["upperarm.r", "uparm.r", "arm.r", "r_upperarm", "rightarm"],
    "ForeArm.R": ["forearm.r", "lowerarm.r", "r_forearm", "rightforearm"],
    "Hand.R": ["hand.r", "r_hand", "righthand", "wrist.r"],
    "Thigh.L": ["thigh.l", "upleg.l", "leg.l", "l_thigh", "leftupleg", "leftthigh"],
    "Shin.L": ["shin.l", "calf.l", "lowerleg.l", "l_shin", "leftleg", "leftshin"],
    "Foot.L": ["foot.l", "ankle.l", "l_foot", "leftfoot"],
    "Thigh.R": ["thigh.r", "upleg.r", "leg.r", "r_thigh", "rightupleg", "rightthigh"],
    "Shin.R": ["shin.r", "calf.r", "lowerleg.r", "r_shin", "rightleg", "rightshin"],
    "Foot.R": ["foot.r", "ankle.r", "r_foot", "rightfoot"],
}

MIXAMO_MAP: Dict[str, str] = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Neck": "Neck",
    "Head": "Head",
    "Shoulder.L": "LeftShoulder",
    "UpperArm.L": "LeftArm",
    "ForeArm.L": "LeftForeArm",
    "Hand.L": "LeftHand",
    "Shoulder.R": "RightShoulder",
    "UpperArm.R": "RightArm",
    "ForeArm.R": "RightForeArm",
    "Hand.R": "RightHand",
    "Thigh.L": "LeftUpLeg",
    "Shin.L": "LeftLeg",
    "Foot.L": "LeftFoot",
    "Thigh.R": "RightUpLeg",
    "Shin.R": "RightLeg",
    "Foot.R": "RightFoot",
}

RIGIFY_MAP: Dict[str, str] = {
    "Hips": "spine",
    "Spine": "spine.001",
    "Neck": "spine.004",
    "Head": "spine.006",
    "Shoulder.L": "shoulder.L",
    "UpperArm.L": "upper_arm.L",
    "ForeArm.L": "forearm.L",
    "Hand.L": "hand.L",
    "Shoulder.R": "shoulder.R",
    "UpperArm.R": "upper_arm.R",
    "ForeArm.R": "forearm.R",
    "Hand.R": "hand.R",
    "Thigh.L": "thigh.L",
    "Shin.L": "shin.L",
    "Foot.L": "foot.L",
    "Thigh.R": "thigh.R",
    "Shin.R": "shin.R",
    "Foot.R": "foot.R",
}

UE_MAP: Dict[str, str] = {
    "Hips": "pelvis",
    "Spine": "spine_01",
    "Neck": "neck_01",
    "Head": "head",
    "Shoulder.L": "clavicle_l",
    "UpperArm.L": "upperarm_l",
    "ForeArm.L": "lowerarm_l",
    "Hand.L": "hand_l",
    "Shoulder.R": "clavicle_r",
    "UpperArm.R": "upperarm_r",
    "ForeArm.R": "lowerarm_r",
    "Hand.R": "hand_r",
    "Thigh.L": "thigh_l",
    "Shin.L": "calf_l",
    "Foot.L": "foot_l",
    "Thigh.R": "thigh_r",
    "Shin.R": "calf_r",
    "Foot.R": "foot_r",
}

UNITY_MAP: Dict[str, str] = {
    "Hips": "Hips",
    "Spine": "Spine",
    "Neck": "Neck",
    "Head": "Head",
    "Shoulder.L": "LeftShoulder",
    "UpperArm.L": "LeftUpperArm",
    "ForeArm.L": "LeftLowerArm",
    "Hand.L": "LeftHand",
    "Shoulder.R": "RightShoulder",
    "UpperArm.R": "RightUpperArm",
    "ForeArm.R": "RightLowerArm",
    "Hand.R": "RightHand",
    "Thigh.L": "LeftUpperLeg",
    "Shin.L": "LeftLowerLeg",
    "Foot.L": "LeftFoot",
    "Thigh.R": "RightUpperLeg",
    "Shin.R": "RightLowerLeg",
    "Foot.R": "RightFoot",
}

PRESETS: Dict[str, Dict[str, str]] = {
    "mixamo": MIXAMO_MAP,
    "rigify": RIGIFY_MAP,
    "ue": UE_MAP,
    "unity": UNITY_MAP,
}


def _norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def semantic_key(bone_name: str) -> Optional[str]:
    """Return the canonical slot a target bone most likely corresponds to."""
    n = _norm(bone_name)
    best: Optional[str] = None
    best_len = 0
    for slot, needles in SEMANTIC.items():
        for needle in needles:
            cand = _norm(needle)
            if cand and cand in n and len(cand) > best_len:
                best, best_len = slot, len(cand)
    return best


def auto_pairs(target_bones: Iterable[str],
               source_bones: Optional[Iterable[str]] = None) -> List[Tuple[str, str]]:
    """Map canonical source bones -> target bones by semantic matching."""
    source_bones = list(source_bones or CANONICAL_BONES)
    by_slot: Dict[str, str] = {}
    for tb in target_bones:
        slot = semantic_key(tb)
        if slot and slot not in by_slot:
            by_slot[slot] = tb
    pairs: List[Tuple[str, str]] = []
    for sb in source_bones:
        tb = by_slot.get(sb)
        if tb:
            pairs.append((sb, tb))
    return pairs


def preset_map(preset: str) -> Dict[str, str]:
    p = PRESETS.get(preset.lower())
    if p is None:
        raise KeyError("Unknown rig preset: {!r} (have {})".format(
            preset, ", ".join(sorted(PRESETS))))
    return dict(p)


def detect_preset(target_bones: Iterable[str]) -> Optional[str]:
    names = set(target_bones)
    best, best_score = None, 0
    for key, mapping in PRESETS.items():
        score = sum(1 for v in mapping.values() if v in names)
        if score > best_score:
            best, best_score = key, score
    return best if best_score >= 6 else None
