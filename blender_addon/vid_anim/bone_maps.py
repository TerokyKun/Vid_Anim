import re

ROLE_TOKENS = {
    "fk", "ik", "def", "org", "mch", "vis", "ctrl", "control", "deform",
    "tweak", "drv", "driver", "swing", "roll", "master", "parent", "widget",
    "top", "base", "reference", "twist", "pose",
}

SIDE_TOKENS = {
    "l": "L", "left": "L", "lf": "L",
    "r": "R", "right": "R", "rt": "R",
}

ROLE_PENALTY = ("ik", "def", "org", "mch", "vis", "tweak", "drv", "swing", "roll", "widget")


def _split(name):
    n = re.sub(r"mixamorig[:_]?", "", name, flags=re.I)
    n = n.replace("(", " ").replace(")", " ")
    return [t for t in re.split(r"[._\-\s]+", n) if t != ""]


def side_of(name):
    toks = _split(name)
    lowered = [t.lower() for t in toks]
    for t in lowered:
        if t in SIDE_TOKENS:
            return SIDE_TOKENS[t]
    joined = "".join(lowered)
    if joined.startswith("left") or joined.startswith("lf"):
        return "L"
    if joined.startswith("right") or joined.startswith("rt"):
        return "R"
    return None


def _strip_roles(tokens):
    out = []
    for t in tokens:
        lt = t.lower()
        if lt in SIDE_TOKENS or lt in ROLE_TOKENS:
            continue
        out.append(lt)
    return out


def base_of(name):
    toks = _strip_roles(_split(name))
    n = "".join(toks)
    digits = re.findall(r"\d+", n)
    num = ""
    if digits:
        num = str(int(digits[-1]))
    if "hip" in n or "pelvis" in n:
        return "hips"
    if "upleg" in n or "thigh" in n or "upperleg" in n:
        return "thigh"
    if "forearm" in n or "lowerarm" in n:
        return "forearm"
    if "shoulder" in n or "clavicle" in n:
        return "shoulder"
    if "upperarm" in n or "arm" in n:
        return "upper_arm"
    if "hand" in n:
        return "hand"
    if "shin" in n or "calf" in n or "lowerleg" in n or "leg" in n:
        return "shin"
    if "foot" in n or "ankle" in n:
        return "foot"
    if "toe" in n:
        return "toe"
    if "neck" in n:
        return "neck"
    if "head" in n:
        return "head"
    if "tail" in n:
        return "tail" + num
    if "ear" in n:
        return "ear"
    if "spine" in n or "chest" in n or "torso" in n or "abdomen" in n or "spine" in n:
        if "chest" in n:
            return "chest"
        return "spine" + num
    if "root" in n or "torso" in n:
        return "hips"
    return None


def semantic_key(name):
    base = base_of(name)
    if not base:
        return None
    side = side_of(name)
    if side and base in ("shoulder", "upper_arm", "forearm", "hand",
                         "thigh", "shin", "foot", "toe"):
        return base + "." + side
    return base


def _role_penalty(name):
    toks = [t.lower() for t in _split(name)]
    pen = 0
    for p in ("ik", "parent", "root", "swing", "roll", "widget",
              "vis", "drv", "tweak", "mch", "org", "def", "top", "twist", "master"):
        if p in toks:
            pen += 4
    if "fk" in toks:
        pen -= 2
    return pen


def build_semantic_index(armature, prefer_controls=True):
    index = {}
    for bone in armature.data.bones:
        key = semantic_key(bone.name)
        if not key:
            continue
        pen = _role_penalty(bone.name) if prefer_controls else 0
        cur = index.get(key)
        if cur is None or pen < cur[0]:
            index[key] = (pen, bone.name)
    return {k: v[1] for k, v in index.items()}


def auto_pairs(source_arm, target_arm):
    src = build_semantic_index(source_arm, prefer_controls=False)
    tgt = build_semantic_index(target_arm, prefer_controls=True)
    pairs = []
    for sem, sname in sorted(src.items()):
        tname = tgt.get(sem)
        if tname:
            pairs.append((sname, tname, sem))
    return pairs
