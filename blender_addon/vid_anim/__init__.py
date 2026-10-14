bl_info = {
    "name": "Vid_Anim",
    "author": "Vid_Anim",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > vid_an",
    "description": "Animate any rig from a video or an example animation (retarget + bake)",
    "category": "Animation",
}

import importlib

import bpy

from . import properties, bone_maps, retarget, mocap, operators, panel

_modules = (properties, bone_maps, retarget, mocap, operators, panel)

_classes = []


def _collect_classes():
    return [properties.VidAnProperties] + list(operators.CLASSES) + list(panel.CLASSES)


def register():
    global _classes
    for m in _modules:
        importlib.reload(m)
    classes = _collect_classes()
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.vid_an = bpy.props.PointerProperty(type=properties.VidAnProperties)
    _classes = classes


def unregister():
    global _classes
    if hasattr(bpy.types.Scene, "vid_an"):
        try:
            del bpy.types.Scene.vid_an
        except Exception:
            pass
    for cls in reversed(_classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass
    _classes = []


if __name__ == "__main__":
    register()
