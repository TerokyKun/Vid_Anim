import os

import bpy

from .operators import _candidate_pythons, _resolve_script


class VIDAN_PT_main(bpy.types.Panel):
    bl_label = "vid_an — анимация по видео"
    bl_idname = "VIDAN_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "vid_an"

    def draw(self, context):
        layout = self.layout
        props = context.scene.vid_an

        top = layout.row(align=True)
        top.label(text="Как это работает:", icon='QUESTION')
        top.operator("vid_an.help", text="Подсказка", icon='INFO')

        box = layout.box()
        box.label(text="1. Ваш риг (Цель)", icon='ARMATURE_DATA')
        box.prop(props, "target_object", text="")
        if props.target_object is None:
            box.label(text="выберите скелет (кости), не меш", icon='INFO')

        box = layout.box()
        box.label(text="2. Видео", icon='CAMERA_DATA')
        box.operator("vid_an.pick_video", icon='FILE_MOVIE')
        if props.video_path:
            box.label(text=os.path.basename(props.video_path), icon='CHECKMARK')
        else:
            box.label(text="видео не выбрано", icon='X')
        box.prop(props, "fill_partial_body")
        row = box.row()
        row.scale_y = 1.6
        row.operator("vid_an.extract_motion", icon='SEQUENCE')

        py, script = _candidate_pythons(props), _resolve_script(props)
        if py and script:
            box.label(text="backend готов", icon='CHECKMARK')
        else:
            box.label(text="backend не найден (см. Дополнительно)", icon='ERROR')

        adv = layout.box()
        icon = 'TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT'
        adv.prop(props, "show_advanced", icon=icon, emboss=False)
        if props.show_advanced:
            adv.prop(props, "video_fps")
            adv.prop(props, "model_kind")
            adv.label(text="Скорость распознавания:")
            adv.prop(props, "video_step")
            adv.prop(props, "video_resize")
            adv.prop(props, "backend_python", text="Python")
            adv.prop(props, "backend_script", text="Скрипт")
            adv.separator()
            adv.label(text="Готовая анимация (FBX/BVH/GLB):")
            adv.operator("vid_an.import_anim", icon='FILE_NEW')
            adv.operator("vid_an.import_pose_json", icon='OUTLINER_OB_ARMATURE')
            adv.operator("vid_an.import_session", icon='IMPORT')
            adv.prop(props, "source_object", text="Источник")
            adv.operator("vid_an.preview_pairs", icon='ZOOM_ALL')
            adv.operator("vid_an.retarget", icon='PLAY')
            adv.operator("vid_an.clear_target", icon='TRASH')
            adv.separator()
            adv.label(text="Экспорт результата:")
            row = adv.row(align=True)
            row.operator("vid_an.export_bvh", icon='EXPORT')
            row.operator("vid_an.export_fbx", icon='EXPORT')
            row.operator("vid_an.export_glb", icon='EXPORT')
            adv.operator("vid_an.export_arrays", icon='EXPORT')
            adv.separator()
            adv.label(text="Настройки переноса:")
            adv.prop(props, "yaw_offset")
            adv.prop(props, "loc_scale")
            adv.prop(props, "fk_mode")
            adv.prop(props, "auto_range")
            adv.prop(props, "frame_step")


CLASSES = (VIDAN_PT_main,)
