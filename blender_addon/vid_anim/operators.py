import os
import re
import subprocess
import sys
import time

import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty

from . import mocap
from . import blender_bridge
from .bone_maps import auto_pairs, semantic_key
from .mocap import build_source_from_json
from .retarget import retarget, iter_retarget as retarget_gen


def _get_props(context):
    return context.scene.vid_an


def _candidate_roots():
    here = os.path.dirname(os.path.abspath(__file__))
    roots = []
    env = os.environ.get("VIDAN_ROOT", "")
    if env:
        roots.append(env)
    roots.append(r"C:\Users\maxkl\Downloads\vid_an")
    roots.append(os.path.normpath(os.path.join(here, "..", "..")))
    roots.append(os.path.normpath(os.path.join(here, "..", "..", "..")))
    return roots


def _candidate_pythons(props):
    py = []
    for root in _candidate_roots():
        for rel in (os.path.join("backend", "venv", "Scripts", "python.exe"),
                    os.path.join("backend", "venv", "bin", "python")):
            cand = os.path.join(root, rel)
            if os.path.exists(cand) and cand not in py:
                py.append(cand)
    if props.backend_python and os.path.exists(props.backend_python):
        if props.backend_python not in py:
            py.append(props.backend_python)
    return py


def _resolve_script(props):
    if props.backend_script and os.path.exists(props.backend_script):
        return props.backend_script
    for root in _candidate_roots():
        cand = os.path.join(root, "backend", "extract_pose.py")
        if os.path.exists(cand):
            return cand
    return ""


def _python_is_valid(py):
    try:
        proc = subprocess.run(
            [py, "-c", "import cv2, mediapipe"],
            capture_output=True, text=True, timeout=60,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _resolve_backend(props):
    """Return (python, script) ready to run the video extractor."""
    script = _resolve_script(props)
    pys = _candidate_pythons(props)
    for py in pys:
        if _python_is_valid(py):
            return py, script
    return (pys[0] if pys else ""), script


def _source_frame_range(obj):
    ad = obj.animation_data
    if not ad or not ad.action:
        return None
    fr = ad.action.frame_range
    return int(fr[0]), int(fr[1])


def _draw_help(self, context):
    col = self.layout.column(align=True)
    lines = [
        "КАК АНИМИРОВАТЬ СВОЙ РИГ:",
        "1. Вставьте видео кнопкой «Загрузить видео».",
        "2. В поле «Цель» выберите свой скелет (риг).",
        "3. Нажмите «Анимировать из видео».",
        "",
        "Меш (модель) не обязателен — анимация",
        "записывается прямо на кости. Меш нужен лишь",
        "чтобы увидеть, как это выглядит.",
        "",
        "ЕСТЬ ГОТОВАЯ АНИМАЦИЯ (FBX/BVH)?",
        "Импортируйте её в блоке «Готовая анимация»,",
        "выберите Цель и нажмите «Перетащить и записать».",
        "",
        "ПЕРВЫЙ ЗАПУСК ВИДЕО:",
        "нужно один раз установить backend —",
        "запустите backend\\setup_windows.bat.",
    ]
    for line in lines:
        col.label(text=line)


class VIDAN_OT_help(bpy.types.Operator):
    bl_idname = "vid_an.help"
    bl_label = "Как пользоваться"
    bl_description = "Подсказка на русском: что куда вставлять"

    def execute(self, context):
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.popup_menu(
            _draw_help, title="vid_an — как пользоваться", icon='QUESTION')
        return {'FINISHED'}


class VIDAN_OT_pick_video(bpy.types.Operator, ImportHelper):
    bl_idname = "vid_an.pick_video"
    bl_label = "Загрузить видео"
    bl_options = {'REGISTER'}

    filename_ext = ""
    filter_glob: StringProperty(
        default="*.mp4;*.mov;*.avi;*.mkv;*.webm;*.m4v;*.mpg;*.mpeg;*.wmv",
        options={'HIDDEN'},
    )

    def execute(self, context):
        props = _get_props(context)
        props.video_path = self.filepath
        self.report({'INFO'}, "Video: " + os.path.basename(self.filepath))
        return {'FINISHED'}


class VIDAN_OT_import_anim(bpy.types.Operator, ImportHelper):
    bl_idname = "vid_an.import_anim"
    bl_label = "Импорт анимации (FBX/BVH)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ""
    filter_glob: StringProperty(default="*.fbx;*.bvh;*.glb;*.gltf;*.dae", options={'HIDDEN'})

    def execute(self, context):
        path = self.filepath
        ext = os.path.splitext(path)[1].lower()
        before = set(bpy.data.objects)
        try:
            if ext == ".fbx":
                bpy.ops.import_scene.fbx(filepath=path)
            elif ext == ".bvh":
                bpy.ops.import_anim.bvh(filepath=path)
            elif ext in (".glb", ".gltf"):
                bpy.ops.import_scene.gltf(filepath=path)
            elif ext == ".dae":
                bpy.ops.import_scene.collada(filepath=path)
            else:
                self.report({'ERROR'}, "Unsupported format: " + ext)
                return {'CANCELLED'}
        except Exception as exc:
            self.report({'ERROR'}, "Import failed: " + str(exc))
            return {'CANCELLED'}

        new_objs = [o for o in bpy.data.objects if o not in before]
        armatures = [o for o in new_objs if o.type == 'ARMATURE']
        props = _get_props(context)
        if armatures:
            props.source_object = armatures[0]
            rng = _source_frame_range(armatures[0])
            if rng and props.auto_range:
                props.frame_start, props.frame_end = rng
        self.report({'INFO'}, "Imported: " + ", ".join(o.name for o in new_objs))
        return {'FINISHED'}


class VIDAN_OT_retarget(bpy.types.Operator):
    bl_idname = "vid_an.retarget"
    bl_label = "Перетащить и записать"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = _get_props(context)
        src = props.source_object
        tgt = props.target_object
        if src is None or tgt is None:
            self.report({'ERROR'}, "Pick both Source Rig and Target Rig")
            return {'CANCELLED'}
        if src == tgt:
            self.report({'ERROR'}, "Source and Target must differ")
            return {'CANCELLED'}

        pairs = auto_pairs(src, tgt)
        if not pairs:
            self.report({'ERROR'}, "Could not match any bones between rigs")
            return {'CANCELLED'}

        fstart, fend = props.frame_start, props.frame_end
        if props.auto_range:
            rng = _source_frame_range(src)
            if rng:
                fstart, fend = rng

        if tgt.animation_data is None:
            tgt.animation_data_create()

        try:
            count = retarget(
                context, src, tgt, pairs,
                frame_start=fstart, frame_end=fend,
                step=props.frame_step,
                yaw_offset_deg=props.yaw_offset,
                loc_scale=props.loc_scale,
                fk_mode=props.fk_mode,
            )
        except Exception as exc:
            self.report({'ERROR'}, "Retarget failed: " + str(exc))
            return {'CANCELLED'}

        self.report({'INFO'}, "Retargeted {} bones over {} frames".format(len(pairs), count))
        return {'FINISHED'}


class VIDAN_OT_clear_target(bpy.types.Operator):
    bl_idname = "vid_an.clear_target"
    bl_label = "Очистить анимацию цели"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        tgt = _get_props(context).target_object
        if tgt is None:
            self.report({'ERROR'}, "Pick a Target Rig")
            return {'CANCELLED'}
        ad = tgt.animation_data
        if ad and ad.action:
            act = ad.action
            ad.action = None
            bpy.data.actions.remove(act)
        for pb in tgt.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
        self.report({'INFO'}, "Target animation cleared")
        return {'FINISHED'}


class VIDAN_OT_preview_pairs(bpy.types.Operator):
    bl_idname = "vid_an.preview_pairs"
    bl_label = "Показать сопоставление костей"

    def execute(self, context):
        props = _get_props(context)
        src, tgt = props.source_object, props.target_object
        if src is None or tgt is None:
            self.report({'ERROR'}, "Pick both rigs")
            return {'CANCELLED'}
        pairs = auto_pairs(src, tgt)
        lines = ["{}  ->  {}   [{}]".format(s, t, sem) for (s, t, sem) in pairs]
        print("\n[vid_an] Bone mapping ({} pairs):".format(len(pairs)))
        for line in lines:
            print("   " + line)
        self.report({'INFO'}, "Matched {} bones (see console)".format(len(pairs)))
        return {'FINISHED'}


class VIDAN_OT_import_pose_json(bpy.types.Operator, ImportHelper):
    bl_idname = "vid_an.import_pose_json"
    bl_label = "Скелет из pose JSON"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        props = _get_props(context)
        try:
            arm = build_source_from_json(context, self.filepath,
                                         fill_partial=props.fill_partial_body)
        except Exception as exc:
            self.report({'ERROR'}, "Build failed: " + str(exc))
            return {'CANCELLED'}
        props.source_object = arm
        rng = _source_frame_range(arm)
        if rng and props.auto_range:
            props.frame_start, props.frame_end = rng
        self.report({'INFO'}, "Built source rig: " + arm.name)
        return {'FINISHED'}


class VIDAN_OT_extract_motion(bpy.types.Operator):
    bl_idname = "vid_an.extract_motion"
    bl_label = "Анимировать из видео"
    bl_options = {'REGISTER'}

    _timer = None
    _proc = None
    _log_path = None
    _log_fh = None
    _out_json = None
    _phase = None
    _gen = None
    _post_props = None
    _arm = None

    def _cleanup(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        try:
            context.workspace.status_text_set(None)
        except Exception:
            pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    def invoke(self, context, event):
        props = _get_props(context)
        if not props.video_path:
            self.report({'ERROR'}, "Сначала выберите видеофайл")
            return {'CANCELLED'}

        backend_py, script = _resolve_backend(props)
        if not backend_py or not script:
            self.report({'ERROR'},
                        "Backend не найден. Запустите backend\\setup_windows.bat "
                        "или укажите пути в «Дополнительно»")
            return {'CANCELLED'}

        video = bpy.path.abspath(props.video_path)
        self._out_json = os.path.splitext(video)[0] + "_pose.json"
        self._log_path = self._out_json + ".log"
        self._phase = None
        self._gen = None
        self._arm = None
        cmd = [backend_py, script, video, self._out_json,
               "--model", props.model_kind, "--fps", str(props.video_fps),
               "--step", str(props.video_step), "--resize", str(props.video_resize)]
        try:
            self._log_fh = open(self._log_path, "w", encoding="utf-8", errors="ignore")
            self._proc = subprocess.Popen(cmd, stdout=self._log_fh,
                                          stderr=subprocess.STDOUT)
        except Exception as exc:
            self._cleanup(context)
            self.report({'ERROR'}, "Не удалось запустить backend: " + str(exc))
            return {'CANCELLED'}

        self._timer = context.window_manager.event_timer_add(0.4, window=context.window)
        context.window_manager.modal_handler_add(self)
        try:
            context.window_manager.progress_begin(0, 100)
            context.window_manager.progress_update(0)
        except Exception:
            pass
        try:
            context.workspace.status_text_set("vid_an: запуск backend...")
        except Exception:
            pass
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._terminate()
            self._cleanup(context)
            self.report({'WARNING'}, "Отменено")
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}

        if self._phase is None:
            rc = self._proc.poll()
            if rc is None:
                self._update_status(context)
                return {'RUNNING_MODAL'}
            if rc != 0:
                self._cleanup(context)
                self.report({'ERROR'}, "Ошибка backend: " + self._read_log_tail())
                return {'CANCELLED'}
            self._close_proc()
            self._start_build(context)
            return {'RUNNING_MODAL'}

        return self._drive_post(context)

    def _close_proc(self):
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self._proc = None

    def _terminate(self):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _read_log(self):
        try:
            with open(self._log_path, "r", encoding="utf-8", errors="ignore") as fh:
                return fh.read()
        except Exception:
            return ""

    def _read_log_tail(self):
        return self._read_log()[-300:]

    def _set_progress(self, context, pct, txt):
        try:
            context.workspace.status_text_set(txt)
        except Exception:
            pass
        if pct is not None:
            try:
                context.window_manager.progress_update(int(pct))
            except Exception:
                pass

    def _update_status(self, context):
        data = self._read_log()
        m = re.findall(r"frame (\d+)/(\d+)", data)
        pct = None
        if "downloading model" in data:
            txt = "vid_an: скачивание модели ИИ (один раз)..."
        elif m:
            i, t = m[-1]
            txt = "vid_an: распознавание кадр {} / {}   (Esc — отмена)".format(i, t if t != "0" else "?")
            if t not in ("", "0"):
                pct = max(0, min(100, int(50.0 * int(i) / int(t))))
        else:
            txt = "vid_an: подготовка backend..."
        # Backend occupies the first half of the bar; baking the second half.
        self._set_progress(context, pct, txt)

    def _start_build(self, context):
        self._post_props = _get_props(context)
        self._phase = 'build'
        self._gen = mocap.iter_build_source_from_json(
            context, self._out_json,
            fill_partial=self._post_props.fill_partial_body)
        self._set_progress(context, 50, "vid_an: строю скелет персонажа...")

    def _retarget_gen(self, context):
        props = self._post_props
        src = props.source_object
        tgt = props.target_object
        pairs = auto_pairs(src, tgt)
        return retarget_gen(
            context, src, tgt, pairs,
            props.frame_start, props.frame_end, step=props.frame_step,
            yaw_offset_deg=props.yaw_offset, loc_scale=props.loc_scale,
            fk_mode=props.fk_mode,
        )

    def _drive_post(self, context):
        deadline = time.time() + 0.03
        while time.time() < deadline:
            try:
                done, total = next(self._gen)
            except StopIteration as stop:
                if self._phase == 'build':
                    self._arm = stop.value
                    props = self._post_props
                    if self._arm is not None:
                        props.source_object = self._arm
                        props.pose_json = self._out_json
                        rng = _source_frame_range(self._arm)
                        if rng and props.auto_range:
                            props.frame_start, props.frame_end = rng
                    if props.target_object is not None:
                        self._phase = 'retarget'
                        self._gen = self._retarget_gen(context)
                        self._set_progress(context, 50, "vid_an: переношу движение...")
                        continue
                    return self._finish_ok(context)
                return self._finish_ok(context)
            except Exception as exc:
                self._cleanup(context)
                self.report({'ERROR'}, "Ошибка обработки: " + str(exc))
                return {'CANCELLED'}
            else:
                frac = (float(done) / total) if total else 0.0
                if self._phase == 'build':
                    self._set_progress(context, 50 * frac,
                                       "vid_an: скелет кадр {} / {}".format(done, total))
                else:
                    self._set_progress(context, 50 + 50 * frac,
                                       "vid_an: перенос кадр {} / {}".format(done, total))
        return {'RUNNING_MODAL'}

    def _finish_ok(self, context):
        self._cleanup(context)
        self.report({'INFO'}, "Готово: движение перенесено")
        return {'FINISHED'}

    def execute(self, context):
        return self.invoke(context, None)


class VIDAN_OT_export_bvh(bpy.types.Operator, ExportHelper):
    bl_idname = "vid_an.export_bvh"
    bl_label = "Экспорт BVH"
    bl_options = {'REGISTER'}

    filename_ext = ".bvh"
    filter_glob: StringProperty(default="*.bvh", options={'HIDDEN'})

    def execute(self, context):
        try:
            blender_bridge.export_bvh(context, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, "BVH export failed: " + str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported BVH: " + os.path.basename(self.filepath))
        return {'FINISHED'}


class VIDAN_OT_export_fbx(bpy.types.Operator, ExportHelper):
    bl_idname = "vid_an.export_fbx"
    bl_label = "Экспорт FBX"
    bl_options = {'REGISTER'}

    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx", options={'HIDDEN'})

    def execute(self, context):
        try:
            blender_bridge.export_fbx(context, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, "FBX export failed: " + str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported FBX: " + os.path.basename(self.filepath))
        return {'FINISHED'}


class VIDAN_OT_export_glb(bpy.types.Operator, ExportHelper):
    bl_idname = "vid_an.export_glb"
    bl_label = "Экспорт GLB (Godot)"
    bl_options = {'REGISTER'}

    filename_ext = ".glb"
    filter_glob: StringProperty(default="*.glb", options={'HIDDEN'})

    def execute(self, context):
        try:
            blender_bridge.export_glb(context, self.filepath)
        except Exception as exc:
            self.report({'ERROR'}, "GLB export failed: " + str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported GLB: " + os.path.basename(self.filepath))
        return {'FINISHED'}


class VIDAN_OT_export_arrays(bpy.types.Operator, ExportHelper):
    bl_idname = "vid_an.export_arrays"
    bl_label = "Экспорт координат (NPZ/JSONL/CSV)"
    bl_options = {'REGISTER'}

    filename_ext = ".npz"
    filter_glob: StringProperty(default="*.npz;*.jsonl;*.csv", options={'HIDDEN'})

    def execute(self, context):
        fmt = os.path.splitext(self.filepath)[1].lstrip(".").lower() or "npz"
        try:
            blender_bridge.export_arrays_to_core(context, None, self.filepath, fmt)
        except Exception as exc:
            self.report({'ERROR'}, "Array export failed: " + str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported arrays: " + os.path.basename(self.filepath))
        return {'FINISHED'}


class VIDAN_OT_import_session(bpy.types.Operator, ImportHelper):
    bl_idname = "vid_an.import_session"
    bl_label = "Импорт сессии mocap (JSONL/NPZ/JSON)"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".jsonl"
    filter_glob: StringProperty(
        default="*.jsonl;*.ndjson;*.npz;*.json", options={'HIDDEN'})

    def execute(self, context):
        props = _get_props(context)
        try:
            legacy = blender_bridge.session_to_legacy_json(self.filepath)
            arm = build_source_from_json(context, legacy,
                                         fill_partial=props.fill_partial_body)
        except Exception as exc:
            self.report({'ERROR'}, "Session import failed: " + str(exc))
            return {'CANCELLED'}
        props.source_object = arm
        props.pose_json = legacy
        rng = _source_frame_range(arm)
        if rng and props.auto_range:
            props.frame_start, props.frame_end = rng
        self.report({'INFO'}, "Built source rig from session: " + arm.name)
        return {'FINISHED'}


CLASSES = (
    VIDAN_OT_help,
    VIDAN_OT_pick_video,
    VIDAN_OT_import_anim,
    VIDAN_OT_import_pose_json,
    VIDAN_OT_import_session,
    VIDAN_OT_retarget,
    VIDAN_OT_clear_target,
    VIDAN_OT_preview_pairs,
    VIDAN_OT_extract_motion,
    VIDAN_OT_export_bvh,
    VIDAN_OT_export_fbx,
    VIDAN_OT_export_glb,
    VIDAN_OT_export_arrays,
)
