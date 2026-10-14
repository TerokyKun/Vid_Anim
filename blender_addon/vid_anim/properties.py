import bpy


def _poll_armature(self, obj):
    return obj is not None and obj.type == 'ARMATURE'


class VidAnProperties(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(
        name="Источник",
        description="Риг (арматура) с готовой анимацией-примером.\n"
                    "Сюда ставится скелет, импортированный из FBX/BVH,\n"
                    "или скелет, построенный из видео.",
        type=bpy.types.Object,
        poll=_poll_armature,
    )
    target_object: bpy.props.PointerProperty(
        name="Цель",
        description="Ваш риг (арматура), на который переносим анимацию.\n"
                    "Ставьте сюда СКЕЛЕТ (кости), не меш.\n"
                    "Меш нужен только чтобы увидеть результат — сама\n"
                    "анимация костям пишется и без него.",
        type=bpy.types.Object,
        poll=_poll_armature,
    )
    frame_start: bpy.props.IntProperty(name="Кадр начала", default=1, min=1)
    frame_end: bpy.props.IntProperty(name="Кадр конца", default=250, min=1)
    frame_step: bpy.props.IntProperty(
        name="Шаг кадров",
        description="Брать каждый N-й кадр (1 = все кадры)",
        default=1, min=1, max=10,
    )
    auto_range: bpy.props.BoolProperty(
        name="Диапазон автоматически",
        description="Взять длительность из исходной анимации/видео",
        default=True,
    )
    yaw_offset: bpy.props.FloatProperty(
        name="Поворот лица",
        description="Развернуть источник вокруг вертикали (Z), если он смотрит\n"
                    "не в ту сторону, что цель. Обычно 0 или 180.",
        default=0.0, min=-180.0, max=180.0,
    )
    loc_scale: bpy.props.FloatProperty(
        name="Масштаб корня",
        description="Масштаб перемещения корневой кости (если габариты\n"
                    "источника и цели сильно различаются)",
        default=1.0, min=0.001, max=100.0,
    )
    fk_mode: bpy.props.BoolProperty(
        name="Управлять только FK",
        description="Отключить IK на конечностях цели, чтобы записанные\n"
                    "FK-ключи реально гнули кости (нужно для Rigify).",
        default=True,
    )
    show_advanced: bpy.props.BoolProperty(
        name="Дополнительно",
        description="Показать технические настройки",
        default=False,
    )

    video_path: bpy.props.StringProperty(
        name="Видео",
        description="Файл видео, из которого извлечь движение",
        subtype='FILE_PATH',
        default="",
    )
    video_fps: bpy.props.FloatProperty(
        name="Частота кадров видео",
        description="FPS видео, если не определяется автоматически",
        default=30.0, min=1.0, max=240.0,
    )
    video_step: bpy.props.IntProperty(
        name="Обрабатывать каждый N-й кадр",
        description="Для ускорения: распознавать позу не в каждом кадре,\n"
                    "а через N (промежуточные кадры Blender достроит сам).\n"
                    "1 = все кадры (точно), 2-3 = быстрее в 2-3 раза.",
        default=1, min=1, max=10,
    )
    video_resize: bpy.props.IntProperty(
        name="Уменьшать кадр до (px)",
        description="Перед распознаванием уменьшить кадр так, чтобы большая\n"
                    "сторона была не больше N пикселей. 0 = не уменьшать.\n"
                    "Меньше = быстрее, но чуть менее точно.",
        default=0, min=0, max=4096,
    )
    pose_json: bpy.props.StringProperty(
        name="Pose JSON",
        description="JSON с 3D-точками скелета, созданный backend-ом",
        subtype='FILE_PATH',
        default="",
    )
    model_kind: bpy.props.EnumProperty(
        name="Модель",
        description="Качество модели MediaPipe (чем выше — тем точнее и медленнее)",
        items=[
            ("lite", "Лёгкая", "Быстро, но менее точно"),
            ("full", "Обычная", "Баланс (рекомендуется)"),
            ("heavy", "Точная", "Медленно, но точнее всего"),
        ],
        default="full",
    )
    backend_python: bpy.props.StringProperty(
        name="Python backend",
        description="Интерпретатор Python из окружения backend (venv).\n"
                    "Можно оставить пустым — будет найден автоматически,\n"
                    "если ранее запускался backend\\setup_windows.bat.",
        subtype='FILE_PATH',
        default="",
    )
    backend_script: bpy.props.StringProperty(
        name="Скрипт извлечения",
        description="Путь к backend/extract_pose.py (пусто = авто)",
        subtype='FILE_PATH',
        default="",
    )
