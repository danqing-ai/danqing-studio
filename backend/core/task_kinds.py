"""集中定义 v3 任务 kind — Plan §13；禁止魔法字符串散落。"""

IMAGE_GENERATION = "image.generation"
IMAGE_EDIT = "image.edit"
IMAGE_UPSCALE = "image.upscale"
VIDEO_GENERATION = "video.generation"
VIDEO_EDIT = "video.edit"
AUDIO_GENERATION = "audio.generation"
AUDIO_EDIT = "audio.edit"

ALL_KINDS: frozenset[str] = frozenset(
    {
        IMAGE_GENERATION,
        IMAGE_EDIT,
        IMAGE_UPSCALE,
        VIDEO_GENERATION,
        VIDEO_EDIT,
        AUDIO_GENERATION,
        AUDIO_EDIT,
    }
)

# Registry ``actions`` keys → scheduler / API task ``kind`` strings.
REGISTRY_ACTION_TO_TASK_KIND: dict[str, dict[str, str]] = {
    "image": {
        "create": IMAGE_GENERATION,
        "rewrite": IMAGE_EDIT,
        "retouch": IMAGE_EDIT,
        "extend": IMAGE_EDIT,
        "upscale": IMAGE_UPSCALE,
    },
    "video": {
        "create": VIDEO_GENERATION,
        "animate": VIDEO_EDIT,
    },
    "audio": {
        "create": AUDIO_GENERATION,
        "cover": AUDIO_EDIT,
        "repaint": AUDIO_EDIT,
    },
}


def task_kind_for_registry_action(media: str, action: str) -> str | None:
    """Map ``models_registry.json`` action name to ``task_kinds`` constant."""
    return REGISTRY_ACTION_TO_TASK_KIND.get(media, {}).get(action)


def is_image_kind(kind: str) -> bool:
    return kind.startswith("image.")


def is_video_kind(kind: str) -> bool:
    return kind.startswith("video.")


def is_audio_kind(kind: str) -> bool:
    return kind.startswith("audio.")
