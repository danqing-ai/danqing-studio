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


def is_image_kind(kind: str) -> bool:
    return kind.startswith("image.")


def is_video_kind(kind: str) -> bool:
    return kind.startswith("video.")


def is_audio_kind(kind: str) -> bool:
    return kind.startswith("audio.")
