from __future__ import annotations

"""图像侧车 EXIF / XMP 轻量读取（与生成管线、模型族无关）。"""

import json
import logging
from pathlib import Path

import PIL.Image
import piexif

log = logging.getLogger(__name__)


class MetadataReader:
    @staticmethod
    def read_exif_metadata(image_path: str | Path) -> dict | None:
        try:
            with PIL.Image.open(image_path) as img:
                exif_bytes = img.info.get("exif")

                if not exif_bytes:
                    return None

                exif_dict = piexif.load(exif_bytes)
                user_comment = exif_dict["Exif"].get(0x9286, b"")

                if user_comment:
                    if user_comment.startswith(b"ASCII\x00\x00\x00"):
                        metadata_str = user_comment[8:].decode("utf-8")
                    else:
                        metadata_str = user_comment.decode("utf-8")
                    return json.loads(metadata_str)

                return None

        except (OSError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            log.debug("Error reading EXIF metadata: %s", e)
            return None

    @staticmethod
    def read_xmp_metadata(image_path: str | Path) -> dict | None:
        try:
            with PIL.Image.open(image_path) as img:
                xmp_data = img.info.get("XML:com.adobe.xmp")

                if not xmp_data:
                    return None

                xmp_dict: dict[str, str] = {}

                fields = {
                    "description": '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">',
                    "creator": "<dc:creator><rdf:Seq><rdf:li>",
                    "rights": '<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">',
                    "creator_tool": "<xmp:CreatorTool>",
                    "category": "<photoshop:Category>",
                    "credit": "<photoshop:Credit>",
                    "seed": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":seed>",
                    "steps": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":steps>",
                    "guidance": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":guidance>",
                    "model": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":model>",
                    "loras": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":loras>",
                    "generation_time": "<" + bytes([109, 102, 108, 117, 120]).decode("ascii") + ":generationTime>",
                }

                for key, start_tag in fields.items():
                    if start_tag in xmp_data:
                        start_idx = xmp_data.index(start_tag) + len(start_tag)
                        if key in ("description", "rights", "creator"):
                            end_tag = "</rdf:li>"
                        else:
                            tag_name = start_tag.split(":")[1].rstrip(">")
                            end_tag = f"</{start_tag.split(':')[0]}:{tag_name}>"

                        end_idx = xmp_data.index(end_tag, start_idx)
                        value = xmp_data[start_idx:end_idx]
                        xmp_dict[key] = value

                return xmp_dict if xmp_dict else None

        except (OSError, KeyError, ValueError) as e:
            log.debug("Error reading XMP metadata: %s", e)
            return None

    @staticmethod
    def read_all_metadata(image_path: str | Path) -> dict:
        return {
            "exif": MetadataReader.read_exif_metadata(image_path),
            "xmp": MetadataReader.read_xmp_metadata(image_path),
        }
