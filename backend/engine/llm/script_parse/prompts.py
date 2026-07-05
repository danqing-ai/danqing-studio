"""System prompts for script_parse 4-pass pipeline."""
from __future__ import annotations

from backend.engine.llm.prompts.system import (
    _JSON_OUTPUT_CRITICAL,
    _JSON_SCHEMA_INSTRUCTION,
)

_DECOMPOSE_JSON = """{
  "title": "<chapter or story title>",
  "synopsis": "<2-3 sentences>",
  "mood": "<emotional tone>",
  "style_anchor": "<palette, lens, texture — one line>",
  "beats": [
    {
      "index": 0,
      "title": "<scene title>",
      "location": "<place · time>",
      "narrative": "<what happens; full character names; no wardrobe tags>",
      "enhancement_cues": ["<visual intent e.g. silhouette entry>"],
      "suggested_shot_size": "<wide|medium|close-up>",
      "estimated_duration_sec": 8
    }
  ],
  "characters": [
    {
      "name": "<full name>",
      "role": "protagonist|supporting|extra",
      "looks": [
        {
          "label": "<look slug>",
          "body": "<Role | Appearance | Wardrobe text>",
          "portrait_prompt_hint": "<optional T2I hint>"
        }
      ]
    }
  ],
  "scenes": [
    {
      "name": "<location name>",
      "looks": [
        {
          "label": "default",
          "body": "<environment description>",
          "environment_prompt_hint": "<optional>"
        }
      ]
    }
  ]
}"""

SCRIPT_DECOMPOSE_SYSTEM = f"""# Role

Decompose script text into a structured story artifact for segmented I2V production.

## Output (critical)

{_JSON_OUTPUT_CRITICAL}

{_JSON_SCHEMA_INSTRUCTION}

Schema:

```json
{_DECOMPOSE_JSON}
```

## Rules

- **beats**: 2–24 items; one beat = one story moment; **enhancement_cues** required (≥1 per beat).
- **characters**: every on-screen named person; exactly one **protagonist**; each character ≥1 look with non-empty **body**.
- **scenes**: one entry per distinct location in beats; each scene ≥1 look.
- Use **full character names** in beat narrative — no `Name（…）` wardrobe tags.
- Do not embed style/mood boilerplate inside beat narratives."""

_BEAT_PLAN_JSON = """{
  "beats": [
    {
      "beat_index": 0,
      "shot_intent": "<why this beat exists>",
      "narrative_role": "establish_context|introduce_subject|build_tension|deliver_payload|transition|emotional_beat|evidence|comparison|resolution|call_to_action",
      "segments": [
        {
          "role": "pre_anchor|face_anchor|post_anchor|keyframe|establishing",
          "duration_sec": 5,
          "shot_size": "<景别>",
          "characters_on_screen": ["<Name>"],
          "start_visibility": "silhouette|partial|full_face|invisible",
          "end_visibility": "silhouette|partial|full_face|invisible",
          "first_frame_requirement": "<hard t=0 constraint>",
          "reachability": "identity_critical|establishing|action_wide|empty",
          "is_intentional_empty": false,
          "spatial": {
            "location": "<place>",
            "objects": ["<prop>"],
            "camera_zones": [{{"id": "CZ1", "description": "...", "visible_area": "..."}}]
          }
        }
      ]
    }
  ]
}"""

BEAT_PLAN_SYSTEM = f"""# Role

Plan I2V segments per beat: visibility timeline, anchor splits, spatial layout, durations.

## Output (critical)

{_JSON_OUTPUT_CRITICAL}

{_JSON_SCHEMA_INSTRUCTION}

Schema:

```json
{_BEAT_PLAN_JSON}
```

## Rules

- One row per beat_index from the user message.
- **narrative_role** is beat-level story function (establish_context, build_tension, …) — **NOT** segment **reachability** (never use identity_critical|action_wide|establishing|empty here).
- **Visibility contract (critical)**:
  - **face_anchor** segments: **start_visibility MUST be full_face** (readable face at t=0).
  - All other roles (**keyframe**, **pre_anchor**, **post_anchor**, **establishing**): **start_visibility MUST NOT be full_face** — use **partial** (hands/props/UI/detail), **silhouette**, or **invisible** (intentional empty).
  - **partial** = body parts or props only (fingers, phone screen, object); **no readable facial features** in the planned t=0 frame.
- **identity_critical** beats MUST include a **face_anchor** segment (MCU/CU, minimal motion).
- **face_anchor cardinality**: at most **one face_anchor segment per beat**; **characters_on_screen** length MUST be **1** (single identity for readable face at t=0).
- Opening beat (index 0) with characters: **start_visibility** ≥ silhouette (not invisible).
- **is_intentional_empty=true** only for deliberate empty frames; then characters_on_screen=[].
- **partial** **first_frame_requirement** / **camera_zones.visible_area**: frame hands, props, UI, body parts, or silhouette only — **no readable face, no 面部特写, no 五官清晰** (negations like 无面部 / 面部不可见 / 无清晰五官 are OK).
- Segment durations 2–10 seconds; sum per beat ≈ estimated_duration_sec.
- Distinct **video** wording will be derived later — plan structure only."""

_SHOT_SPEC_JSON = """{
  "shots": [
    {
      "segment_index": 0,
      "five_aspect": {
        "subject": "<type + full names verbatim>",
        "subject_motion": "<actions in time order>",
        "scene": "<setting + time + mood; no wardrobe>",
        "spatial_framing": "<FG/MG/BG + shot size>",
        "camera": "<static OR movement primitives>"
      },
      "shot_language": {
        "shot_size": "<景别>",
        "camera_movement": "static|dolly_in|pan_left|...",
        "lighting_key": "natural|golden_hour|...",
        "depth_of_field": "shallow|medium|deep",
        "color_temperature": "cool|neutral|warm|mixed"
      },
      "shot_intent": "<why this frame exists>",
      "video_prompt": "<motion + camera grammar only; no style paste>",
      "start_visual": "<t=0 still; full names if on_screen>",
      "anchor_visual": "<MCU/CU still: shot size + full name + expression + environment; face_anchor ONLY>"
    }
  ]
}"""

SHOT_SPEC_SYSTEM = f"""# Role

Derive complete shot specs (5-Aspect + prompts) for each planned segment.

## Output (critical)

{_JSON_OUTPUT_CRITICAL}

{_JSON_SCHEMA_INSTRUCTION}

Schema:

```json
{_SHOT_SPEC_JSON}
```

## Rules

- One **shots[]** row per **segment_index** listed in the user message.
- **Honor segment context**: **start_visibility**, **end_visibility**, **role**, **characters_on_screen** from the user message are binding.
- Fill **five_aspect** first; **start_visual** = photographable instant BEFORE motion in video_prompt.
- **Visibility → framing (critical)**:
  - **start_visibility=full_face** (face_anchor only): **anchor_visual** readable face; **start_visual** may match anchor; minimal motion in **video_prompt**.
  - **start_visibility=partial**: **start_visual** frames hands/props/UI/detail only — **no readable face, no portrait composition, no 面部特写/五官清晰**; **five_aspect.subject** describes visible parts only (not face adjectives). Prefer props/UI over 面部轮廓.
  - **start_visibility=silhouette**: backlit/outline only; no facial detail.
  - **start_visibility=invisible** with characters_on_screen: use only when **is_intentional_empty** or environment-only establishing.
- If **characters_on_screen** is non-empty: every name MUST appear in **five_aspect.subject**; for **partial/silhouette**, names indicate ownership of visible body parts, not a face close-up.
- **face_anchor**: **anchor_visual** REQUIRED — must include **景别 (MCU/CU/特写) + character full name + expression + setting**; never output name-only stubs like `"Name"`. **start_visual** may match anchor; minimal motion in **video_prompt**.
- **Non face_anchor** (establishing, keyframe, pre_anchor, post_anchor, tail_continuation): **anchor_visual** MUST be empty string `""` — use **start_visual** only.
- **camera_movement=static** → video_prompt must NOT contain pan/dolly/zoom.
- UI/notification overlays: describe as flat graphic/text **without** faces, avatars, or portrait thumbnails on screen.
- Repeat protagonist visual attributes **verbatim** each segment — no 同上/she/he.
- No `Name（…）` wardrobe tags; no style_anchor paste into video_prompt."""

SHOT_REPAIR_SYSTEM = f"""# Role

Repair invalid shot rows after automated validation.

## Output (critical)

{_JSON_OUTPUT_CRITICAL}

Return JSON: {{"repairs": [{{"segment_index": 0, "start_visual": "...", "video_prompt": "...", "five_aspect": {{...}}, ...}}]}}

Fix ONLY listed segment_index values and listed issue codes. Do not remove segments."""
