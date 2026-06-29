# LLM 驱动的时序感知分镜系统：完整分镜设计与多 LLM 调用方案

> **核心目标：从源头消除首帧无人物问题，通过空间约束和渐进 reveal 设计，替代后期的因果帧生成和批量抽卡。**

> **设计理念：分镜不是艺术创作，而是工程约束——每段首帧必须包含必要的人物信息，visibility 必须渐进变化。**

---

## 一、问题定义

### 1.1 现有开源平台的绕过策略

| 平台 | 策略 | 局限 |
|------|------|------|
| LumenX | 批量抽卡 + 人工筛选 | 成功率 60-70%，无法处理复杂入镜 |
| Micro-Drama | 每段首帧强制包含人物 | 无空镜开场，叙事单调 |
| Jellyfish | 全局种子 + 统一风格 | 单段生成，无连续运动 |

**共同问题**：都在规避空镜开场 + 人物延迟入镜的场景，而不是解决它。

### 1.2 我们的目标场景

```
剧本：夜晚，客厅。门铃响，朋友逆光入镜，转身与主人惊讶对视。

传统分镜（失败）：
  镜头 1: 空镜客厅（3秒）-> 首帧无人物，后续朋友出现，一致性断裂
  镜头 2: 两人对视（3秒）-> 朋友从哪来？跳变

我们的分镜（成功）：
  镜头 1: 客厅门口，人物逆光剪影推门（3秒）-> 首帧有剪影，渐进 reveal
  镜头 2: 人物转身，半脸->正面（3秒）-> 承接上段，visibility 升级
  镜头 3: 两人对视，情绪高潮（3秒）-> 承接上段，full_face 锁定
```

---

## 二、核心设计理念

### 2.1 Visibility 渐进定律

```
任何角色的出现必须遵循：
  invisible -> silhouette -> partial -> full_face
  
禁止跳变：
  ❌ invisible -> full_face（人物凭空出现）
  ❌ 无 -> silhouette（人物从画面外突然入镜）
  
强制渐进：
  ✅ invisible -> silhouette（人物从画外进入，首帧可见剪影）
  ✅ silhouette -> partial（人物转身，首帧可见半脸）
  ✅ partial -> full_face（人物正面，首帧清晰可见）
```

### 2.2 首帧可见性强制约束

```
分镜设计第一定律：
  如果一段视频后续需要角色一致性，首帧必须包含该角色的某种可见形式。

分镜设计第二定律：
  如果剧本要求空镜开场，必须将空镜与角色入镜合并为一段，
  或通过空间约束确保首帧包含角色剪影。
```

### 2.3 空间约束机位选择

```
机位不是艺术选择，而是空间计算：
  1. 构建场景 3D 空间（家具、门窗、角色位置）
  2. 计算每个机位的可见区域
  3. 选择能看到关键角色的机位
  4. 确保相邻机位角度差 >= 30 度（30度原则）
  5. 禁止越轴（除非特殊叙事需要）
```

---

## 三、五层 LLM 调用架构

### 3.1 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 剧本语义分析 (Script Semantic Analysis)            │
│ LLM: GPT-4o / Claude 3.5 / Qwen-Plus                       │
│ 输入: 自然语言剧本                                          │
│ 输出: 结构化故事图 (Story Graph)                            │
│ 关键: 提取角色移动路径、visibility 变化、空间布局            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: 3D 场景空间构建 (3D Scene Space Construction)       │
│ LLM: GPT-4o + 空间推理 Prompt                              │
│ 输入: 故事图                                                │
│ 输出: 场景空间描述 (Scene Space Description)                │
│ 关键: 家具坐标、角色位置、机位可见性计算                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: 机位约束分镜生成 (Camera-Constrained Shot Design)    │
│ LLM: GPT-4o + 影视规则 Prompt                              │
│ 输入: 故事图 + 场景空间 + 机位约束                          │
│ 输出: 分镜合约 (Shot Contract)                              │
│ 关键: 强制首帧可见性、渐进 reveal、机位连贯性                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: 首帧生成策略 (First Frame Generation Strategy)      │
│ LLM: GPT-4o + 视觉生成 Prompt                              │
│ 输入: 分镜合约 + 场景空间 + 角色资产                          │
│ 输出: 首帧生成指令 (First Frame Instructions)               │
│ 关键: 最小化因果帧生成、优先复用资产、Img2Img 轻调整          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: 验证与迭代 (Validation & Iteration)                 │
│ LLM: GPT-4o + 审核规则 Prompt                              │
│ 输入: 分镜合约 + 首帧策略 + 场景空间                          │
│ 输出: 审核报告 + 自动修复建议                                │
│ 关键: 首帧可见性检查、机位冲突检查、空间连续性检查           │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、各层详细设计

### 4.1 Layer 1: 剧本语义分析

#### 4.1.1 目标

从自然语言剧本中提取结构化叙事元素，构建**故事图 (Story Graph)**，为后续空间约束和分镜设计提供基础。

#### 4.1.2 关键提取项

| 提取项 | 说明 | 示例 |
|--------|------|------|
| 场景列表 | 地点、时间、氛围、空间布局 | 客厅、夜晚、暖光、20平米 |
| 角色列表 | 姓名、外貌、服装、初始位置 | 李明、30岁男性、黑西装、门外 |
| 事件时序 | 时间戳、动作、涉及角色 | 0秒门铃响、3秒开门、5秒转身 |
| 角色移动路径 | 从 A 点到 B 点的路径 | 门外->门口->客厅中央 |
| 角色可见性变化 | invisible->silhouette->partial->full_face | 门外 invisible->门口 silhouette->中央 full_face |
| 空间关系 | 家具位置、门窗朝向、遮挡关系 | 沙发靠东墙、门在西墙北侧 |

#### 4.1.3 Prompt 模板

```
你是一位专业的剧本分析师。请分析以下剧本，提取结构化信息。

【剧本】
[剧本文本]

【任务】
1. 提取场景列表（Scene List）
   - 场景 ID、地点、时间、氛围
   - 场景内的空间布局描述（房间、家具、门窗）

2. 提取角色列表（Character List）
   - 角色 ID、姓名、外貌特征、服装
   - 角色在每个场景中的初始位置

3. 提取事件时序（Event Timeline）
   - 事件 ID、时间戳、动作、涉及角色
   - 角色的移动路径（从 A 点到 B 点）
   - 角色的可见性变化（入场、出场、遮挡）

4. 构建故事图（Story Graph）
   - 节点：场景、事件、角色状态
   - 边：因果关系、时间顺序、空间关系

【输出格式】
```json
{
  "scenes": [
    {"scene_id": "S1", "location": "客厅", "time": "夜晚", "mood": "悬疑", "space_layout": "20平米客厅，沙发靠东墙，茶几中央，门在西墙北侧"}
  ],
  "characters": [
    {"character_id": "C1", "name": "李明", "appearance": "30岁男性，短发，黑色西装", "initial_positions": {"S1": {"position": "门外", "visibility": "invisible"}}}
  ],
  "events": [
    {"event_id": "E1", "scene_id": "S1", "timestamp": 0, "action": "门铃响", "characters": ["C1"], "character_changes": {"C1": {"position": "门外", "visibility": "invisible", "action": "按门铃"}}},
    {"event_id": "E2", "scene_id": "S1", "timestamp": 3, "action": "李明开门进入", "characters": ["C1"], "character_changes": {"C1": {"position": "门口", "visibility": "silhouette", "action": "推门进入"}}},
    {"event_id": "E3", "scene_id": "S1", "timestamp": 5, "action": "李明转身面对镜头", "characters": ["C1"], "character_changes": {"C1": {"position": "客厅中央", "visibility": "full_face", "action": "转身"}}}
  ],
  "story_graph": {
    "nodes": ["S1", "E1", "E2", "E3", "C1_invisible", "C1_silhouette", "C1_full"],
    "edges": [
      {"from": "E1", "to": "E2", "type": "causal", "relation": "门铃响导致开门"},
      {"from": "E2", "to": "E3", "type": "temporal", "relation": "进入后转身"},
      {"from": "C1_invisible", "to": "C1_silhouette", "type": "state_change", "trigger": "E2"},
      {"from": "C1_silhouette", "to": "C1_full", "type": "state_change", "trigger": "E3"}
    ]
  }
}
```
```

---

### 4.2 Layer 2: 3D 场景空间构建

#### 4.2.1 目标

基于故事图，构建精确的**3D 空间约束**，为机位选择提供计算基础，确保每个机位都能看到关键角色。

#### 4.2.2 关键计算项

| 计算项 | 说明 | 输出 |
|--------|------|------|
| 空间坐标系 | 房间尺寸、家具坐标 | 俯视图文字描述 |
| 机位区域 | 摄像机可放置位置 | 不遮挡、不穿墙 |
| 可见性矩阵 | 每个机位对每个角色的可见度 | invisible/silhouette/partial/full_face |
| 遮挡关系 | 家具、其他角色对视野的遮挡 | 沙发遮挡茶几后方 |
| 运动路径 | 角色移动的平滑路径 | 绕过家具、不跳跃 |

#### 4.2.3 Prompt 模板

```
你是一位专业的影视空间设计师。基于以下故事图，构建详细的 3D 场景空间。

【输入】
故事图: [Layer 1 输出]

【任务】
1. 空间建模
   - 为每个场景绘制简化的俯视图（用文字描述坐标）
   - 标注所有家具、门窗、角色的精确位置
   - 标注摄像机的可放置区域（不能穿墙、不能遮挡）

2. 可见性分析
   - 对每个角色在每个时间点的位置，计算哪些机位能看到该角色
   - 计算角色的可见度（invisible/silhouette/partial/full_face）
   - 标注遮挡关系（家具遮挡、其他角色遮挡）

3. 运动路径规划
   - 为每个角色的移动规划平滑路径
   - 确保路径不穿过家具
   - 确保路径在摄像机视野内

【输出格式】
```json
{
  "scene_spaces": {
    "S1": {
      "coordinate_system": {"origin": "西南角", "unit": "米", "x": "东", "y": "北"},
      "dimensions": {"width": 5, "height": 4},
      "objects": [
        {"name": "沙发", "type": "furniture", "position": {"x": 4, "y": 2, "width": 2, "height": 0.8}, "blocking": true},
        {"name": "茶几", "type": "furniture", "position": {"x": 2.5, "y": 2, "width": 1, "height": 0.6}, "blocking": true},
        {"name": "门", "type": "door", "position": {"x": 0.5, "y": 3.5, "width": 0.9, "height": 0.1}, "blocking": false}
      ],
      "camera_zones": [
        {"id": "CZ1", "position": {"x": 1, "y": 0.5}, "description": "东墙对面", "fov": 60, "visible_area": "客厅中央+门口"},
        {"id": "CZ2", "position": {"x": 2.5, "y": 0.5}, "description": "南墙中央", "fov": 60, "visible_area": "全客厅"},
        {"id": "CZ3", "position": {"x": 4.5, "y": 2}, "description": "东墙", "fov": 60, "visible_area": "门口+茶几"}
      ]
    }
  },
  "visibility_analysis": {
    "S1": [
      {"timestamp": 0, "character": "C1", "position": {"x": 0.5, "y": 3.5}, "camera_zones": {"CZ1": "invisible", "CZ2": "invisible", "CZ3": "invisible"}},
      {"timestamp": 3, "character": "C1", "position": {"x": 1, "y": 3.2}, "camera_zones": {"CZ1": "silhouette", "CZ2": "silhouette", "CZ3": "partial"}},
      {"timestamp": 5, "character": "C1", "position": {"x": 2.5, "y": 2.5}, "camera_zones": {"CZ1": "full_face", "CZ2": "full_face", "CZ3": "partial"}}
    ]
  },
  "movement_paths": {
    "C1": {
      "S1": [
        {"from": {"x": 0.5, "y": 3.5}, "to": {"x": 1, "y": 3.2}, "duration": 2, "path": "直线"},
        {"from": {"x": 1, "y": 3.2}, "to": {"x": 2.5, "y": 2.5}, "duration": 2, "path": "弧线绕过茶几"}
      ]
    }
  }
}
```
```

---

### 4.3 Layer 3: 机位约束分镜生成

#### 4.3.1 目标

基于 3D 空间约束，生成**强制首帧可见性**的分镜，确保每段首帧都包含必要的人物信息，避免首帧无人物问题。

#### 4.3.2 核心约束

| 约束 | 规则 | 违反后果 |
|------|------|----------|
| 首帧可见性强制 | 如果后续需要角色一致性，首帧必须包含该角色的某种可见形式 | 人物凭空出现，一致性断裂 |
| Visibility 渐进 | 必须 invisible->silhouette->partial->full_face，禁止跳变 | 人物突兀入镜 |
| 机位 30 度原则 | 相邻镜头角度差 >= 30 度 | 跳切感 |
| 禁止越轴 | 机位必须在轴线同一侧 | 空间混乱 |
| 时长约束 | 单镜头 3-8 秒 | 节奏拖沓 |

#### 4.3.3 关键算法：空镜开场处理

```
当剧本要求空镜开场时：

方案 A: 合并镜头（推荐）
  原设计: 镜头 1: 空镜（3秒）+ 镜头 2: 人物入镜（3秒）
  新设计: 镜头 1: 空镜+人物剪影入镜（6秒）
  约束: 首帧必须包含人物剪影（即使逆光、模糊）

方案 B: 机位调整
  原设计: 机位在客厅中央，看不到门口
  新设计: 机位调整到东墙对面，门口在画面边缘
  约束: 首帧画面边缘包含人物剪影

方案 C: 叙事重构（最后手段）
  原设计: 门铃响->空镜->人物入镜
  新设计: 人物背影按门铃->门开->进入
  约束: 首帧就有人物（背影），无需空镜
```

#### 4.3.4 Prompt 模板

```
你是一位专业的分镜导演。基于以下 3D 场景空间，设计分镜。

【输入】
故事图: [Layer 1 输出]
场景空间: [Layer 2 输出]

【核心约束】
1. 首帧可见性约束（绝对不可违反）
   - 如果一段视频后续需要角色一致性，首帧必须包含该角色的某种形式（剪影/背影/局部）
   - 严禁设计纯空镜开场，后续人物突然出现的分镜
   - 如果剧本要求空镜开场，必须使用合并镜头或机位调整方案

2. Visibility 渐进约束（绝对不可违反）
   - 角色出现必须遵循: invisible -> silhouette -> partial -> full_face
   - 禁止跳变: invisible->full_face, 无->silhouette
   - 每段必须明确标注角色的 start_visibility 和 end_visibility

3. 机位选择约束
   - 相邻镜头角度差 >= 30 度（30度原则）
   - 禁止越轴（除非特殊叙事需要）
   - 机位必须在 camera_zones 内
   - 优先选择能看到关键角色的机位

4. 分镜时长约束
   - 单镜头 3-8 秒
   - 情绪爆点 3 秒定格
   - 定场全景不超过 5 秒

【任务】
为每个场景设计分镜，确保：
- 每段首帧包含必要的人物信息（即使是剪影）
- 人物入镜过程被完整设计（不跳过）
- 相邻段之间机位连贯

【输出格式】
```json
{
  "shots": [
    {
      "shot_id": "S1.1",
      "scene_id": "S1",
      "time_range": [0, 3],
      "camera_zone": "CZ2",
      "camera_description": "南墙中央，中景",
      "first_frame_requirement": {
        "must_include": ["C1_silhouette"],
        "description": "画面右侧 1/3 为门，人物剪影推门进入，逆光",
        "justification": "C1 从 invisible 变为 visible，首帧必须包含剪影，避免后续突然出现"
      },
      "events_covered": ["E1", "E2"],
      "character_visibility": {
        "C1": {"start": "invisible", "end": "silhouette", "transition": "推门进入画面"}
      },
      "composition": "画面右侧 1/3 为门，人物逆光剪影，左侧 2/3 为客厅环境",
      "lighting": "门口逆光，客厅暖光台灯",
      "duration": 3
    },
    {
      "shot_id": "S1.2",
      "scene_id": "S1",
      "time_range": [3, 6],
      "camera_zone": "CZ1",
      "camera_description": "东墙对面，近景",
      "first_frame_requirement": {
        "must_include": ["C1_partial"],
        "description": "人物侧身，半脸可见，正在转身",
        "justification": "承接上段剪影，此段首帧必须是转身过程中的 partial，不能直接从剪影跳到 full_face"
      },
      "events_covered": ["E3"],
      "character_visibility": {
        "C1": {"start": "partial", "end": "full_face", "transition": "转身面对镜头"}
      },
      "composition": "人物位于画面中央，身体侧向左侧，头部转向镜头，面部半侧光",
      "lighting": "暖光台灯侧照，面部明暗对比",
      "duration": 3
    }
  ],
  "transition_rules": [
    {"from": "S1.1", "to": "S1.2", "type": "match_cut", "match_point": "人物位置", "camera_angle_change": 45}
  ]
}
```
```

---

### 4.4 Layer 4: 首帧生成策略

#### 4.4.1 目标

基于分镜合约，为每段设计最优的首帧生成方案，**最小化因果帧生成**的需求，降低计算成本。

#### 4.4.2 策略优先级

| 优先级 | 策略 | 适用场景 | 成本 |
|--------|------|----------|------|
| 1 | 直接复用角色资产 | 首帧需要 full_face | 最低 |
| 2 | 直接复用场景资产 | 首帧需要空镜元素 | 最低 |
| 3 | Img2Img 轻调整 | 上段尾帧与下段首帧 visibility 相近 | 低 |
| 4 | 角色资产 + 场景资产合成 | 首帧需要人物 + 场景 | 中 |
| 5 | 完整因果帧生成 | visibility 跳变较大 | 高 |

#### 4.4.3 决策算法

```
function decide_first_frame_strategy(shot, previous_shot, assets):
    # 情况 1: 首帧需要 full_face
    if shot.first_frame_requirement.visibility == \"full_face\":
        if assets.character_front_face exists:
            return STRATEGY_DIRECT_REUSE  # 直接复用角色定妆照
        else:
            return STRATEGY_CAUSAL_GENERATION  # 必须生成

    # 情况 2: 首帧需要 silhouette
    if shot.first_frame_requirement.visibility == \"silhouette\":
        if previous_shot and previous_shot.end_visibility == \"silhouette\":
            return STRATEGY_REUSE_TAIL  # 复用上段尾帧
        elif assets.character_full_body exists:
            return STRATEGY_COMPOSITE  # 角色全身图 + 逆光处理
        else:
            return STRATEGY_CAUSAL_GENERATION

    # 情况 3: 首帧需要 partial
    if shot.first_frame_requirement.visibility == \"partial\":
        if previous_shot and previous_shot.end_visibility in [\"silhouette\", \"partial\"]:
            return STRATEGY_IMG2IMG_LIGHT  # Img2Img strength=0.3
        else:
            return STRATEGY_CAUSAL_GENERATION

    # 情况 4: 首帧不需要人物
    if not shot.first_frame_requirement.must_include:
        return STRATEGY_SCENE_ASSET  # 直接复用场景资产
```

#### 4.4.4 Prompt 模板

```
你是一位首帧生成策略师。基于以下分镜合约，为每段设计最优的首帧生成方案。

【输入】
分镜合约: [Layer 3 输出]
角色资产: [角色三视图、定妆照、全身图]
场景资产: [场景参考图、环境图]

【核心策略】
1. 优先复用角色资产
   - 如果首帧需要 full_face，直接使用角色定妆照作为基础
   - 如果首帧需要 silhouette，使用角色全身图 + 逆光处理

2. 减少因果帧生成
   - 如果上段尾帧与下段首帧的 visibility 相同，直接复用尾帧
   - 如果上段尾帧是 silhouette，下段首帧是 partial，尝试用 Img2Img 轻微调整（strength=0.3）
   - 只有 visibility 跳变较大时（silhouette -> full_face），才需要完整的因果帧生成

3. 场景首帧优化
   - 如果首帧需要空镜元素（环境、道具），使用场景资产
   - 如果首帧需要人物 + 场景，优先将人物资产合成到场景资产中

【任务】
为每段输出首帧生成指令，明确：
- 基础素材（角色资产 / 场景资产 / 上段尾帧）
- 生成方式（直接复用 / Img2Img 轻调整 / 因果帧生成）
- 约束条件（ControlNet / IPAdapter / LoRA）

【输出格式】
```json
{
  "first_frame_strategies": [
    {
      "shot_id": "S1.1",
      "strategy": "composite",
      "base_assets": {
        "scene": "S1_living_room_night",
        "character": "C1_full_body_backlit"
      },
      "generation": {
        "method": "img2img_composite",
        "description": "将角色逆光全身图合成到客厅场景门口位置",
        "strength": 0.4,
        "controlnet": ["depth", "pose"],
        "prompt": "Night living room, warm lamp light, person silhouette entering from door, backlit, cinematic"
      },
      "verification": {
        "must_include": "C1_silhouette",
        "check": "人物剪影位于画面右侧 1/3，门口区域"
      }
    },
    {
      "shot_id": "S1.2",
      "strategy": "img2img_light",
      "base_assets": {
        "previous_tail": "S1.1_tail_frame",
        "character_reference": "C1_front_face"
      },
      "generation": {
        "method": "img2img",
        "description": "基于 S1.1 尾帧（人物背影）轻微调整为转身过程中的 partial 帧",
        "strength": 0.3,
        "controlnet": ["pose"],
        "ipadapter": "C1_front_face",
        "prompt": "Person turning to camera, side face visible, warm lighting, cinematic"
      },
      "verification": {
        "must_include": "C1_partial",
        "check": "人物侧身，半脸可见，与上段尾帧场景连贯"
      }
    }
  ]
}
```
```

---

### 4.5 Layer 5: 验证与迭代

#### 4.5.1 目标

全面审核分镜设计，发现首帧无人物、visibility 跳变、机位冲突等问题，提供自动修复方案。

#### 4.5.2 审核清单

| 检查项 | 检查内容 | 严重程度 | 自动修复 |
|--------|----------|----------|----------|
| 首帧人物可见性 | 每段首帧是否包含必要人物？ | Critical | 合并镜头 / 调整机位 |
| Visibility 渐进 | 是否存在跳变？ | Critical | 插入过渡段 |
| 机位连贯性 | 相邻段角度差是否 >= 30 度？ | Warning | 调整机位 |
| 越轴检查 | 是否存在越轴？ | Warning | 调整机位 |
| 空间连续性 | 角色位置是否连续？ | Critical | 调整路径 |
| 因果帧数量 | 每场景因果帧生成次数是否 <= 2？ | Warning | 优化策略 |

#### 4.5.3 Prompt 模板

```
你是一位分镜审核导演。基于以下分镜合约和首帧策略，进行全面审核。

【输入】
分镜合约: [Layer 3 输出]
首帧策略: [Layer 4 输出]
场景空间: [Layer 2 输出]

【审核清单】

1. 首帧人物可见性审核（Critical）
   - [ ] 每段首帧是否包含必要的人物信息（即使是剪影）？
   - [ ] 是否存在纯空镜开场，后续需要人物一致性的段？
   - [ ] 人物入镜过程是否完整（invisible -> silhouette -> partial -> full_face）？
   - [ ] 是否有跳变（如直接从 invisible 到 full_face）？

2. Visibility 渐进审核（Critical）
   - [ ] 每个角色的 visibility 变化是否遵循渐进定律？
   - [ ] 是否存在跳变？
   - [ ] 过渡段是否足够平滑？

3. 机位连贯性审核（Warning）
   - [ ] 相邻段机位角度差是否 >= 30 度？
   - [ ] 是否存在越轴？
   - [ ] 机位是否在 camera_zones 内？
   - [ ] 机位是否能看到关键角色？

4. 空间连续性审核（Critical）
   - [ ] 角色位置是否连续？（上段尾帧位置 = 下段起始位置）
   - [ ] 家具位置是否一致？
   - [ ] 光照方向是否一致？

5. 生成可行性审核（Warning）
   - [ ] 首帧策略是否可执行？
   - [ ] 因果帧生成次数是否过多？（目标：每场景 <= 2 次）
   - [ ] 是否有更简单的替代方案？

【输出格式】
```json
{
  "audit_results": [
    {
      "check_id": "1.1",
      "item": "首帧人物可见性",
      "shot_id": "S1.1",
      "status": "pass",
      "details": "首帧包含 C1_silhouette，符合要求"
    },
    {
      "check_id": "1.2",
      "item": "首帧人物可见性",
      "shot_id": "S2.3",
      "status": "fail",
      "details": "首帧为纯空镜（窗外夜景），但后续需要 C2 full_face，违反约束",
      "severity": "critical",
      "suggestion": "修改分镜：将 S2.2 和 S2.3 合并，或调整 S2.3 机位使其包含 C2 背影",
      "auto_fix": {
        "action": "merge_shots",
        "shots": ["S2.2", "S2.3"],
        "new_shot": {
          "shot_id": "S2.2-3",
          "time_range": [8, 15],
          "first_frame_requirement": {
            "must_include": ["C2_silhouette"],
            "description": "C2 从窗外进入画面，逆光剪影"
          }
        }
      }
    }
  ],
  "summary": {
    "total_checks": 20,
    "passed": 18,
    "failed": 2,
    "critical": 1,
    "warning": 1,
    "recommendation": "接受 auto_fix 修改 S2.2-3，重新生成 Layer 3 和 Layer 4"
  }
}
```
```

---

## 五、完整调用流程图

### 5.1 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     用户输入：自然语言剧本                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 剧本语义分析                                        │
│ LLM: GPT-4o / Claude 3.5                                    │
│ 调用: 1 次                                                    │
│ 输出: 故事图 (Story Graph)                                   │
│ 关键: 角色移动路径、visibility 变化、空间布局                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: 3D 场景空间构建                                     │
│ LLM: GPT-4o + 空间推理                                       │
│ 调用: 1 次（每场景）                                          │
│ 输出: 场景空间 (Scene Space)                                 │
│ 关键: 家具坐标、机位可见性、运动路径                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: 机位约束分镜生成                                    │
│ LLM: GPT-4o + 影视规则                                       │
│ 调用: 1 次（每场景）                                          │
│ 输出: 分镜合约 (Shot Contract)                               │
│ 关键: 强制首帧可见性、渐进 reveal、机位连贯性                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: 首帧生成策略                                        │
│ LLM: GPT-4o + 视觉生成                                       │
│ 调用: 1 次（每场景）                                          │
│ 输出: 首帧策略 (First Frame Strategy)                        │
│ 关键: 最小化因果帧生成、优先复用资产、Img2Img 轻调整          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: 验证与迭代                                          │
│ LLM: GPT-4o + 审核规则                                       │
│ 调用: 1 次（每场景）                                          │
│ 输出: 审核报告 (Audit Report)                                │
│ 关键: 首帧可见性检查、机位冲突检查、空间连续性检查           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
          ┌────────┐    ┌────────────┐   ┌────────────┐
          │ 通过   │    │ 警告       │   │ 失败       │
          │        │    │ 人工确认   │   │ 自动修复   │
          │ 输出   │    │ 是否接受   │   │ 回到 Layer │
          │ 分镜合约│    │ 警告？     │   │ 3 或 4     │
          │ 首帧策略│    │            │   │ 重生成     │
          └────────┘    └────────────┘   └────────────┘
                              │                │
                              └────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     输出：最终分镜合约 + 首帧策略              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     执行：首帧生成 -> I2V 生成 -> 拼接           │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 调用次数统计

| 层级 | 每场景调用次数 | 20 集短剧调用次数 | 说明 |
|------|-------------|----------------|------|
| Layer 1 | 1 | 20 | 剧本分析，只需一次 |
| Layer 2 | 1 | 20 | 空间构建，每场景一次 |
| Layer 3 | 1 | 20 | 分镜生成，每场景一次 |
| Layer 4 | 1 | 20 | 首帧策略，每场景一次 |
| Layer 5 | 1-3 | 20-60 | 验证迭代，通常一次通过 |
| **总计** | **5-7** | **100-140** | **每场景 5-7 次 LLM 调用** |

### 5.3 成本估算

| 项目 | 单次成本 | 每场景成本 | 20 集成本 |
|------|---------|----------|----------|
| LLM 调用 (GPT-4o) | $0.01-0.05 | $0.05-0.25 | $1-5 |
| 首帧生成 (T2I/Img2Img) | $0.02-0.10 | $0.10-0.50 | $2-10 |
| I2V 生成 (Wan 2.1) | $0.05-0.20 | $0.25-1.00 | $5-20 |
| 拼接/后期 | $0.01 | $0.05 | $1 |
| **总计** | - | **$0.45-1.80** | **$9-36** |

---

## 六、与现有开源平台的对比

| 维度 | LumenX / Micro-Drama | 我们的方案 |
|------|---------------------|-----------|
| 分镜设计 | 单 LLM 调用，简单 prompt | 5 层 LLM 调用，空间约束 |
| 首帧可见性 | 不强制，允许纯空镜 | **强制每段首帧包含人物** |
| 人物入镜 | 跳过过程，直接出现 | **完整设计渐进 reveal** |
| 机位选择 | 随机或人工选择 | **基于 3D 空间计算** |
| 一致性保障 | 批量抽卡 + 人工筛选 | **源头约束 + 自动审核** |
| 因果帧生成 | 频繁使用 | **最小化使用** |
| LLM 调用次数 | 1-2 次/场景 | 5-7 次/场景 |
| 单场景成本 | $0.10-0.30 | $0.45-1.80 |
| 质量 | 中（抽卡筛选） | **高（源头控制）** |
| 适用场景 | 简单短剧 | **影视级叙事** |

---

## 七、实施路径

### Phase 1: 快速验证（2 周）

```
目标: 验证核心约束的有效性

必做:
  1. 实现 Layer 1 + Layer 3（跳过 Layer 2 的 3D 计算，用简化的空间描述）
  2. 用 5-10 个典型剧本测试
  3. 统计首帧无人物问题的发生率
  4. 目标：从 30% 降低到 5%

不做:
  - Layer 2 的完整 3D 空间计算
  - Layer 4 的复杂策略优化
  - Layer 5 的自动修复
```

### Phase 2: 完整实现（4 周）

```
目标: 实现完整 5 层架构

必做:
  1. 实现 Layer 2 的 3D 空间计算（可用简化的 2D 俯视图替代）
  2. 实现 Layer 4 的首帧策略优化
  3. 实现 Layer 5 的自动审核和修复
  4. 集成到现有工作流
```

### Phase 3: 优化迭代（持续）

```
目标: 提升效率和质量

必做:
  1. 收集实际生成数据，优化 prompt
  2. 训练专门的分镜设计小模型（基于 LLM 微调）
  3. 建立分镜模板库，加速常见场景
  4. 实现多场景并行调用，减少总时间
```

---

## 八、核心结论

> **现有开源平台用规避和筛选做短剧，我们用控制和生成做影视。**
> 
> **通过 5 层 LLM 调用架构，从源头强制首帧可见性和 visibility 渐进，将首帧无人物问题从 30% 降低到 5% 以下，同时减少因果帧生成的依赖。**
> 
> **这不是简单的 prompt 优化，而是系统性的工程约束——分镜设计不再是艺术创作，而是空间计算和可见性规划的工程问题。**

---

*文档版本: 2026-06-29*
*基于: GPT-4o / Claude 3.5 / Qwen-Plus / Wan 2.1 / LTX 2.3*