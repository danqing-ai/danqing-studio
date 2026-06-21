"""Chinese synopsis fixtures for long-video storyboard unit tests (no LLM)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynopsisBriefCase:
    """One brief + mock LLM outputs exercising plan → expand → shot build."""

    name: str
    brief: str
    target_duration_sec: float
    segment_duration_sec: float
    plan_script: str
    expand_script: str
    """When set, simulates Expand returning one duplicated visual blob (project1 bug)."""
    broken_expand_blob: str = ""


SYNOPSIS_BRIEF_CASES: tuple[SynopsisBriefCase, ...] = (
    SynopsisBriefCase(
        name="zhao_jinmai_vs_wukong",
        brief=(
            "赵今麦听信豆包建议，只身挑战孙悟空，结果被一根毫毛秒杀，"
            "再次落入地府，被阎罗王嘲笑。"
        ),
        target_duration_sec=30.0,
        segment_duration_sec=5.0,
        plan_script=(
            "[Anchor] 赵今麦，现代简约白T恤与黑色短发，神情认真却略带莽撞，"
            "写实电影感，35mm 浅景深。\n"
            "[Beat 1] 赵今麦低头刷手机，豆包对话框弹出「去挑战孙悟空吧」。\n"
            "[Beat 2] 赵今麦独自走上云雾缭绕的山巅，远处金箍棒隐约可见。\n"
            "[Beat 3] 孙悟空端坐云端，懒洋洋地拔下一根毫毛。\n"
            "[Beat 4] 毫毛化作分身一闪，赵今麦被击飞，画面骤暗。\n"
            "[Beat 5] 赵今麦坠入幽冥地府，鬼火与锁链环绕。\n"
            "[Beat 6] 阎罗王端坐案后，掩口嘲笑，字幕感冷色顶光。"
        ),
        expand_script=(
            "[Visual 1] 近景，赵今麦在卧室暖光下盯手机屏幕，豆包界面占满屏幕，"
            "浅景深，写实。\n"
            "[Motion 1] 镜头缓慢推近屏幕，她犹豫后握拳起身。\n"
            "[Visual 2] 广角，她沿石阶走向云海山巅，远处孙悟空剪影。\n"
            "[Motion 2] 跟拍侧移，山风掀动衣角，云层翻涌。\n"
            "[Visual 3] 中景，孙悟空云端盘坐，指尖拈着一根毫毛，金色逆光。\n"
            "[Motion 3] 固定机位，毫毛离指，微光粒子散开。\n"
            "[Visual 4] 快切，毫毛分身掠过，赵今麦被震飞，尘土与光屑。\n"
            "[Motion 4] 手持晃动，她倒飞出镜，画面骤黑。\n"
            "[Visual 5] 低角度，地府甬道鬼火摇曳，赵今麦踉跄落地。\n"
            "[Motion 5] 缓慢前推，锁链与雾气向她聚拢。\n"
            "[Visual 6] 中近景，阎罗王案后掩口，顶光打亮嘲笑表情。\n"
            "[Motion 6] 轻微推近，冷色环境光，赵今麦狼狈抬头。"
        ),
        broken_expand_blob=(
            "赵今麦，现代简约白T恤与黑色短发\n"
            "[Beat 1] 赵今麦低头刷手机，豆包对话框弹出\n"
            "[Beat 2] 赵今麦独自走上云雾缭绕的山巅\n"
            "[Beat 3] 孙悟空端坐云端，懒洋洋地拔下一根毫毛\n"
            "[Beat 4] 毫毛化作分身一闪，赵今麦被击飞\n"
            "[Beat 5] 赵今麦坠入幽冥地府，鬼火与锁链环绕\n"
            "[Beat 6] 阎罗王端坐案后，掩口嘲笑"
        ),
    ),
    SynopsisBriefCase(
        name="neon_detective",
        brief=(
            "雨夜霓虹巷里，穿红风衣的女侦探追查线索，"
            "最终在废弃地铁站与幕后黑手对峙。"
        ),
        target_duration_sec=20.0,
        segment_duration_sec=5.0,
        plan_script=(
            "[Anchor] 女侦探，深红长风衣，湿发贴颊，霓虹蓝粉侧光，"
            "赛博 noir 电影感。\n"
            "[Beat 1] 雨巷远景，女侦探撑伞快步前行，霓虹倒影在积水里。\n"
            "[Beat 2] 巷口停步，查看证物袋里的旧照片。\n"
            "[Beat 3] 废弃地铁入口，女侦探推门而入，轨道深处有微光。\n"
            "[Beat 4] 站台尽头，幕后黑手背影出现，女侦探拔枪对峙。"
        ),
        expand_script=(
            "[Visual 1] 雨夜霓虹巷远景，红风衣女侦探撑伞，积水反光。\n"
            "[Motion 1] 缓慢跟拍，她沿巷道向前，雨丝划过镜头。\n"
            "[Visual 2] 近景，她停步查看证物袋内旧照片，侧脸被霓虹染色。\n"
            "[Motion 2] 推近至照片，手指轻触画面。\n"
            "[Visual 3] 废弃地铁入口，铁门半开，轨道深处微光。\n"
            "[Motion 3] 手持跟拍，她推门进入，回声渐强。\n"
            "[Visual 4] 站台尽头对峙，黑手背影与举枪侦探形成剪影。\n"
            "[Motion 4] 缓慢环绕，紧张对峙，远处列车灯闪烁。"
        ),
    ),
    SynopsisBriefCase(
        name="wuxia_duel",
        brief=(
            "少年剑客奉师命下山，于华山之巅与魔教教主决战，"
            "一剑破敌后飘然远去。"
        ),
        target_duration_sec=25.0,
        segment_duration_sec=5.0,
        plan_script=(
            "[Anchor] 少年剑客，青衫束发，长剑在背，水墨武侠电影感，"
            "晨雾与松风。\n"
            "[Beat 1] 山门告别，师父递过剑穗，少年抱拳。\n"
            "[Beat 2] 华山险径，少年剑客踏云而上，衣袂猎猎。\n"
            "[Beat 3] 峰顶对决，魔教教主黑袍现身，双剑交击。\n"
            "[Beat 4] 一剑破敌，教主倒地，少年收剑。\n"
            "[Beat 5] 云海之上，少年剑客飘然远去，只留下剑穗飘落。"
        ),
        expand_script=(
            "[Visual 1] 古刹山门，少年剑客向师父抱拳，剑穗递出。\n"
            "[Motion 1] 固定机位，晨雾流动，他转身下山。\n"
            "[Visual 2] 华山险径，青衫少年踏石而上，松枝摇曳。\n"
            "[Motion 2] 侧跟拍，衣袂与发丝随风。\n"
            "[Visual 3] 峰顶，黑袍教主现身，双剑交击火花四溅。\n"
            "[Motion 3] 环绕快切，剑光交错。\n"
            "[Visual 4] 决定性一剑，教主倒地，少年收剑入鞘。\n"
            "[Motion 4] 慢动作收剑，尘雾散开。\n"
            "[Visual 5] 云海远景，少年背影远去，剑穗飘落。\n"
            "[Motion 5] 拉远，人影没入云涛。"
        ),
    ),
    SynopsisBriefCase(
        name="space_emergency",
        brief=(
            "宇航员在失联空间站的舱外进行紧急维修，"
            "突然遭遇未知发光体逼近，险象环生。"
        ),
        target_duration_sec=15.0,
        segment_duration_sec=5.0,
        plan_script=(
            "[Anchor] 宇航员，白色舱外服，头盔面罩映地球弧光，"
            "硬科幻 IMAX 质感。\n"
            "[Beat 1] 舱外飘浮，宇航员检查太阳能板断裂处。\n"
            "[Beat 2] 焊接火花飞溅，通讯指示灯闪烁失联。\n"
            "[Beat 3] 远处出现未知发光体，缓慢逼近。\n"
        ),
        expand_script=(
            "[Visual 1] 舱外中景，宇航员飘浮于太阳能板旁，地球在背景。\n"
            "[Motion 1] 缓慢环绕，他打开工具包。\n"
            "[Visual 2] 特写，焊接火花与断裂线缆，头盔 HUD 显示 NO SIGNAL。\n"
            "[Motion 2] 推近面罩，反射里出现异光。\n"
            "[Visual 3] 广角，未知发光体自深空逼近，宇航员回头。\n"
            "[Motion 3] 手持急推，他抓紧扶手，警报红光闪烁。"
        ),
    ),
    SynopsisBriefCase(
        name="palace_intrigue",
        brief=(
            "宫女无意中撞破皇后下毒，连夜逃亡御花园，"
            "被禁军围堵后跳入太液池脱身。"
        ),
        target_duration_sec=25.0,
        segment_duration_sec=5.0,
        plan_script=(
            "[Anchor] 年轻宫女，素色宫装，发髻微乱，烛火暖色，"
            "古装悬疑电影感。\n"
            "[Beat 1] 偏殿角落，宫女窥见皇后往茶盏中投毒。\n"
            "[Beat 2] 惊觉后退，碰倒铜盆，声响惊动侍卫。\n"
            "[Beat 3] 连夜穿过回廊，灯笼摇曳，禁军火把逼近。\n"
            "[Beat 4] 御花园月门下被围堵，宫女决意跃入太液池。\n"
            "[Beat 5] 水下潜行，荷叶与倒影掩护，宫女游向暗渠。"
        ),
        expand_script=(
            "[Visual 1] 偏殿烛火，宫女在屏风后窥视皇后投毒。\n"
            "[Motion 1] 缓慢推近，她屏息，手捂口鼻。\n"
            "[Visual 2] 近景，后退碰倒铜盆，侍卫转头。\n"
            "[Motion 2] 手持跟拍，她转身奔逃。\n"
            "[Visual 3] 回廊夜景，灯笼摇曳，禁军火把自远处逼近。\n"
            "[Motion 3] 侧跟拍，她提裙疾行。\n"
            "[Visual 4] 月门下围堵，禁军列阵，她退至池边。\n"
            "[Motion 4] 固定机位，她纵身跃入池中，水花四溅。\n"
            "[Visual 5] 水下视角，荷叶倒影，她潜游向暗渠。\n"
            "[Motion 5] 缓慢前推，气泡与月光条纹。"
        ),
    ),
)
