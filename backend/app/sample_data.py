from __future__ import annotations

from app.models import CaseSample, SpreadMetrics


DEMO_CASES: list[CaseSample] = [
    CaseSample(
        id="police-trust-001",
        title="AI合成执法冲突图文引发涉警公信力风险",
        scenario="涉警公信力谣言",
        platform="短视频平台",
        publish_time="2026-05-28 21:16",
        source_url="公开平台样本，已脱敏",
        content=(
            "网传某地民警在夜间处置纠纷时推搡群众，配图称现场多人受伤，"
            "并暗示警方隐瞒执法记录。"
        ),
        image_description=(
            "画面呈现夜间街面、警灯、制服人员与围观人群，但人物手部边缘、"
            "肩章细节和路牌文字存在明显异常。"
        ),
        spread=SpreadMetrics(
            views=186000,
            reposts=4820,
            comments=3290,
            likes=9100,
            velocity="2小时内快速扩散",
        ),
        manual_label="疑似AI生成图像叠加误导性叙事",
        tags=["涉警", "AI合成", "执法公信力", "情绪煽动"],
        sensitivity_notes="涉及公安机关执法公信力，评论区出现聚集性负面情绪。",
    ),
    CaseSample(
        id="disaster-risk-002",
        title="旧图/AI图包装成本地灾害险情造成恐慌扩散",
        scenario="灾害险情谣言",
        platform="社交媒体",
        publish_time="2026-05-30 07:42",
        source_url="公开平台样本，已脱敏",
        content=(
            "网传本市南部山区突发塌方，多辆校车被困，要求市民立即转发避险。"
        ),
        image_description=(
            "配图为泥石流冲毁道路和车辆排队画面，天气、植被、道路标识与本地"
            "近期实况不一致，图片压缩痕迹明显。"
        ),
        spread=SpreadMetrics(
            views=432000,
            reposts=12600,
            comments=8410,
            likes=25100,
            velocity="30分钟内跨群转发",
        ),
        manual_label="旧图嫁接或AI改写灾害场景",
        tags=["灾害", "公共恐慌", "旧图嫁接", "跨群传播"],
        sensitivity_notes="内容涉及灾害险情和学生安全，可能诱发集中报警与抢险资源挤兑。",
    ),
    CaseSample(
        id="group-polarization-003",
        title="伪造性别冲突事件截图煽动群体对立",
        scenario="群体对立煽动型谣言",
        platform="论坛社区",
        publish_time="2026-05-29 23:05",
        source_url="公开平台样本，已脱敏",
        content=(
            "网传某商圈发生性别冲突事件，贴文配聊天截图和现场照片，称警方偏袒一方，"
            "号召网友线下集合声援。"
        ),
        image_description=(
            "聊天截图字体间距不一致，头像重复；现场照片人物面部纹理异常，商场标识"
            "与声称地点不匹配。"
        ),
        spread=SpreadMetrics(
            views=268000,
            reposts=7200,
            comments=15800,
            likes=19800,
            velocity="夜间评论量快速上涨",
        ),
        manual_label="疑似伪造截图和AI生成现场图",
        tags=["群体对立", "性别议题", "网暴风险", "线下聚集"],
        sensitivity_notes="评论区出现群体标签化、网暴动员和线下聚集倾向。",
    ),
    CaseSample(
        id="low-risk-004",
        title="低传播量误传公共提示信息",
        scenario="低风险误传",
        platform="本地生活群",
        publish_time="2026-05-31 10:10",
        source_url="公开平台样本，已脱敏",
        content=(
            "群内转发称某路段临时交通管制，提醒绕行，但未附权威来源。"
        ),
        image_description="配图为普通道路拥堵截图，无明显AI生成痕迹，未出现激烈情绪表达。",
        spread=SpreadMetrics(
            views=820,
            reposts=12,
            comments=8,
            likes=17,
            velocity="小范围缓慢传播",
        ),
        manual_label="低风险误传，需核验后提示",
        tags=["交通", "低传播", "误传"],
        sensitivity_notes="传播范围有限，未出现明显情绪煽动或线下扰动迹象。",
    ),
]


def get_case(case_id: str) -> CaseSample:
    for case in DEMO_CASES:
        if case.id == case_id:
            return case
    raise KeyError(case_id)
