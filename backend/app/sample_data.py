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


TAMPER_DEMO_CASES: list[CaseSample] = [
    CaseSample(
        id="tamper-demo-order-after-sale-001",
        title="寿司郎消费小票日期改写不在场证明核查",
        scenario="嫌疑人不在场证明消费凭证日期疑似改写",
        platform="用户提供脱敏演示样本",
        publish_time="2026-06-20 15:39",
        source_url="本地演示样本：微信图片_20260621003512_937_88.png",
        content=(
            "嫌疑人提交一张寿司郎上海 LCM 置汇旭辉店消费小票，声称案发时正在店内就餐。"
            "小票显示收据号、门店、人数、消费项目和合计金额，但日期字段疑似被局部修改，"
            "需要围绕日期、时间、收银流水和支付记录进行复核。"
        ),
        image_description=(
            "寿司郎中文消费小票照片，重点复核 2026/06/20 日期字段及其周边纸面纹理。"
        ),
        spread=SpreadMetrics(
            views=1200,
            reposts=18,
            comments=45,
            likes=60,
            velocity="演示样例，不代表真实传播",
        ),
        manual_label="document_field_tampered",
        tags=["AI篡改取证", "寿司郎", "消费小票", "日期改写", "不在场证明"],
        sensitivity_notes="用户提供的脱敏演示图，用于展示消费凭证日期字段局部改写核查；仅输出候选异常区域和人工复核建议。",
    ),
    CaseSample(
        id="tamper-demo-bank-transfer-001",
        title="餐饮/零售消费凭证局部修补备用样例",
        scenario="Tax Invoice 消费凭证局部修补核查",
        platform="HF SROIE / receipt document tamper",
        publish_time="2026-06-18 09:05",
        source_url="hf_tamper_document_forensics / receipt document tamper",
        content=(
            "备用演示样本来自公开英文收据/Tax Invoice 数据池，用于复核消费凭证局部擦除、"
            "修补或字段覆盖风险；前端核心演示以一张篡改样本和一张真实对照为主。"
        ),
        image_description=(
            "英文消费凭证风格图片，独立来自篡改线数据池，不关联生成检测线素材。"
        ),
        spread=SpreadMetrics(
            views=980,
            reposts=9,
            comments=22,
            likes=35,
            velocity="演示样例，不代表真实传播",
        ),
        manual_label="document_field_tampered",
        tags=["AI篡改取证", "SROIE", "Tax Invoice", "备用样例"],
        sensitivity_notes="图片来自篡改线 receipt/document tamper 数据池，不包含真实个人身份信息；仅输出候选异常区域和人工复核建议。",
    ),
    CaseSample(
        id="tamper-demo-medical-complaint-001",
        title="办公用品消费凭证真实原图对照",
        scenario="Tax Invoice 真实消费凭证低风险对照",
        platform="HF SROIE / receipt authentic control",
        publish_time="2026-06-18 09:10",
        source_url="hf_tamper_document_forensics: doc_authentic_sroie_rth_source_0417_84bd60da3911.jpg",
        content=(
            "英文办公用品/文具商店 Tax Invoice 真实原图对照，包含 invoice no、date、"
            "商品、amount、GST、payment 等字段；用于低风险材料复核演示。"
        ),
        image_description=(
            "K STATIONERY & OFFICE SUPPLIES 办公用品消费凭证真实原图，无 manifest bbox。"
        ),
        spread=SpreadMetrics(
            views=1500,
            reposts=25,
            comments=58,
            likes=72,
            velocity="演示样例，不代表真实传播",
        ),
        manual_label="authentic_unmodified",
        tags=["AI篡改取证", "SROIE", "Tax Invoice", "真实对照"],
        sensitivity_notes="图片来自篡改线 hf_tamper_document_forensics 数据池真实原图对照，不包含真实个人身份信息；仅输出辅助研判和人工复核建议。",
    ),
]


def get_case(case_id: str) -> CaseSample:
    for case in [*DEMO_CASES, *TAMPER_DEMO_CASES]:
        if case.id == case_id:
            return case
    raise KeyError(case_id)
