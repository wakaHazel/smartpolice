from __future__ import annotations

from app.models import CaseSample, SpreadMetrics


DEMO_CASES: list[CaseSample] = [
    CaseSample(
        id="demo-doubao-collapse-disaster-001",
        title="Nano Banana生成虚假坍塌灾情图片研判",
        scenario="灾害险情谣言",
        platform="本地演示导入 / 短视频平台模拟传播",
        publish_time="2026-06-14 20:10",
        source_url="本地演示图片：backend/demo_assets/nano-banana-tunnel-collapse-social.png",
        content=(
            "演示样本为使用 Nano Banana 生成的虚假隧道施工坍塌抢险图片，模拟网传"
            "“某地在建隧道突发塌方，救援车辆、消防人员和大型机械正在现场处置”的"
            "公共安全线索。该图不对应真实灾情，用于演示社交平台压缩传播图片的来源研判。"
        ),
        image_description=(
            "单张 PNG：画面为夜间隧道施工现场，含中文施工标识、救护车、警灯、"
            "消防救援人员、挖掘机和吊装设备。"
        ),
        spread=SpreadMetrics(
            views=268000,
            reposts=7600,
            comments=11200,
            likes=18400,
            velocity="演示设定：灾情关键词带动同城群和短视频评论区快速转发",
        ),
        manual_label="已知为 Nano Banana 生成的虚假灾情图片，用于演示图像来源研判",
        manual_risk_score=88,
        tags=["Nano Banana生成", "隧道塌方", "施工抢险", "社交平台压缩图"],
        sensitivity_notes="坍塌、救援等灾情画面容易引发恐慌和集中求证；演示时需明确该图为生成样本。",
        review_note="演示视频主案例：使用用户提供的原始清晰图，避免低清报道截图影响模型识别。",
    ),
    CaseSample(
        id="demo-gptimage-station-police-conflict-001",
        title="GPT-image生成车站警民执法冲突图片研判",
        scenario="涉警公信力谣言",
        platform="本地演示导入 / 社交群与短视频平台模拟传播",
        publish_time="2026-06-14 21:05",
        source_url="本地演示图片：backend/demo_assets/gptimage-station-police-conflict-original.jpg",
        content=(
            "演示样本为使用 GPT-image 生成的虚假中国车站警民执法冲突图片，模拟网传"
            "“某车站民警与旅客发生激烈肢体冲突、现场大量群众围观拍摄”的涉警舆情线索。"
            "该图不对应真实执法事件，用于演示高敏感涉警图片的来源研判和证据链报告生成。"
        ),
        image_description=(
            "单张 JPEG：画面位于中国车站候车大厅，多名着警服人员与群众发生拉扯，"
            "背景有中文宣传横幅、站台编号和电子显示屏。"
        ),
        spread=SpreadMetrics(
            views=356000,
            reposts=12800,
            comments=19600,
            likes=22100,
            velocity="演示设定：涉警冲突关键词带动本地群聊和同城话题快速扩散",
        ),
        manual_label="已知为 GPT-image 生成的虚假涉警冲突图片，用于演示图像来源研判",
        manual_risk_score=91,
        tags=["GPT-image生成", "车站", "涉警冲突", "原始清晰图"],
        sensitivity_notes="涉警执法冲突类画面容易损害公安机关公信力并激化线下围观，应优先固定原图和传播链。",
        review_note="演示视频涉警高敏感案例：使用用户提供的 GPT-image 原始清晰图，展示报告生成链路。",
    ),
    CaseSample(
        id="demo-real-beijing-road-street-001",
        title="公开来源真实灾情救援照片核验",
        scenario="灾害险情核查",
        platform="Wikimedia Commons / 演示导入",
        publish_time="2008-05-14",
        source_url="https://commons.wikimedia.org/wiki/File:Sichuan_earthquake_save..JPG",
        content=(
            "演示样本为汶川地震后救援人员在受损建筑废墟中开展搜救的公开来源真实照片。"
            "该图用于和 AI 生成灾情图片形成对照，核查重点是避免把真实灾情救援现场误判为生成图。"
        ),
        image_description=(
            "单张 JPEG 实拍照片：救援人员在受损建筑和瓦砾现场转运伤员，画面具有真实灾害"
            "救援场景、自然光照和现场杂乱细节。"
        ),
        spread=SpreadMetrics(
            views=64200,
            reposts=1480,
            comments=620,
            likes=3100,
            velocity="演示设定：灾情图片被转发求证，需区分真实救援照片与AI编造灾情图",
        ),
        manual_label="公开来源真实灾情救援照片，用于演示真实照片核验",
        manual_risk_score=32,
        tags=["真实照片", "Wikimedia Commons", "汶川地震", "救援现场", "Public Domain"],
        sensitivity_notes="真实灾情救援图片仍需核验来源、时间和地点，避免被拼接进新的本地灾情谣言叙事。",
        review_note="真实照片对照案例：来源 Wikimedia Commons；文件 Sichuan earthquake save..JPG。",
    ),
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
