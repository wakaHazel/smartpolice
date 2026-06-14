from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "video"
WORK_DIR = OUT_DIR / "semifinal_demo_work"
VIDEO_OUT = OUT_DIR / "semifinal_demo.mp4"

WIDTH = 1920
HEIGHT = 1080
BG = (246, 249, 247)
DARK = (24, 40, 36)
ACCENT = (20, 104, 83)
MUTED = (87, 104, 98)
PANEL = (255, 255, 255)
LINE = (203, 218, 211)


SLIDES = [
    {
        "title": "AI 生成图像取证研判与警务证据链系统",
        "subtitle": "智警杯警用大模型作品赛半决赛演示",
        "bullets": [
            "面向社交平台传播扰动后的 AIGC 图像辅助核查",
            "目标不是自动定案，而是提供可复核的疑似生成线索",
            "模型输出接入 hash、审计 ID、证据链和报告草稿",
        ],
        "narration": "大家好，本作品是AI生成图像取证研判与警务证据链系统，面向社交平台传播扰动后的AIGC图像辅助核查。系统不替代人工定案，而是把疑似生成线索、图片哈希、模型版本和审计编号组织成可复核证据链。",
        "duration": 28,
    },
    {
        "title": "问题背景",
        "subtitle": "元数据可能失效，视觉线索需要可复核",
        "bullets": [
            "GPT-image-2 等模型能生成高度拟真的公共安全场景图片",
            "社交平台会带来 JPEG 压缩、截图转存、裁剪和水印覆盖",
            "C2PA、EXIF、平台标识可能被剥离或削弱",
            "公安核查需要快速固定证据，同时保留人工复核边界",
        ],
        "narration": "GPT-image-2等生成模型可以快速生成高度拟真的公共安全场景图片。图片一旦进入社交平台，可能经历压缩、截图转存、裁剪和水印覆盖，原始元数据或平台标识可能被剥离。因此系统需要在元数据失效时提供可复核的视觉检测线索。",
        "duration": 34,
    },
    {
        "title": "技术路线",
        "subtitle": "本地视觉检测组件 + 鲁棒评测 + 警务证据链",
        "bullets": [
            "外部图片训练池：4691 张，内置演示样例不进入训练",
            "特征：视觉语义、频域纹理、JPEG 痕迹、文字覆盖、水印代理",
            "模型：115 维特征 + ExtraTrees + 类别原型兜底",
            "治理：candidate/active 生命周期、模型卡、门控激活",
        ],
        "narration": "技术路线不是简单套一个大模型接口，而是建设本地监督训练闭环。当前训练池包含四千六百九十一张外部图片，抽取视觉语义、频域纹理、JPEG压缩痕迹、文字覆盖和水印代理特征，再训练本地视觉检测和归因头。候选模型不会自动替换正式模型，必须经过门控或显式激活。",
        "duration": 44,
    },
    {
        "title": "系统首页",
        "subtitle": "展示训练池、active 模型、扰动增强与复测入口",
        "image": "output/playwright/workspace-desktop.png",
        "bullets": [
            "首页展示当前 active 模型、训练池规模和核心指标",
            "评委可以看到系统并非只有报告页面，而有可复现训练和评估入口",
        ],
        "narration": "打开系统首页，可以看到AI生成图像取证研判中心。页面展示训练池规模、当前active模型、扰动增强数量和复测入口。评委可以直观看到，系统不是只有文书展示，而是包含可复现的训练和评估链路。",
        "duration": 36,
    },
    {
        "title": "证据链演示",
        "subtitle": "上传图片后固定 hash、模型版本、审计 ID 与报告草稿",
        "image": "output/playwright/ui-e2e-real-chain-final.png",
        "bullets": [
            "证据条目包含图像、来源、传播和模型分析四类线索",
            "报告草稿明确写入人工复核声明",
        ],
        "narration": "选择案例后，系统会保存图片文件、大小、宽高、哈希和上传时间。启动图像取证研判后，系统输出疑似生成来源线索、置信度、候选分布和不确定项。证据链页面把图像证据、来源证据、传播证据和模型分析条目组织起来，报告草稿会写入模型版本、图片哈希、审计编号和人工复核声明。",
        "duration": 54,
    },
    {
        "title": "核心实验结果",
        "subtitle": "强项与边界同时披露",
        "bullets": [
            "active clean validation：Accuracy 0.708，Macro-F1 0.503",
            "GPT-image-2 Recall：0.915，仅作为疑似来源线索",
            "六条件扰动平均 Macro-F1：0.655",
            "最弱条件：screenshot_resave，GPT-image-2 recall 降至 0.222",
        ],
        "narration": "当前active模型在一百二十条clean source-holdout上，准确率为零点七零八，Macro F1为零点五零三，GPT-image-2召回为零点九一五。六条件扰动复测的平均Macro F1为零点六五五，其中截图转存是最弱条件，GPT-image-2召回下降到零点二二二。因此我们不把结果写成确定归因，而写成疑似来源线索。",
        "duration": 48,
    },
    {
        "title": "方向校正后的技术突破",
        "subtitle": "低误报初筛优先，GPT-image-2 作为第二层线索",
        "bullets": [
            "阈值校准候选：threshold 0.650",
            "360 条扰动验证：Real FPR 0.033，Generated Recall 0.692",
            "GPT-image-2 专项：clean AUC/Recall 1.000/1.000",
            "来源互留未达标：Source Macro-F1 0.309，仍需独立平台样本",
        ],
        "narration": "最新方向校正后，我们不再追求多生成器泛化满分，而是把低误报真实生成初筛放在第一层。阈值校准候选在三百六十条扰动验证上，把真实图误报率压到零点零三三，同时生成图召回为零点六九二。GPT-image-2专项组件clean表现很强，但来源互留仍未达标，因此只作为第二层疑似线索。",
        "duration": 48,
    },
    {
        "title": "泛化评估与诚实边界",
        "subtitle": "借鉴公开 benchmark 协议，不借用 leaderboard 分数",
        "bullets": [
            "strict source-holdout mean Macro-F1：0.124",
            "label-covered Macro-F1：0.354，binary Macro-F1：0.464",
            "GenImage/AIGIBench/SIDA/RRDataset/ITW-SM 作为协议参考",
            "后续补平台转码、截图、水印、多次压缩与真实图 hard negatives",
        ],
        "narration": "泛化评估上，系统借鉴GenImage、AIGIBench、SIDA、RRDataset和ITW-SM的协议思想，但不借用它们的排行榜分数。当前strict source-holdout均值Macro F1为零点一二四，label-covered诊断Macro F1为零点三五四，说明公开来源之间存在明显分布差异。这是后续补样本和盲测的重点。",
        "duration": 42,
    },
    {
        "title": "警务价值",
        "subtitle": "辅助研判、平台协查、证据留存和报告草稿",
        "bullets": [
            "快速固定图片 hash、来源 URL、截图和审计 ID",
            "把模型结果转换为证据链条目和处置建议",
            "输出报告草稿，服务人工复核和平台协查",
            "可用于涉警公信力、灾害险情、群体对立等谣言场景",
        ],
        "narration": "项目的警务价值在于辅助研判。系统快速固定图片哈希、来源链接、截图和审计编号，把模型结果转换为证据链条目和处置建议，并生成报告草稿。它适合用于涉警公信力、灾害险情、群体对立等公共安全谣言的早期核查。",
        "duration": 34,
    },
    {
        "title": "边界声明",
        "subtitle": "模型线索不替代证据标准",
        "bullets": [
            "不声称确定来自 GPT-image-2",
            "不替代 C2PA、水印、平台元数据、发布链路和人工取证",
            "不训练基础多模态大模型本体",
            "提交材料主动披露弱项和下一步计划",
        ],
        "narration": "最后强调边界。系统不声称确定某图片来自GPT-image-2，不替代C2PA、水印、平台元数据、发布链路和人工取证，也不声称训练了基础多模态大模型本体。它的定位是可复核的技术辅助线索和证据链整理工具。",
        "duration": 26,
    },
]


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = char
    if current:
        lines.append(current)
    return lines


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
    max_width: int,
    line_gap: int,
) -> int:
    x, y = xy
    for line in wrap_text(text, font, max_width):
        draw.text((x, y), line, font=font, fill=color)
        y += font.getbbox(line)[3] - font.getbbox(line)[1] + line_gap
    return y


def draw_bullet(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font: ImageFont.ImageFont, max_width: int) -> int:
    draw.ellipse((x, y + 14, x + 10, y + 24), fill=ACCENT)
    return draw_text_block(draw, text, (x + 28, y), font, DARK, max_width - 28, 12) + 8


def paste_screenshot(canvas: Image.Image, relative: str, box: tuple[int, int, int, int]) -> None:
    path = ROOT / relative
    if not path.exists():
        return
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = box
    max_w = x2 - x1
    max_h = y2 - y1
    image.thumbnail((max_w, max_h))
    panel = Image.new("RGB", (max_w, max_h), PANEL)
    px = (max_w - image.width) // 2
    py = (max_h - image.height) // 2
    panel.paste(image, (px, py))
    canvas.paste(panel, (x1, y1))


def render_slide(index: int, slide: dict[str, object]) -> Path:
    title_font = find_font(58 if index == 0 else 50)
    subtitle_font = find_font(30)
    body_font = find_font(32)
    small_font = find_font(24)

    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 92), fill=(232, 242, 237))
    draw.text((92, 26), "SmartPolice 半决赛演示", font=small_font, fill=MUTED)
    draw.text((WIDTH - 230, 26), f"{index + 1:02d}/{len(SLIDES):02d}", font=small_font, fill=MUTED)

    y = 150
    y = draw_text_block(draw, str(slide["title"]), (120, y), title_font, ACCENT, 1680, 16)
    if slide.get("subtitle"):
        y = draw_text_block(draw, str(slide["subtitle"]), (122, y + 8), subtitle_font, MUTED, 1680, 10)
    y += 24

    image_path = slide.get("image")
    if image_path:
        paste_screenshot(canvas, str(image_path), (1020, 205, 1780, 835))
        bullet_width = 800
    else:
        bullet_width = 1500

    draw.rounded_rectangle((100, y - 8, 100 + bullet_width + 40, min(890, y + 70 * len(slide["bullets"]) + 42)), radius=18, fill=PANEL, outline=LINE, width=2)
    bullet_y = y + 22
    for bullet in slide["bullets"]:  # type: ignore[index]
        bullet_y = draw_bullet(draw, str(bullet), 138, bullet_y, body_font, bullet_width)

    narration = str(slide["narration"])
    caption_lines = wrap_text(narration, find_font(25), 1680)[:3]
    draw.rounded_rectangle((100, 910, 1820, 1030), radius=16, fill=(238, 246, 242), outline=LINE, width=1)
    cy = 930
    for line in caption_lines:
        draw.text((130, cy), line, font=find_font(25), fill=DARK)
        cy += 32

    path = WORK_DIR / f"slide_{index:02d}.png"
    canvas.save(path)
    return path


def synthesize_wav(slide: dict[str, object], index: int) -> Path:
    wav = WORK_DIR / f"audio_{index:02d}.wav"
    text = str(slide["narration"]).replace("'", "’")
    ps = WORK_DIR / f"tts_{index:02d}.ps1"
    ps.write_text(
        "$ErrorActionPreference='Stop'\n"
        "$voice = New-Object -ComObject SAPI.SpVoice\n"
        "$voice.Rate = 1\n"
        "$voice.Volume = 100\n"
        "$voices = $voice.GetVoices()\n"
        "foreach ($v in $voices) { if ($v.GetDescription() -like '*Huihui*') { $voice.Voice = $v; break } }\n"
        "$stream = New-Object -ComObject SAPI.SpFileStream\n"
        f"$stream.Open('{wav}', 3, $false)\n"
        "$voice.AudioOutputStream = $stream\n"
        f"$voice.Speak('{text}') | Out-Null\n"
        "$stream.Close()\n",
        encoding="utf-8",
    )
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps)], check=True)
    return wav


def ffmpeg_path() -> str:
    from shutil import which

    found = which("ffmpeg")
    if found:
        return found
    fallback = Path(r"D:\Program Files\EVCapture\ffmpeg.exe")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError("ffmpeg not found")


def ffprobe_duration(path: Path) -> float:
    from shutil import which

    ffprobe = which("ffprobe") or str(Path(r"D:\Program Files\EVCapture\ffprobe.exe"))
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def build_video() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path()
    parts: list[Path] = []

    for index, slide in enumerate(SLIDES):
        image = render_slide(index, slide)
        wav = synthesize_wav(slide, index)
        duration = max(float(slide["duration"]), ffprobe_duration(wav) + 1.0)
        part = WORK_DIR / f"part_{index:02d}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loop",
                "1",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(image),
                "-i",
                str(wav),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-shortest",
                str(part),
            ],
            check=True,
        )
        parts.append(part)

    concat = WORK_DIR / "concat.txt"
    concat.write_text("\n".join(f"file '{part.as_posix()}'" for part in parts), encoding="utf-8")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            str(VIDEO_OUT),
        ],
        check=True,
    )
    print(VIDEO_OUT)


if __name__ == "__main__":
    build_video()
