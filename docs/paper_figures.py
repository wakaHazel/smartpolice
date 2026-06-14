from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "output" / "figures"

INK = (24, 24, 27)
MUTED = (82, 82, 91)
LINE = (113, 113, 122)
LIGHT = (244, 244, 245)
MID = (228, 228, 231)
WHITE = (255, 255, 255)
ACCENT = (30, 64, 175)
ACCENT_LIGHT = (239, 246, 255)
WARN = (146, 64, 14)
WARN_LIGHT = (255, 247, 237)
OK = (21, 128, 61)
OK_LIGHT = (240, 253, 244)
RED = (185, 28, 28)
RED_LIGHT = (254, 242, 242)
BLUE = (37, 99, 235)
BLUE_LIGHT = (239, 246, 255)
ORANGE = (217, 119, 6)
ORANGE_LIGHT = (255, 251, 235)
PURPLE = (91, 33, 182)
PURPLE_LIGHT = (245, 243, 255)
TEAL = (15, 118, 110)
TEAL_LIGHT = (240, 253, 250)

MPL_DPI = 180
MPL_BLUE = "#2563EB"
MPL_TEAL = "#0F766E"
MPL_ORANGE = "#D97706"
MPL_RED = "#B91C1C"
MPL_GREEN = "#15803D"
MPL_PURPLE = "#7C3AED"
MPL_GRAY = "#52525B"
MPL_LIGHT_GRID = "#E4E4E7"


def _configure_matplotlib() -> None:
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in font_candidates:
        candidate = Path(font_path)
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#71717A",
            "axes.labelcolor": "#18181B",
            "xtick.color": "#3F3F46",
            "ytick.color": "#3F3F46",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )


_configure_matplotlib()


PERTURBATION_RESULTS = [
    ("无扰动", 1.000, 0.933, 1.000, 0.645),
    ("轻度JPEG", 0.833, 0.740, 0.889, 0.466),
    ("强JPEG", 0.858, 0.765, 0.889, 0.489),
    ("截图重保存", 0.583, 0.496, 0.222, 0.370),
    ("中心裁剪", 0.708, 0.614, 0.444, 0.403),
    ("水印覆盖", 0.750, 0.659, 0.667, 0.415),
]

BINARY_GATE_RESULTS = [
    ("无扰动", 0.980, 0.938, 0.944, 0.067),
    ("轻度JPEG", 0.965, 0.867, 0.806, 0.061),
    ("强JPEG", 0.974, 0.886, 0.842, 0.061),
    ("截图重保存", 0.973, 0.787, 0.628, 0.018),
    ("中心裁剪", 0.970, 0.716, 0.505, 0.012),
    ("水印覆盖", 0.973, 0.818, 0.679, 0.012),
]

SOURCE_HOLDOUT_METRICS = [
    ("严格多类别\n平均F1", 0.124),
    ("已见类别\n平均F1", 0.139),
    ("二分类\n平均F1", 0.401),
    ("生成图\n召回率", 0.660),
    ("真实图\n误报率", 0.242),
    ("覆盖标签\n平均F1", 0.354),
]

FEATURE_ABLATION = [
    ("全部特征", 0.620, 0.250),
    ("仅视觉取证", 0.504, 0.250),
    ("去除文字上下文", 0.500, 0.250),
    ("去除视觉语义", 0.703, 0.500),
    ("去除频域纹理", 0.636, 0.250),
    ("去除压缩痕迹", 0.665, 0.250),
    ("去除传播扰动", 0.500, 0.250),
]

CLEAN_VALIDATION_METRICS = [
    ("总体正确率", 0.708),
    ("多类别平均F1", 0.503),
    ("GPT-image2\n查准率", 0.956),
    ("GPT-image2\n召回率", 0.915),
    ("真实图\n查准率", 0.588),
    ("真实图\n召回率", 0.833),
]

BINARY_SUMMARY_METRICS = [
    ("推荐阈值", 0.650, "阈值"),
    ("扰动平均\n曲线下面积", 0.971, "扰动平均"),
    ("扰动平均\n二分类F1", 0.815, "扰动平均"),
    ("扰动平均\n真实图误报率", 0.033, "越低越好"),
    ("来源留出\n真实图误报率", 0.179, "仍需验证"),
    ("来源留出\n生成图召回率", 0.391, "仍需验证"),
]

GPT_SPECIALIST_METRICS = [
    ("同池GPT-image2\n查准率", 1.000, "内部验证"),
    ("同池GPT-image2\n召回率", 0.962, "内部验证"),
    ("候选验证\n平均F1", 0.690, "验证集"),
    ("来源留出\n平均F1", 0.275, "未达 0.450"),
    ("覆盖标签\n平均F1", 0.618, "诊断"),
    ("来源留出\n真实图误报率", 0.056, "诊断"),
]

PLATFORM_HOLDOUT_RESULTS = [
    ("原图", 0.667, 0.933, 0.000, 0.000),
    ("微博下载", 0.333, 0.867, 0.000, 0.000),
    ("小红书下载", 0.667, 0.933, 0.000, 0.000),
]

PLATFORM_REVERSE_RESULTS = [
    ("原图\n反向", 0.667, 0.800, 0.000, 0.000),
    ("微博下载\n反向", 0.200, 0.933, 0.000, 0.000),
    ("小红书下载\n反向", 0.667, 0.800, 0.000, 0.000),
]

PLATFORM_ARTIFACT_MATRIX = [
    ("微博下载", 0.00, 1.00, 0.967, 22 / 60),
    ("小红书下载", 58 / 60, 1.00, 1.000, 0.00),
    ("微博截图", 0.00, 0.00, 0.267, 50 / 60),
]

CURRENT_LABEL_COUNTS = [
    ("gpt-image2", 1151),
    ("real", 1060),
    ("seedream-4", 521),
    ("nano-banana", 464),
    ("midjourney", 451),
    ("sdxl", 410),
    ("flux", 365),
    ("dall-e-3", 353),
    ("sd3", 342),
    ("sd21", 326),
    ("gpt-image1", 163),
    ("gpt-image1.5", 100),
    ("stable-diffusion", 98),
    ("unknown", 40),
    ("imagegbt", 37),
    ("dall-e", 33),
]

FOCUSED_EXPERIMENT_COUNTS = [
    ("gpt-image2", 1151),
    ("real", 1060),
    ("Stable Diffusion 家族", 1176),
    ("seedream-4/豆包", 521),
    ("nano-banana", 464),
    ("midjourney", 451),
]

ACTIVE_LABEL_COUNTS = {
    "real": 1025,
    "gpt-image2": 838,
    "midjourney": 389,
    "sdxl": 364,
    "dall-e-3": 353,
    "flux": 350,
    "sd21": 326,
    "sd3": 326,
    "nano-banana": 214,
    "seedream-4": 178,
    "gpt-image1": 163,
    "gpt-image1.5": 100,
    "imagegbt": 37,
    "stable-diffusion": 28,
}

VALIDATION_LABEL_COUNTS = [
    ("gpt-image2", 47),
    ("real", 12),
    ("seedream-4", 11),
    ("flux", 10),
    ("nano-banana", 8),
    ("gpt-image1", 7),
    ("gpt-image1.5", 6),
    ("sd21", 4),
    ("sd3", 4),
    ("imagegbt", 3),
]

TOP_SOURCES = [
    ("Defactify", 1577, "DALL-E-3 / Midjourney / real / SD"),
    ("Scam-AI gpt-image-2", 738, "GPT-image2"),
    ("Qwen-Image-Bench", 600, "Flux / GPT-image / Nano / Seedream"),
    ("DeepSafe", 578, "DALL-E-3 / Flux / Midjourney / real / SD"),
    ("Liminal-Dreamcore", 313, "GPT-image2"),
    ("Robo531", 312, "Flux / ImageGBT / Nano / real / SDXL"),
]

PERTURBATION_PROTOCOL_FIGURE = [
    ("clean", "原始导入图像", "探针子集基线"),
    ("jpeg_q85", "轻度 JPEG 转码", "一次转发近似"),
    ("jpeg_q60", "强 JPEG 压缩", "多次转发近似"),
    ("screenshot", "截图后重保存", "当前最弱条件"),
    ("center_crop", "中心裁剪", "平台/用户裁切"),
    ("watermark", "角标水印覆盖", "平台标识遮挡"),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        return ImageFont.load_default()
    return ImageFont.truetype(str(path), size=size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=7)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        current = ""
        for char in raw_line:
            candidate = current + char
            if current and _text_size(draw, candidate, font)[0] > max_width:
                lines.append(current)
                current = char
            else:
                current = candidate
        if current:
            lines.append(current)
    return "\n".join(lines)


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int] = INK,
    pad: int = 28,
) -> None:
    x0, y0, x1, y1 = box
    wrapped = _wrap(draw, text, font, max(10, x1 - x0 - pad * 2))
    tw, th = _text_size(draw, wrapped, font)
    draw.multiline_text(
        (x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 2),
        wrapped,
        font=font,
        fill=fill,
        spacing=7,
        align="center",
    )


def _left_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int] = INK,
    width: int = 360,
) -> None:
    draw.multiline_text(xy, _wrap(draw, text, font, width), font=font, fill=fill, spacing=7)


def _box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = LINE,
    text_fill: tuple[int, int, int] = INK,
    radius: int = 10,
    width: int = 2,
    pad: int = 28,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
    _center_text(draw, xy, text, font, fill=text_fill, pad=pad)


def _architecture_card(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: list[str],
    *,
    fill: tuple[int, int, int] = WHITE,
    outline: tuple[int, int, int] = LINE,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=8, fill=fill, outline=outline, width=2)
    title_font = _font(24, True)
    body_font = _font(20)
    draw.text((x0 + 26, y0 + 20), title, font=title_font, fill=INK)
    draw.line((x0 + 24, y0 + 58, x1 - 24, y0 + 58), fill=outline, width=2)
    body = "\n".join(lines)
    draw.multiline_text(
        (x0 + 26, y0 + 76),
        _wrap(draw, body, body_font, x1 - x0 - 52),
        font=body_font,
        fill=INK,
        spacing=8,
    )


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], width: int = 4) -> None:
    draw.line([start, end], fill=LINE, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - direction * 18, ey - 10), (ex - direction * 18, ey + 10)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 10, ey - direction * 18), (ex + 10, ey - direction * 18)]
    draw.polygon(points, fill=LINE)


def _blend(c1: tuple[int, int, int], c2: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _metric_color(value: float, *, reverse: bool = False) -> tuple[int, int, int]:
    if reverse:
        return _blend(OK_LIGHT, RED_LIGHT, value)
    if value < 0.45:
        return _blend(RED_LIGHT, WARN_LIGHT, value / 0.45)
    if value < 0.75:
        return _blend(WARN_LIGHT, BLUE_LIGHT, (value - 0.45) / 0.30)
    return _blend(BLUE_LIGHT, OK_LIGHT, (value - 0.75) / 0.25)


def _format_metric(value: float) -> str:
    return f"{value:.3f}"


def _base_canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((54, 34), title, font=_font(32, True), fill=INK)
    draw.line((54, 84, width - 54, 84), fill=MID, width=3)
    return image, draw


def _save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=MPL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _axis_note(fig: plt.Figure, note: str) -> None:
    fig.text(0.02, 0.012, note, ha="left", va="bottom", fontsize=9.5, color=MPL_GRAY)


def _mpl_barh(
    path: Path,
    title: str,
    rows: list[tuple[str, float]],
    *,
    color: str = MPL_BLUE,
    note: str = "",
    xmax: float | None = None,
    figsize: tuple[float, float] = (9.6, 4.6),
    integer_values: bool = False,
) -> None:
    labels = [label for label, _ in rows]
    values = np.array([value for _, value in rows], dtype=float)
    xmax = float(xmax if xmax is not None else max(values) * 1.12)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=color, alpha=0.88, height=0.66)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="x", color=MPL_LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        label = f"{int(value)}" if integer_values else f"{value:.3f}"
        ax.text(min(value + xmax * 0.015, xmax * 0.985), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9.5, color="#18181B")
    if note:
        _axis_note(fig, note)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    _save_fig(fig, path)


def _mpl_grouped_bar(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    *,
    note: str = "",
    ylim: tuple[float, float] | None = (0, 1.05),
    figsize: tuple[float, float] = (10.4, 4.8),
    rotate: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = min(0.8 / max(len(series), 1), 0.28)
    offset0 = -width * (len(series) - 1) / 2
    for i, (name, values, color) in enumerate(series):
        ax.bar(x + offset0 + i * width, values, width, label=name, color=color, alpha=0.88)
    ax.set_xticks(x, labels, rotation=rotate, ha="right" if rotate else "center")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", pad=10)
    ax.grid(axis="y", color=MPL_LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=min(3, len(series)), loc="upper right")
    if note:
        _axis_note(fig, note)
    fig.tight_layout(rect=(0, 0.07 if note else 0, 1, 1))
    _save_fig(fig, path)


def _mpl_heatmap(
    path: Path,
    title: str,
    data: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    *,
    note: str = "",
    cmap: str = "YlGnBu",
    vmin: float = 0,
    vmax: float = 1,
    figsize: tuple[float, float] = (9.8, 5.0),
    fmt: str = ".2f",
) -> None:
    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", pad=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = (vmin + vmax) / 2
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            color = "white" if value > threshold else "#18181B"
            ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=9.5, color=color, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_visible(False)
    if note:
        _axis_note(fig, note)
    fig.tight_layout(rect=(0, 0.07 if note else 0, 1, 1))
    _save_fig(fig, path)


def _mpl_binary_matrix(
    path: Path,
    title: str,
    data: list[list[int]],
    row_labels: list[str],
    col_labels: list[str],
    *,
    note: str = "",
    figsize: tuple[float, float] = (9.8, 4.8),
) -> None:
    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title, loc="left", pad=10)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(arr.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(arr.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            mark = "●" if arr[i, j] else ""
            ax.text(j, i, mark, ha="center", va="center", fontsize=15, color="white" if arr[i, j] else "#18181B", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, ticks=[0, 1])
    cbar.ax.set_yticklabels(["无", "有"])
    cbar.outline.set_visible(False)
    if note:
        _axis_note(fig, note)
    fig.tight_layout(rect=(0, 0.07 if note else 0, 1, 1))
    _save_fig(fig, path)


def _pipeline(path: Path) -> None:
    image, draw = _base_canvas(1800, 430, "警务线索筛查与证据链组织流程")
    labels = [
        "公开传播图片\nURL/图片/批量流",
        "原始保全\nhash/来源快照",
        "本地视觉研判\n疑似生成线索",
        "风险排序\n紧急程度",
        "证据链草稿\n版本/审计ID",
        "专家复核\n平台/原图/链路",
    ]
    box_w, box_h = 235, 104
    y = 178
    gap = 54
    start_x = 72
    f = _font(24, True)
    for i, label in enumerate(labels):
        x = start_x + i * (box_w + gap)
        fill = ACCENT_LIGHT if i in (2, 3) else LIGHT
        _box(draw, (x, y, x + box_w, y + box_h), label, font=f, fill=fill, outline=LINE)
        if i < len(labels) - 1:
            _arrow(draw, (x + box_w + 8, y + box_h // 2), (x + box_w + gap - 10, y + box_h // 2))
    _left_text(
        draw,
        (96, 330),
        "模型概率只进入线索排序和风险提示；最终判断依赖人工复核与多源证据。",
        _font(23),
        fill=MUTED,
        width=1500,
    )
    image.save(path)


def _paper_pipeline_matrix(path: Path) -> None:
    rows = ["公开传播图片", "原始保全", "视觉研判", "风险排序", "报告草稿", "人工复核"]
    cols = ["输入", "保全", "模型线索", "排序", "材料组织", "人工判断"]
    data = [
        [1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]
    _mpl_binary_matrix(
        path,
        "警务线索筛查流程的阶段覆盖矩阵",
        data,
        rows,
        cols,
        note="● 表示该阶段显式产生或保留该类信息；模型输出只进入线索排序与材料组织，最终判断保留在人工复核阶段。",
        figsize=(9.8, 4.4),
    )


def _system_architecture(path: Path) -> None:
    image, draw = _base_canvas(1800, 820, "系统总体架构与数据流")
    small_font = _font(18)

    main_y = 170
    support_y = 475
    card_w = 290
    card_h = 180
    gap = 52
    x_positions = [72, 72 + (card_w + gap), 72 + 2 * (card_w + gap), 72 + 3 * (card_w + gap), 72 + 4 * (card_w + gap)]

    cards = {
        "input": (x_positions[0], main_y, x_positions[0] + card_w, main_y + card_h),
        "frontend": (x_positions[1], main_y, x_positions[1] + card_w, main_y + card_h),
        "backend": (x_positions[2], main_y, x_positions[2] + card_w, main_y + card_h),
        "model": (x_positions[3], main_y, x_positions[3] + card_w, main_y + card_h),
        "review": (x_positions[4], main_y, x_positions[4] + card_w, main_y + card_h),
        "store": (x_positions[1], support_y, x_positions[1] + card_w, support_y + card_h),
        "eval": (x_positions[2], support_y, x_positions[2] + card_w, support_y + card_h),
        "report": (x_positions[3], support_y, x_positions[3] + card_w, support_y + card_h),
    }

    _architecture_card(draw, cards["input"], "输入材料", ["图片文件", "URL 与来源页面", "平台回收样本"], fill=LIGHT)
    _architecture_card(draw, cards["frontend"], "前端工作台", ["案例列表", "上传分析", "结果展示与复核入口"], fill=ACCENT_LIGHT, outline=ACCENT)
    _architecture_card(draw, cards["backend"], "后端服务", ["文件保全", "接口编排", "训练与评测任务"], fill=BLUE_LIGHT, outline=BLUE)
    _architecture_card(draw, cards["model"], "视觉检测组件", ["115 维图像特征", "GPT-image2 候选概率", "低置信提示"], fill=OK_LIGHT, outline=OK)
    _architecture_card(draw, cards["review"], "人工复核", ["原图与平台链路", "水印 / C2PA", "处置建议"], fill=LIGHT)
    _architecture_card(draw, cards["store"], "数据与证据链", ["SQLite + 文件系统", "sha256 / 模型版本", "审计 ID / 报告草稿"], fill=PURPLE_LIGHT, outline=PURPLE)
    _architecture_card(draw, cards["eval"], "评测与模型治理", ["平台回收样本", "半数校准 / 半数留出", "候选/启用生命周期"], fill=WARN_LIGHT, outline=WARN)
    _architecture_card(draw, cards["report"], "输出材料", ["模型分析记录", "风险排序", "可编辑报告草稿"], fill=WHITE, outline=LINE)

    for left, right in [("input", "frontend"), ("frontend", "backend"), ("backend", "model"), ("model", "review")]:
        lx0, ly0, lx1, ly1 = cards[left]
        rx0, ry0, rx1, ry1 = cards[right]
        _arrow(draw, (lx1 + 8, (ly0 + ly1) // 2), (rx0 - 12, (ry0 + ry1) // 2), width=5)

    _arrow(draw, ((cards["backend"][0] + cards["backend"][2]) // 2, cards["backend"][3] + 12), ((cards["eval"][0] + cards["eval"][2]) // 2, cards["eval"][1] - 12), width=4)
    _arrow(draw, ((cards["backend"][0] + cards["backend"][2]) // 2 - 44, cards["backend"][3] + 12), ((cards["store"][0] + cards["store"][2]) // 2, cards["store"][1] - 12), width=4)
    _arrow(draw, ((cards["model"][0] + cards["model"][2]) // 2, cards["model"][3] + 12), ((cards["report"][0] + cards["report"][2]) // 2, cards["report"][1] - 12), width=4)
    _arrow(draw, (cards["store"][2] + 10, (cards["store"][1] + cards["store"][3]) // 2), (cards["eval"][0] - 12, (cards["eval"][1] + cards["eval"][3]) // 2), width=4)
    _arrow(draw, (cards["eval"][2] + 10, (cards["eval"][1] + cards["eval"][3]) // 2), (cards["report"][0] - 12, (cards["report"][1] + cards["report"][3]) // 2), width=4)

    _left_text(
        draw,
        (90, 720),
        "数据流：输入材料经前端进入后端保全；视觉检测组件输出 GPT-image2 可疑线索；模型版本、文件哈希和审计记录写入数据层；评测脚本管理候选/启用模型；输出材料进入人工复核。",
        small_font,
        fill=MUTED,
        width=1580,
    )
    image.save(path)


def _feature_logic(path: Path) -> None:
    image, draw = _base_canvas(1800, 720, "本地视觉检测组件与特征来源")
    body_font = _font(24)
    bold_font = _font(26, True)

    feature_groups = [
        ("视觉语义代理", "CLIP 相似度\n图文上下文差异"),
        ("文字上下文代理", "文字海报\n水印/角标信号"),
        ("频域与纹理", "频谱能量\n边缘与纹理残差"),
        ("压缩痕迹", "JPEG 块效应\n字节分布/格式"),
        ("传播扰动代理", "截图形态\n尺寸/裁剪/水印"),
    ]
    x0, y0, w, h, gap = 70, 135, 300, 88, 24
    for i, (title, detail) in enumerate(feature_groups):
        y = y0 + i * (h + gap)
        _box(draw, (x0, y, x0 + w, y + h), f"{title}\n{detail}", font=body_font, fill=WHITE)

    _box(
        draw,
        (630, 220, 1065, 500),
        "115 维图像特征\n\n标准化\n类别原型兜底\nExtraTreesClassifier\n低置信阈值",
        font=bold_font,
        fill=ACCENT_LIGHT,
        outline=ACCENT,
    )

    outputs = [
        ("候选类别", "疑似生成/真实/未知类"),
        ("候选来源", "GPT-image2 等线索"),
        ("风险提示", "置信度与不确定度"),
        ("审计记录", "模型版本/训练快照"),
    ]
    rx, rw, rh = 1310, 375, 78
    for i, (title, detail) in enumerate(outputs):
        y = 170 + i * 104
        _box(draw, (rx, y, rx + rw, y + rh), f"{title}\n{detail}", font=body_font, fill=LIGHT)

    _arrow(draw, (394, 370), (606, 370), width=5)
    _arrow(draw, (1090, 370), (1282, 370), width=5)
    _left_text(
        draw,
        (665, 555),
        "防泄漏约束：标签、数据来源和来源细节只作为监督与审计字段，不作为模型输入；文本上下文中的生成器名称会被清洗。",
        _font(22),
        fill=MUTED,
        width=920,
    )
    image.save(path)


def _paper_feature_matrix(path: Path) -> None:
    rows = ["视觉语义代理", "文字上下文代理", "频域与纹理", "压缩痕迹", "传播扰动代理"]
    cols = ["语义线索", "文字/水印", "纹理频域", "压缩痕迹", "尺寸形态", "泄漏清洗"]
    data = [
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1],
    ]
    _mpl_binary_matrix(
        path,
        "115 维特征组与信号类型覆盖矩阵",
        data,
        rows,
        cols,
        note="矩阵展示特征组的取证信号覆盖范围；标签、来源字段和生成器名称不作为模型输入。",
        figsize=(10.2, 4.2),
    )


def _evaluation_logic(path: Path) -> None:
    image, draw = _base_canvas(1800, 680, "实验协议与能力边界")
    f = _font(22)
    bf = _font(25, True)
    rows = [
        ("同池验证", "120 条无扰动留出样本\n基础能力检查", OK_LIGHT, OK),
        ("扰动复测", "JPEG/截图/裁剪/水印\n传播后性能变化", ACCENT_LIGHT, ACCENT),
        ("按来源留出", "按来源整组留出\n跨来源泛化诊断", WARN_LIGHT, WARN),
        ("阈值校准", "控制真实图误报\n低误报初筛门控", LIGHT, LINE),
        ("特征消融", "移除/保留特征组\n解释模型依赖", WHITE, LINE),
    ]
    positions = [
        (150, 145, 460, 255),
        (570, 145, 880, 255),
        (990, 145, 1300, 255),
        (360, 315, 670, 425),
        (780, 315, 1090, 425),
    ]
    for i, (title, detail, fill, outline) in enumerate(rows):
        bx0, by0, bx1, by1 = positions[i]
        _box(draw, (bx0, by0, bx1, by1), f"{title}\n{detail}", font=f, fill=fill, outline=outline, pad=34)
        _arrow(draw, ((bx0 + bx1) // 2, by1 + 16), ((bx0 + bx1) // 2, 495), width=4)

    _box(
        draw,
        (240, 505, 1560, 585),
        "论文结论只对对应实验条件负责：同池分数说明快照内部可用性；按来源留出和扰动结果决定真实部署边界。",
        font=bf,
        fill=LIGHT,
        outline=LINE,
        pad=54,
    )
    _left_text(
        draw,
        (300, 622),
        "口径：强项与弱项同时报告；GPT-image2 是疑似来源线索，二分类阈值只作筛查门控。",
        _font(23),
        fill=MUTED,
        width=1180,
    )
    image.save(path)


def _paper_evaluation_coverage_matrix(path: Path) -> None:
    rows = ["平台回收样本", "转码痕迹观察", "合成平台扰动", "候选模型训练", "半数校准/半数留出", "人工复核输出"]
    cols = ["真实平台", "训练增强", "留出评测", "低误报", "审计记录"]
    data = [
        [1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 1, 1],
        [1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]
    _mpl_binary_matrix(
        path,
        "平台下载转码增强评测协议",
        data,
        rows,
        cols,
        note="真实平台回收样本只用于黑盒观察、阈值校准与留出评测；训练侧使用可观察痕迹参数化后的合成平台扰动。",
        figsize=(10.2, 4.8),
    )


def _deployment_loop(path: Path) -> None:
    image, draw = _base_canvas(1800, 560, "最终应用形态：连续监测与人工筛查")
    f = _font(24)
    bf = _font(26, True)
    left = [
        ("合法数据权限", "公开传播流\n平台协查接口"),
        ("24 小时监测", "图片采集\n去重聚合"),
        ("每日线索清单", "按紧急程度\n排序预警"),
    ]
    right = [
        ("民警专家复核", "原图/链路/元数据\n人工判断"),
        ("处置建议", "辟谣/协查/留证\n分级响应"),
    ]
    for i, (title, detail) in enumerate(left):
        y = 145 + i * 112
        _box(draw, (95, y, 410, y + 82), f"{title}\n{detail}", font=f, fill=ACCENT_LIGHT, outline=ACCENT)
        if i < len(left) - 1:
            _arrow(draw, (252, y + 88), (252, y + 104), width=4)
    _box(
        draw,
        (645, 195, 1165, 365),
        "系统作用\n把海量图片压缩为高风险可疑线索\n提高专家深度研判效率",
        font=bf,
        fill=LIGHT,
        outline=LINE,
    )
    _arrow(draw, (420, 300), (620, 300), width=5)
    _arrow(draw, (1190, 300), (1365, 245), width=5)
    _arrow(draw, (1190, 300), (1365, 355), width=5)
    for i, (title, detail) in enumerate(right):
        y = 200 + i * 122
        _box(draw, (1390, y, 1700, y + 86), f"{title}\n{detail}", font=f, fill=OK_LIGHT, outline=OK)
    _left_text(
        draw,
        (590, 430),
        "边界：模型概率不是证据结论；系统提供线索排序、材料整理和早期预警，最终结论仍由人工结合多源证据作出。",
        _font(23),
        fill=MUTED,
        width=1050,
    )
    image.save(path)


def _paper_deployment_matrix(path: Path) -> None:
    rows = ["24h 公开流监测", "去重聚合", "模型初筛", "紧急程度排序", "专家复核", "处置建议"]
    cols = ["高通量", "低误报", "可追溯", "人工复核", "预警输出"]
    data = [
        [1, 0, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1],
    ]
    _mpl_binary_matrix(
        path,
        "连续监测场景下的任务-能力矩阵",
        data,
        rows,
        cols,
        note="系统价值在于把海量公开传播图片压缩为可疑线索清单；模型概率不替代专家结论。",
        figsize=(9.8, 4.2),
    )


def _metric_cards(
    path: Path,
    title: str,
    metrics: list[tuple[str, float] | tuple[str, float, str]],
    footer: str,
    *,
    reverse_labels: tuple[str, ...] = ("FPR",),
) -> None:
    image, draw = _base_canvas(1800, 560, title)
    x0, y0 = 105, 145
    card_w, card_h = 245, 165
    gap = 32
    for i, metric in enumerate(metrics):
        label = metric[0]
        value = metric[1]
        note = metric[2] if len(metric) > 2 else ""
        reverse = any(key in label for key in reverse_labels)
        fill = _metric_color(value, reverse=reverse)
        x = x0 + i * (card_w + gap)
        draw.rounded_rectangle((x, y0, x + card_w, y0 + card_h), radius=14, fill=fill, outline=LINE, width=2)
        _center_text(draw, (x + 12, y0 + 14, x + card_w - 12, y0 + 78), label, _font(23, True), fill=INK, pad=4)
        _center_text(draw, (x + 12, y0 + 72, x + card_w - 12, y0 + 128), _format_metric(value), _font(38, True), fill=INK, pad=4)
        if note:
            _center_text(draw, (x + 12, y0 + 126, x + card_w - 12, y0 + card_h - 10), note, _font(18), fill=MUTED, pad=4)
    _box(draw, (230, 390, 1570, 480), footer, font=_font(23), fill=LIGHT, outline=LINE)
    image.save(path)


def _horizontal_bar_chart(
    path: Path,
    title: str,
    rows: list[tuple[str, float]],
    *,
    footer: str,
    value_suffix: str = "",
    color: tuple[int, int, int] = BLUE,
    max_value: float | None = None,
    height: int = 760,
) -> None:
    image, draw = _base_canvas(1800, height, title)
    max_value = max_value or max(value for _, value in rows)
    x0, y0 = 430, 135
    chart_w = 1080
    available_h = max(220, height - y0 - 115)
    gap = 7 if len(rows) > 8 else 12
    row_h = min(54, max(26, int((available_h - gap * (len(rows) - 1)) / max(len(rows), 1))))
    label_font = _font(22 if len(rows) <= 8 else 18 if len(rows) <= 12 else 16)
    value_font = _font(20 if len(rows) <= 8 else 17 if len(rows) <= 12 else 15, True)
    for i, (label, value) in enumerate(rows):
        y = y0 + i * (row_h + gap)
        draw.text((105, y + (row_h - _text_size(draw, label, label_font)[1]) / 2), label, font=label_font, fill=INK)
        draw.rounded_rectangle((x0, y, x0 + chart_w, y + row_h), radius=8, fill=LIGHT, outline=MID, width=1)
        bar_w = int(chart_w * value / max(max_value, 1))
        fill = _blend(BLUE_LIGHT, color, min(1.0, value / max(max_value, 1)))
        draw.rounded_rectangle((x0, y, x0 + bar_w, y + row_h), radius=8, fill=fill)
        draw.text((x0 + chart_w + 28, y + (row_h - 23) / 2), f"{value:g}{value_suffix}", font=value_font, fill=INK)
    _left_text(draw, (105, height - 80), footer, _font(22), fill=MUTED, width=1500)
    image.save(path)


def _label_distribution_chart(path: Path) -> None:
    rows = [(label, float(count)) for label, count in CURRENT_LABEL_COUNTS]
    _mpl_barh(
        path,
        "全量外部审计数据池标签分布",
        rows,
        note="全量审计池共 5914 张；辅助标签保留用于开放集未知类、辅助负类和边界分析，不等于本文核心实验类别。",
        color=MPL_TEAL,
        integer_values=True,
        figsize=(9.6, 5.4),
    )


def _focused_experiment_pool_chart(path: Path) -> None:
    rows = [(label, float(count)) for label, count in FOCUSED_EXPERIMENT_COUNTS]
    _mpl_barh(
        path,
        "本文有效实验子池标签分布",
        rows,
        note="有效实验子池共 4823 张，聚焦 GPT-image2 检测；其他代表性生成器主要承担对照、辅助负类和边界分析作用。",
        color=MPL_GREEN,
        integer_values=True,
        figsize=(9.4, 4.2),
    )


def _snapshot_comparison_chart(path: Path) -> None:
    labels = [label for label, _ in CURRENT_LABEL_COUNTS if label in ACTIVE_LABEL_COUNTS]
    current = [dict(CURRENT_LABEL_COUNTS)[label] for label in labels]
    active = [ACTIVE_LABEL_COUNTS.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    y = np.arange(len(labels))
    ax.barh(y - 0.18, current, height=0.34, label="全量审计池", color=MPL_BLUE, alpha=0.86)
    ax.barh(y + 0.18, active, height=0.34, label="历史 active 快照", color=MPL_ORANGE, alpha=0.86)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title("全量审计池与历史 active 快照对比", loc="left", pad=10)
    ax.grid(axis="x", color=MPL_LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right")
    _axis_note(fig, "历史 active 快照只用于解释既有模型表现，不代表本文继续追求全量标签封闭类别归因。")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save_fig(fig, path)


def _top_sources_chart(path: Path) -> None:
    rows = [(label, float(count)) for label, count, _ in TOP_SOURCES]
    _mpl_barh(
        path,
        "主要数据来源样本量",
        rows,
        note="大来源组提高样本规模，也会带来来源风格耦合；因此本文把来源互留作为关键诊断。",
        color=MPL_PURPLE,
        integer_values=True,
        figsize=(9.4, 4.2),
    )


def _training_snapshot_diagram(path: Path) -> None:
    rows = [
        ("外部数据池", 5914),
        ("冻结训练快照", 4691),
        ("clean 训练", 4571),
        ("同池验证", 120),
        ("临时扰动增强特征", 1200),
        ("特征维度", 115),
    ]
    _mpl_barh(
        path,
        "启用模型训练快照与配置量化摘要",
        [(label, float(value)) for label, value in rows],
        note="模型为 ExtraTreesClassifier：n_estimators=360, min_samples_leaf=2, max_features=sqrt, class_weight=balanced，低置信阈值为0.082。",
        color=MPL_BLUE,
        integer_values=True,
        figsize=(9.6, 4.3),
    )


def _validation_distribution_chart(path: Path) -> None:
    rows = [(label, float(count)) for label, count in VALIDATION_LABEL_COUNTS]
    _mpl_barh(
        path,
        "同池验证集标签分布（Top 10）",
        rows,
        note="主验证集 n=120，GPT-image2 样本数较高，而真实图样本数为12；因此多类别平均F1与真实图误报边界必须结合其他评测读取。",
        color=MPL_ORANGE,
        integer_values=True,
        figsize=(9.4, 4.2),
    )


def _perturbation_protocol_diagram(path: Path) -> None:
    rows = ["无扰动", "轻度JPEG", "强JPEG", "截图重保存", "中心裁剪", "水印覆盖"]
    cols = ["JPEG 转码", "截图重存", "中心裁剪", "水印覆盖", "探针基线", "弱项标记"]
    data = [
        [0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ]
    _mpl_binary_matrix(
        path,
        "传播扰动复测条件设计矩阵",
        data,
        rows,
        cols,
        note="同一探针子集分别施加扰动，并统计总体正确率、多类别平均F1、GPT-image2召回率和平均置信度；该矩阵用于条件敏感性分析。",
        figsize=(10.2, 4.4),
    )


def _clean_validation_bars(path: Path) -> None:
    rows = [(label.replace("\n", " "), value) for label, value in CLEAN_VALIDATION_METRICS]
    _mpl_barh(
        path,
        "历史启用模型的同池验证指标",
        rows,
        note="120 条同池主验证集。GPT-image2 线索较强，但多类别平均F1为0.503，真实图样本数为12，显示历史多标签快照诊断仍不均衡。",
        color=MPL_BLUE,
        xmax=1.0,
        figsize=(9.4, 4.0),
    )


def _source_holdout_bars(path: Path) -> None:
    rows = [(label.replace("\n", " "), value) for label, value in SOURCE_HOLDOUT_METRICS]
    _mpl_barh(
        path,
        "按来源留出验证的跨来源诊断",
        rows,
        note="严格多类别平均F1为0.124，真实图误报率为0.242，表明跨来源泛化和真实图误报控制仍需继续验证。",
        color=MPL_RED,
        xmax=1.0,
        figsize=(9.4, 4.2),
    )


def _binary_summary_bars(path: Path) -> None:
    rows = [
        ("推荐阈值", 0.650),
        ("扰动平均曲线下面积", 0.971),
        ("扰动平均二分类F1", 0.815),
        ("扰动平均真实图误报率", 0.033),
        ("来源留出真实图误报率", 0.179),
        ("来源留出生成图召回率", 0.391),
    ]
    colors = [MPL_BLUE, MPL_GREEN, MPL_GREEN, MPL_GREEN, MPL_RED, MPL_ORANGE]
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    y = np.arange(len(rows))
    bars = ax.barh(y, values, color=colors, alpha=0.88, height=0.66)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_title("二分类门控：阈值校准与跨来源落差", loc="left", pad=10)
    ax.grid(axis="x", color=MPL_LIGHT_GRID, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9.5)
    _axis_note(fig, "扰动验证支持低误报初筛方向，但来源留出条件下真实图误报率与生成图召回率仍需继续优化。")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save_fig(fig, path)


def _gpt_specialist_bars(path: Path) -> None:
    rows = [(label.replace("\n", " "), value) for label, value, _ in GPT_SPECIALIST_METRICS]
    _mpl_barh(
        path,
        "GPT-image2 专项候选：内部表现与来源互留",
        rows,
        note="内部条件下线索明显，但来源留出平均F1为0.275，尚未达到0.450门槛，当前保留为候选组件。",
        color=MPL_PURPLE,
        xmax=1.0,
        figsize=(9.6, 4.2),
    )


def _platform_artifact_heatmap(path: Path) -> None:
    metrics = ["同 SHA 比例", "同尺寸比例", "字节比中位数", "量化表变化比例"]
    _mpl_heatmap(
        path,
        "平台回收样本可观察文件变化",
        [list(row[1:]) for row in PLATFORM_ARTIFACT_MATRIX],
        [row[0] for row in PLATFORM_ARTIFACT_MATRIX],
        metrics,
        note="60 对小规模黑盒回收样本。微博下载呈保留尺寸 JPEG 重编码痕迹；小红书下载本次基本等价原图；微博截图作为低分辨率失败边界。",
        cmap="YlOrBr",
        figsize=(10.4, 4.4),
        fmt=".2f",
    )


def _platform_enhancement_bars(path: Path) -> None:
    labels = [row[0] for row in PLATFORM_HOLDOUT_RESULTS]
    series = [
        ("启用模型召回率", [row[1] for row in PLATFORM_HOLDOUT_RESULTS], MPL_GRAY),
        ("候选模型召回率", [row[2] for row in PLATFORM_HOLDOUT_RESULTS], MPL_GREEN),
        ("启用模型误报率", [row[3] for row in PLATFORM_HOLDOUT_RESULTS], MPL_RED),
        ("候选模型误报率", [row[4] for row in PLATFORM_HOLDOUT_RESULTS], MPL_ORANGE),
    ]
    _mpl_grouped_bar(
        path,
        "平台下载留出测试：启用模型与候选模型对比",
        labels,
        series,
        note="官方切分：奇数30对校准阈值，偶数30对汇报。候选模型=faa78335-c4c5-4825-9095-13779af5cfec；启用模型未被自动替换。",
        figsize=(10.4, 4.8),
        rotate=12,
    )


def _platform_reverse_bars(path: Path) -> None:
    labels = [row[0] for row in PLATFORM_REVERSE_RESULTS]
    series = [
        ("启用模型召回率", [row[1] for row in PLATFORM_REVERSE_RESULTS], MPL_GRAY),
        ("候选模型召回率", [row[2] for row in PLATFORM_REVERSE_RESULTS], MPL_GREEN),
        ("启用模型误报率", [row[3] for row in PLATFORM_REVERSE_RESULTS], MPL_RED),
        ("候选模型误报率", [row[4] for row in PLATFORM_REVERSE_RESULTS], MPL_ORANGE),
    ]
    _mpl_grouped_bar(
        path,
        "平台下载反向切分复核",
        labels,
        series,
        note="反向切分：偶数30对校准阈值，奇数30对汇报。三项条件误报率均保持 0.000，微博下载链路提升最明显。",
        figsize=(10.0, 4.4),
        rotate=12,
    )


def _perturbation_matrix(path: Path) -> None:
    metrics = ["总体正确率", "多类别平均F1", "GPT-image2召回率", "平均置信度"]
    _mpl_heatmap(
        path,
        "传播扰动探针热力矩阵",
        [list(row[1:]) for row in PERTURBATION_RESULTS],
        [row[0] for row in PERTURBATION_RESULTS],
        metrics,
        note="截图转存同时拉低多类别平均F1、GPT-image2召回率和平均置信度，是下一轮补样和困难负样本设计重点。",
        cmap="YlGnBu",
        figsize=(9.6, 4.8),
    )


def _source_holdout_matrix(path: Path) -> None:
    image, draw = _base_canvas(1800, 590, "按来源留出泛化诊断")
    x0, y0 = 100, 150
    card_w, card_h = 250, 165
    gap = 28
    f = _font(24)
    bf = _font(25, True)
    for i, (label, value) in enumerate(SOURCE_HOLDOUT_METRICS):
        x = x0 + i * (card_w + gap)
        reverse = "误报率" in label
        fill = _metric_color(value, reverse=reverse)
        draw.rounded_rectangle((x, y0, x + card_w, y0 + card_h), radius=12, fill=fill, outline=LINE, width=2)
        _center_text(draw, (x + 14, y0 + 15, x + card_w - 14, y0 + 82), label, bf, fill=INK, pad=8)
        _center_text(draw, (x + 14, y0 + 78, x + card_w - 14, y0 + card_h - 15), _format_metric(value), _font(38, True), fill=INK, pad=8)

    _box(
        draw,
        (210, 390, 1590, 480),
        "结论：按来源留出验证决定部署边界。严格多类别平均F1为0.124，真实图误报率为0.242，说明跨来源泛化和真实图误报控制仍需继续验证。",
        font=f,
        fill=LIGHT,
        outline=LINE,
    )
    image.save(path)


def _clean_validation_cards(path: Path) -> None:
    _metric_cards(
        path,
        "历史启用模型同池验证指标卡",
        CLEAN_VALIDATION_METRICS,
        "口径：120 条同池主验证集。GPT-image2 线索较强，但多类别平均F1为0.503，真实图样本数为12，说明历史多标签快照诊断仍不均衡。",
    )


def _binary_summary_cards(path: Path) -> None:
    _metric_cards(
        path,
        "真实/生成二分类门控摘要",
        BINARY_SUMMARY_METRICS,
        "口径：扰动验证显示低误报潜力，但来源留出真实图误报率为0.179、生成图召回率为0.391，跨来源表现仍需继续验证。",
    )


def _gpt_specialist_cards(path: Path) -> None:
    _metric_cards(
        path,
        "GPT-image2 专项候选指标卡",
        GPT_SPECIALIST_METRICS,
        "结论：内部条件下线索明显，来源留出平均F1为0.275，尚未达到0.450门槛，当前保留为候选组件。",
    )


def _binary_gate_chart(path: Path) -> None:
    labels = [row[0] for row in BINARY_GATE_RESULTS]
    series = [
        ("曲线下面积", [row[1] for row in BINARY_GATE_RESULTS], MPL_BLUE),
        ("二分类F1", [row[2] for row in BINARY_GATE_RESULTS], MPL_GREEN),
        ("生成图召回率", [row[3] for row in BINARY_GATE_RESULTS], MPL_ORANGE),
        ("真实图误报率", [row[4] for row in BINARY_GATE_RESULTS], MPL_RED),
    ]
    _mpl_grouped_bar(
        path,
        "二分类门控分条件表现",
        labels,
        series,
        note="同源扰动验证中真实图误报率较低，但生成图召回率在裁剪和截图条件下降明显；来源留出仍需单独验证。",
        figsize=(10.2, 4.6),
        rotate=20,
    )


def _feature_ablation_chart(path: Path) -> None:
    labels = [row[0].replace("\n", " ") for row in FEATURE_ABLATION]
    series = [
        ("多类别平均F1", [row[1] for row in FEATURE_ABLATION], MPL_BLUE),
        ("GPT-image2召回率", [row[2] for row in FEATURE_ABLATION], MPL_ORANGE),
    ]
    _mpl_grouped_bar(
        path,
        "特征消融对比",
        labels,
        series,
        note="去除若干特征后指标反升，提示当前特征可能与来源风格耦合；消融结果用于定位风险和补样方向。",
        figsize=(10.4, 4.8),
        rotate=24,
    )


def ensure_paper_figures() -> dict[str, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "system_architecture": FIGURE_DIR / "paper_system_architecture.png",
        "pipeline": FIGURE_DIR / "paper_pipeline_flow.png",
        "feature_logic": FIGURE_DIR / "paper_feature_logic.png",
        "evaluation_logic": FIGURE_DIR / "paper_evaluation_logic.png",
        "deployment_loop": FIGURE_DIR / "paper_deployment_loop.png",
        "label_distribution_chart": FIGURE_DIR / "paper_label_distribution_chart.png",
        "focused_experiment_pool_chart": FIGURE_DIR / "paper_focused_experiment_pool_chart.png",
        "snapshot_comparison_chart": FIGURE_DIR / "paper_snapshot_comparison_chart.png",
        "top_sources_chart": FIGURE_DIR / "paper_top_sources_chart.png",
        "training_snapshot_diagram": FIGURE_DIR / "paper_training_snapshot_diagram.png",
        "validation_distribution_chart": FIGURE_DIR / "paper_validation_distribution_chart.png",
        "perturbation_protocol_diagram": FIGURE_DIR / "paper_perturbation_protocol_diagram.png",
        "platform_artifact_heatmap": FIGURE_DIR / "paper_platform_artifact_heatmap.png",
        "platform_enhancement_bars": FIGURE_DIR / "paper_platform_enhancement_bars.png",
        "platform_reverse_bars": FIGURE_DIR / "paper_platform_reverse_bars.png",
        "clean_validation_cards": FIGURE_DIR / "paper_clean_validation_cards.png",
        "clean_validation_bars": FIGURE_DIR / "paper_clean_validation_bars.png",
        "perturbation_matrix": FIGURE_DIR / "paper_perturbation_matrix.png",
        "source_holdout_matrix": FIGURE_DIR / "paper_source_holdout_matrix.png",
        "source_holdout_bars": FIGURE_DIR / "paper_source_holdout_bars.png",
        "binary_summary_cards": FIGURE_DIR / "paper_binary_summary_cards.png",
        "binary_summary_bars": FIGURE_DIR / "paper_binary_summary_bars.png",
        "binary_gate_chart": FIGURE_DIR / "paper_binary_gate_chart.png",
        "feature_ablation_chart": FIGURE_DIR / "paper_feature_ablation_chart.png",
        "gpt_specialist_cards": FIGURE_DIR / "paper_gpt_specialist_cards.png",
        "gpt_specialist_bars": FIGURE_DIR / "paper_gpt_specialist_bars.png",
    }
    _system_architecture(outputs["system_architecture"])
    _paper_pipeline_matrix(outputs["pipeline"])
    _paper_feature_matrix(outputs["feature_logic"])
    _paper_evaluation_coverage_matrix(outputs["evaluation_logic"])
    _paper_deployment_matrix(outputs["deployment_loop"])
    _label_distribution_chart(outputs["label_distribution_chart"])
    _focused_experiment_pool_chart(outputs["focused_experiment_pool_chart"])
    _snapshot_comparison_chart(outputs["snapshot_comparison_chart"])
    _top_sources_chart(outputs["top_sources_chart"])
    _training_snapshot_diagram(outputs["training_snapshot_diagram"])
    _validation_distribution_chart(outputs["validation_distribution_chart"])
    _perturbation_protocol_diagram(outputs["perturbation_protocol_diagram"])
    _platform_artifact_heatmap(outputs["platform_artifact_heatmap"])
    _platform_enhancement_bars(outputs["platform_enhancement_bars"])
    _platform_reverse_bars(outputs["platform_reverse_bars"])
    _clean_validation_bars(outputs["clean_validation_bars"])
    _perturbation_matrix(outputs["perturbation_matrix"])
    _source_holdout_bars(outputs["source_holdout_bars"])
    _binary_summary_bars(outputs["binary_summary_bars"])
    _binary_gate_chart(outputs["binary_gate_chart"])
    _feature_ablation_chart(outputs["feature_ablation_chart"])
    _gpt_specialist_bars(outputs["gpt_specialist_bars"])
    return outputs
