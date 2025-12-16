#!/usr/bin/env python3
"""
DnD Treasure Card Generator

Generates PNG cards for treasure rewards from a JSON input, and optionally
arranges them into an A4 multi-page PDF (3x3 grid), mirroring the layout and
styling of monster_cards.py (same fonts, parchment/leather backgrounds, and
alternating front/back pages).

Each treasure card shows:
    - title (category, e.g., TREASURE CHEST REWARD)
    - name (e.g., WEATHERED LONGSWORD)
    - effect text (rules text)
    - lore blurb (italicized)
    - optional image (uses the same rounded portrait panel as monsters)

Usage:
    python treasure_cards.py input.json --outdir cards --pdf OUTFILE.pdf --dpi 300

Input JSON schema (list of treasures):
    {
      "id": "optional-id",
      "title": "TREASURE CHEST REWARD",   # optional, default "Treasure"
      "name": "WEATHERED LONGSWORD",      # required
      "effect": "Mechanical effect text.",
      "lore": "Italicized lore description.",
      "image": "optional/path/or/filename.png",  # optional
      "background": "#RRGGBB or image path",      # optional, like monsters
      "fonts": {                                   # optional per-card overrides
         "name": "path/to/font.ttf",
         "effect": "path/to/font.ttf",
         "lore": "path/to/font.ttf"
      }
    }

Dependencies:
    - Pillow
    - reportlab

This script intentionally duplicates some logic from monster_cards.py
(card sizing, PDF composition, font theme resolution) to keep it
standalone, but any changes there can be ported here mechanically.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageChops
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# --- Constants mirrored from monster_cards.py ---
A4_WIDTH_PX_300DPI = 2480
A4_HEIGHT_PX_300DPI = 3508
DEFAULT_GRID_COLS = 3
DEFAULT_GRID_ROWS = 3
DEFAULT_CARDS_PER_PAGE = DEFAULT_GRID_COLS * DEFAULT_GRID_ROWS
DEFAULT_MARGIN = 90
DEFAULT_GUTTER = 45
FONT_SCALE = 1.0


# ----------------------------- Utilities -------------------------------------

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_probably_grayscale(img_rgba: Image.Image, sample_step: int = 12, tolerance: int = 6) -> bool:
    """Heuristic: sample pixels; if R≈G≈B for most, treat as grayscale."""
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    w, h = img_rgba.size
    if w == 0 or h == 0:
        return True
    px = img_rgba.load()
    checked = 0
    colored = 0
    for y in range(0, h, max(1, sample_step)):
        for x in range(0, w, max(1, sample_step)):
            r, g, b, a = px[x, y]
            if a < 25:
                continue
            checked += 1
            if abs(r - g) > tolerance or abs(r - b) > tolerance or abs(g - b) > tolerance:
                colored += 1
            if checked >= 350:
                break
        if checked >= 350:
            break
    if checked == 0:
        return True
    # If more than ~3% of sampled opaque pixels are "colored", treat as color.
    return (colored / checked) <= 0.03


def to_black_and_white(img_rgba: Image.Image, contrast: float = 1.12) -> Image.Image:
    """Convert an RGBA image to grayscale while preserving alpha."""
    if img_rgba.mode != "RGBA":
        img_rgba = img_rgba.convert("RGBA")
    r, g, b, a = img_rgba.split()
    gray = ImageOps.grayscale(Image.merge("RGB", (r, g, b)))
    # Mild contrast boost helps small illustrations read better after desaturation.
    try:
        gray = ImageEnhance.Contrast(gray).enhance(contrast)
    except Exception:
        pass
    out = Image.merge("RGBA", (gray, gray, gray, a))
    return out


def safe_load_image(path_or_url: Optional[str], target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Same behavior as in monster_cards: load local image or provide placeholder."""
    placeholder_size = target_size or (256, 256)
    if not path_or_url:
        return placeholder_image(placeholder_size)
    p = str(path_or_url)
    if p.startswith("http://") or p.startswith("https://"):
        warn("Remote URLs for images are not fetched. Using placeholder.")
        return placeholder_image(placeholder_size)
    if not os.path.isfile(p):
        warn(f"Image not found: {p}. Using placeholder.")
        return placeholder_image(placeholder_size)
    try:
        img = Image.open(p).convert("RGBA")
        # Convert color images to black & white to match the card style.
        if not _is_probably_grayscale(img):
            img = to_black_and_white(img)
        if target_size:
            img = ImageOps.contain(img, target_size, method=Image.LANCZOS)
        return img
    except Exception as e:
        warn(f"Failed to load image {p}: {e}. Using placeholder.")
        return placeholder_image(placeholder_size)


def placeholder_image(size: Tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGBA", size, (200, 200, 200, 255))
    draw = ImageDraw.Draw(img)
    r = min(w, h) // 3
    cx, cy = w // 2, h // 3
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(160, 160, 160, 255))
    draw.rectangle((w * 0.3, h * 0.55, w * 0.7, h * 0.95), fill=(160, 160, 160, 255))
    draw.text((10, h - 24), "No Image", fill=(80, 80, 80, 255))
    return img


def parse_color(s: Optional[str], fallback=(245, 245, 245)) -> Tuple[int, int, int]:
    if not s:
        return fallback
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
        if len(s) == 3:
            r, g, b = [int(c * 2, 16) for c in s]
            return (r, g, b)
        if len(s) == 6:
            try:
                r = int(s[0:2], 16)
                g = int(s[2:4], 16)
                b = int(s[4:6], 16)
                return (r, g, b)
            except Exception:
                return fallback
    return fallback


def draw_rounded_image(base: Image.Image, img: Image.Image, box: Tuple[int, int, int, int], radius: int = 20) -> None:
    x0, y0, x1, y1 = map(int, box)
    w, h = x1 - x0, y1 - y0
    img_contained = ImageOps.contain(img, (w, h), method=Image.LANCZOS)
    iw, ih = img_contained.size
    ox = (w - iw) // 2
    oy = (h - ih) // 2
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(img_contained, (ox, oy), img_contained)
    round_mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(round_mask)
    mask_draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    current_alpha = canvas.split()[3]
    combined_alpha = ImageChops.multiply(current_alpha, round_mask)
    canvas.putalpha(combined_alpha)
    base.paste(canvas, (x0, y0), canvas)


def fit_font(fontpath_or_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if fontpath_or_name:
        if os.path.isfile(fontpath_or_name):
            try:
                return ImageFont.truetype(fontpath_or_name, size=size)
            except Exception as e:
                warn(f"Failed to load font from path {fontpath_or_name}: {e}")
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def find_font_file(candidates: List[str]) -> Optional[str]:
    search_roots = [os.path.join(os.path.dirname(__file__), "fonts"), os.getcwd()]
    for root in search_roots:
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None


@dataclass
class FontsConfig:
    default: Optional[str] = None
    name: Optional[str] = None
    lore: Optional[str] = None
    stats: Optional[str] = None


def resolve_theme_fonts(theme: Optional[str]) -> Optional[FontsConfig]:
    if not theme:
        return None
    theme = theme.lower().strip()
    THEMES = {
        "fantasy": {
            "name": [
                "CinzelDecorative-Black.ttf",
                "CinzelDecorative-Bold.ttf",
                "UncialAntiqua-Regular.ttf",
                "MedievalSharp-Regular.ttf",
            ],
            "lore": [
                "Cardo-Italic.ttf",
                "EBGaramond-Italic.ttf",
                "IMFellEnglish-Italic.ttf",
                "CormorantGaramond-Italic.ttf",
            ],
            "stats": [
                "Cinzel-Bold.ttf",
                "Cinzel-Regular.ttf",
                "Caudex-Bold.ttf",
                "CormorantGaramond-Bold.ttf",
            ],
            "default": [
                "CormorantGaramond-Regular.ttf",
                "Cardo-Regular.ttf",
                "EBGaramond-Regular.ttf",
                "IMFellEnglish-Regular.ttf",
            ],
        }
    }
    cfg = THEMES.get(theme)
    if not cfg:
        warn(f"Unknown theme '{theme}'. Available: {', '.join(THEMES.keys())}")
        return None
    name = find_font_file(cfg["name"]) or None
    lore = find_font_file(cfg["lore"]) or None
    stats = find_font_file(cfg["stats"]) or None
    default = find_font_file(cfg["default"]) or None
    if not any([name, lore, stats, default]):
        warn("No themed fonts found in ./fonts. Please place TTFs there (see README notes). Using defaults.")
        return None
    if name:
        info(f"Theme '{theme}': using name font {os.path.basename(name)}")
    if lore:
        info(f"Theme '{theme}': using lore font {os.path.basename(lore)}")
    if stats:
        info(f"Theme '{theme}': using stats font {os.path.basename(stats)}")
    if default:
        info(f"Theme '{theme}': using default font {os.path.basename(default)}")
    return FontsConfig(default=default, name=name, lore=lore, stats=stats)


# --------------------------- Data Model --------------------------------------

@dataclass
class Treasure:
    id: Optional[str]
    title: Optional[str]
    rarity: Optional[str]
    name: str
    effect: str
    lore: str
    image: Optional[str]
    background: Optional[str]
    fonts_overrides: Dict[str, Optional[str]] = field(default_factory=dict)


def normalize_rarity(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = str(value).strip().lower()
    if v in {"common", "c", "normal"}:
        return "Common"
    if v in {"rare", "r"}:
        return "Rare"
    if v in {"legendary", "legend", "l"}:
        return "Legendary"
    if v in {"curse", "cursed", "trap", "negative"}:
        return "Curse"
    return str(value).strip()


def apply_rarity_border(img_rgb: Image.Image, rarity: Optional[str]) -> Image.Image:
    r = normalize_rarity(rarity)
    if not r:
        return img_rgb

    # Border colors are only an overlay hint; feel free to tweak.
    colors = {
        "Common": (60, 120, 200),
        "Rare": (200, 60, 60),
        "Legendary": (140, 80, 200),
        "Curse": (20, 20, 20),
    }
    color = colors.get(r)
    if not color:
        return img_rgb

    base = img_rgb.convert("RGBA")
    w, h = base.size
    thickness = max(8, int(min(w, h) * 0.026))
    inset = max(10, int(min(w, h) * 0.02))
    radius = max(14, int(min(w, h) * 0.03))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # Subtle outer glow (two soft strokes).
    glow = tuple(int(c * 0.85) for c in color)
    d.rounded_rectangle(
        (2, 2, w - 3, h - 3),
        radius=radius,
        outline=glow + (90,),
        width=max(2, thickness // 4),
    )
    d.rounded_rectangle(
        (4, 4, w - 5, h - 5),
        radius=max(6, radius - 4),
        outline=glow + (70,),
        width=max(2, thickness // 5),
    )

    # Main ornamental frame (rounded).
    d.rounded_rectangle(
        (0, 0, w - 1, h - 1),
        radius=radius,
        outline=color + (255,),
        width=thickness,
    )

    # Inner accent line.
    inner_box = (inset, inset, w - 1 - inset, h - 1 - inset)
    inner_radius = max(6, radius - inset // 2)
    accent = tuple(int(c + (255 - c) * 0.45) for c in color)
    d.rounded_rectangle(
        inner_box,
        radius=inner_radius,
        outline=accent + (210,),
        width=max(2, thickness // 6),
    )

    # Corner flourishes: small diamonds + short ticks.
    fx = inset + max(8, thickness)
    fy = inset + max(8, thickness)
    size = max(10, int(min(w, h) * 0.02))
    tick = max(10, int(min(w, h) * 0.035))
    tick_w = max(2, thickness // 7)
    flourish = accent + (220,)

    corners = [
        (fx, fy),
        (w - 1 - fx, fy),
        (fx, h - 1 - fy),
        (w - 1 - fx, h - 1 - fy),
    ]
    for cx, cy in corners:
        diamond = [(cx, cy - size), (cx + size, cy), (cx, cy + size), (cx - size, cy)]
        d.polygon(diamond, outline=flourish)

    # Ticks from corners along edges.
    d.line((inner_box[0], inner_box[1] + tick, inner_box[0], inner_box[1]), fill=flourish, width=tick_w)
    d.line((inner_box[0] + tick, inner_box[1], inner_box[0], inner_box[1]), fill=flourish, width=tick_w)

    d.line((inner_box[2], inner_box[1] + tick, inner_box[2], inner_box[1]), fill=flourish, width=tick_w)
    d.line((inner_box[2] - tick, inner_box[1], inner_box[2], inner_box[1]), fill=flourish, width=tick_w)

    d.line((inner_box[0], inner_box[3] - tick, inner_box[0], inner_box[3]), fill=flourish, width=tick_w)
    d.line((inner_box[0] + tick, inner_box[3], inner_box[0], inner_box[3]), fill=flourish, width=tick_w)

    d.line((inner_box[2], inner_box[3] - tick, inner_box[2], inner_box[3]), fill=flourish, width=tick_w)
    d.line((inner_box[2] - tick, inner_box[3], inner_box[2], inner_box[3]), fill=flourish, width=tick_w)

    out = Image.alpha_composite(base, overlay)
    return out.convert("RGB")


@dataclass
class CardConfig:
    width: int
    height: int
    dpi: int
    fonts: FontsConfig


# --------------------------- Rendering ---------------------------------------

def measure_wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, line_spacing: int = 4) -> Tuple[int, int, List[str]]:
    def text_size(s: str) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), s, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h

    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        tentative = " ".join(current + [word]) if current else word
        w, _ = text_size(tentative)
        if w <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    line_heights = [text_size(line)[1] for line in lines] or [text_size(" ")[1]]
    height = sum(line_heights) + line_spacing * (len(lines) - 1 if lines else 0)
    width = min(max(text_size(line)[0] for line in lines) if lines else 0, max_width)
    return width, height, lines


def best_fit_wrapped(draw: ImageDraw.ImageDraw, text: str, font_path: Optional[str], max_width: int, max_height: int, min_size: int, max_size: int) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    lo, hi = min_size, max_size
    best_font = fit_font(font_path, min_size)
    best_lines: List[str] = []
    while lo <= hi:
        mid = (lo + hi) // 2
        font = fit_font(font_path, mid)
        _, h, lines = measure_wrapped_text(draw, text, font, max_width)
        if h <= max_height:
            best_font, best_lines = font, lines
            lo = mid + 1
        else:
            hi = mid - 1
    if not best_lines:
        best_lines = [text]
    return best_font, best_lines


def best_fit_wrapped_fill(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Optional[str],
    max_width: int,
    max_height: int,
    min_size: int,
    max_size: int,
    fill_target: float = 0.9,
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    """Like best_fit_wrapped, but tries to *use* available height.

    It picks the largest font that fits; if it still leaves too much empty
    space (text too short), it keeps increasing size until reaching max_size
    or approaching `fill_target` of the height.
    """
    if max_width <= 0 or max_height <= 0:
        return fit_font(font_path, min_size), [text]

    font, lines = best_fit_wrapped(
        draw,
        text,
        font_path,
        max_width=max_width,
        max_height=max_height,
        min_size=min_size,
        max_size=max_size,
    )
    _, h, _ = measure_wrapped_text(draw, text, font, max_width)
    target_h = int(max_height * fill_target)
    if h >= target_h:
        return font, lines

    # Nudge font size upward to better fill the available space.
    current_size = getattr(font, "size", min_size)
    best_font, best_lines = font, lines
    for size in range(int(current_size) + 1, int(max_size) + 1):
        candidate = fit_font(font_path, size)
        _, cand_h, cand_lines = measure_wrapped_text(draw, text, candidate, max_width)
        if cand_h > max_height:
            break
        best_font, best_lines = candidate, cand_lines
        if cand_h >= target_h:
            break
    return best_font, best_lines


def render_treasure_card(t: Treasure, cfg: CardConfig) -> Image.Image:
    W, H = cfg.width, cfg.height
    LW, LH = H, W

    bg_color = parse_color(t.background if t.background and t.background.startswith("#") else None, fallback=(245, 238, 220))
    bg_path = None
    if t.background and not t.background.startswith("#"):
        bg_path = t.background
    else:
        candidate = os.path.join(os.path.dirname(__file__), "parchment.jpg")
        if os.path.isfile(candidate):
            bg_path = candidate

    base = Image.new("RGBA", (LW, LH), bg_color + (255,))
    if bg_path and os.path.isfile(bg_path):
        try:
            bg_img = Image.open(bg_path).convert("RGBA")
            bg_img = ImageOps.fit(bg_img, (LW, LH), method=Image.LANCZOS)
            base.paste(bg_img, (0, 0))
        except Exception as e:
            warn(f"Failed to load card background {bg_path}: {e}")

    draw = ImageDraw.Draw(base)

    pad = int(min(LW, LH) * 0.06)
    x, y = pad, pad
    content_w = LW - 2 * pad
    content_h = LH - 2 * pad

    # Image column ~33% of content width, with a slightly larger gutter between
    # image and text so they don't feel cramped next to each other.
    img_col_w = int(content_w * 0.33)
    text_col_x = x + img_col_w + int(pad * 0.8)
    text_col_w = LW - text_col_x - pad

    fc = cfg.fonts
    name_font_path = t.fonts_overrides.get("name") or fc.name or fc.default
    effect_font_path = t.fonts_overrides.get("effect") or fc.stats or fc.default
    lore_font_path = t.fonts_overrides.get("lore") or fc.lore or fc.default

    # Slightly inset image box so it doesn't visually collide with text column.
    img_inset = int(pad * 0.05)
    img_box = (x + img_inset, y + img_inset, x + img_col_w - img_inset, y + content_h - img_inset)
    ImageDraw.Draw(base).rectangle(img_box, fill=(230, 220, 200, 255))
    profile = safe_load_image(t.image, (img_col_w - 2 * img_inset, content_h - 2 * img_inset))
    try:
        shadow = profile.copy().convert("RGBA")
        shadow = ImageEnhance.Brightness(shadow).enhance(0.35)
    except Exception:
        shadow = None
    # Also pull the shadow in a touch so the visual footprint is slightly smaller.
    shadow_offset = max(10, int(LW * 0.025))
    if shadow is not None:
        shadow_box = (
            x + shadow_offset + img_inset,
            y + img_inset,
            x + shadow_offset + img_col_w - img_inset,
            y + content_h - img_inset,
        )
        draw_rounded_image(base, shadow, shadow_box, radius=24)
    draw_rounded_image(base, profile, img_box, radius=24)

    ty = y

    # Reserve a dedicated header band so title/subtitle always have breathing room.
    header_h = int(content_h * 0.24)
    header_y0 = ty
    header_y1 = ty + header_h

    title_text = (t.title or "").strip()
    subtitle_text = (t.name or "Unknown Treasure").strip()

    # Title: allow wrapping to 2 lines but keep it large.
    if title_text:
        title_max_h = int(header_h * 0.38)
        title_font, title_lines = best_fit_wrapped_fill(
            draw,
            title_text,
            name_font_path,
            max_width=text_col_w,
            max_height=title_max_h,
            min_size=max(12, int(LH * 0.04)),
            max_size=max(14, int(LH * 0.06)),
            fill_target=0.85,
        )
        # Hard cap to 2 lines for the header title.
        title_lines = title_lines[:2]
        _, title_h, _ = measure_wrapped_text(draw, "\n".join(title_lines), title_font, text_col_w)
        line_h = draw.textbbox((0, 0), "Ag", font=title_font)[3] - draw.textbbox((0, 0), "Ag", font=title_font)[1]
        for line in title_lines:
            draw.text((text_col_x, ty), line, font=title_font, fill=(40, 30, 20, 255))
            ty += line_h + 2
        ty += int(pad * 0.10)

    # Subtitle (name): allow wrapping to 2 lines and keep bold/large.
    subtitle_max_h = header_y1 - ty
    subtitle_font, subtitle_lines = best_fit_wrapped_fill(
        draw,
        subtitle_text,
        name_font_path,
        max_width=text_col_w,
        max_height=max(0, subtitle_max_h),
        min_size=max(16, int(LH * 0.065)),
        max_size=max(18, int(LH * 0.095)),
        fill_target=0.9,
    )
    subtitle_lines = subtitle_lines[:2]
    subtitle_line_h = draw.textbbox((0, 0), "Ag", font=subtitle_font)[3] - draw.textbbox((0, 0), "Ag", font=subtitle_font)[1]
    for line in subtitle_lines:
        draw.text((text_col_x, ty), line, font=subtitle_font, fill=(30, 20, 10, 255))
        draw.text((text_col_x + 1, ty), line, font=subtitle_font, fill=(30, 20, 10, 255))
        ty += subtitle_line_h + 2

    # Move to the end of the header band regardless of how much text was used.
    ty = header_y1 + int(pad * 0.10)

    remaining_h = content_h - (ty - y)
    effect_pct = 0.58
    lore_pct = 0.42
    inner_margin = max(6, int(LH * 0.014))
    total_block_margins = inner_margin
    usable_h = max(0, remaining_h - total_block_margins)
    effect_h = int(usable_h * effect_pct)
    lore_h = max(0, usable_h - effect_h)

    effect_font, effect_lines = best_fit_wrapped_fill(
        draw,
        t.effect or "",
        effect_font_path,
        max_width=text_col_w,
        max_height=int(effect_h),
        min_size=10,
        max_size=max(14, int(effect_h * 0.22)),
        fill_target=0.9,
    )
    effect_line_h = draw.textbbox((0, 0), "Ag", font=effect_font)[3] - draw.textbbox((0, 0), "Ag", font=effect_font)[1]
    used_h = 0
    for line in effect_lines:
        if used_h + effect_line_h > effect_h:
            break
        draw.text((text_col_x, ty), line, font=effect_font, fill=(35, 25, 15, 255))
        ty += effect_line_h + 2
        used_h += effect_line_h + 2

    ty = y + (content_h - lore_h)

    lore_font, lore_lines = best_fit_wrapped_fill(
        draw,
        t.lore or "",
        lore_font_path,
        max_width=text_col_w,
        max_height=int(lore_h),
        min_size=10,
        max_size=max(14, int(lore_h * 0.22)),
        fill_target=0.9,
    )
    italic_fill = (50, 40, 30, 255)
    lore_line_h = draw.textbbox((0, 0), "Ag", font=lore_font)[3] - draw.textbbox((0, 0), "Ag", font=lore_font)[1]
    drawn = 0
    for line in lore_lines:
        if drawn + lore_line_h > lore_h:
            break
        draw.text((text_col_x, ty), line, font=lore_font, fill=italic_fill)
        ty += lore_line_h + 2
        drawn += lore_line_h + 2

    final_img = base.convert("RGB").rotate(-90, expand=True)
    final_img = ImageOps.contain(final_img, (W, H), method=Image.LANCZOS)
    return final_img


# --------------------------- PDF Layout --------------------------------------

def compose_pdf(cards: List[Image.Image], outfile: str, dpi: int, margin: int, gutter: int, cards_per_page: int = DEFAULT_CARDS_PER_PAGE, page_background_path: Optional[str] = None, full_bleed: bool = False, cut_lines: bool = False, cut_line_width_pt: float = 0.5) -> None:
    if not cards:
        warn("No cards to compose into PDF.")
        return
    page_w_px = A4_WIDTH_PX_300DPI
    page_h_px = A4_HEIGHT_PX_300DPI
    scale = dpi / 300.0
    target_page_w = int(page_w_px * scale)
    target_page_h = int(page_h_px * scale)

    cols = DEFAULT_GRID_COLS
    rows = DEFAULT_GRID_ROWS
    per_page = min(cards_per_page, cols * rows)

    if full_bleed:
        margin = 0
        gutter = 0
    usable_w = target_page_w - 2 * margin - (cols - 1) * gutter
    usable_h = target_page_h - 2 * margin - (rows - 1) * gutter
    card_w = usable_w // cols
    card_h = usable_h // rows

    c = pdf_canvas.Canvas(outfile, pagesize=A4)
    px_to_pt = 72.0 / dpi

    for i in range(0, len(cards), per_page):
        batch = cards[i : i + per_page]
        if page_background_path and os.path.isfile(page_background_path):
            try:
                bg_img = Image.open(page_background_path).convert("RGBA")
                bg_img = ImageOps.fit(bg_img, (target_page_w, target_page_h), method=Image.LANCZOS)
                buf_bg = io.BytesIO()
                bg_img.save(buf_bg, format="PNG")
                img_bg_data = buf_bg.getvalue()
                bg_reader = ImageReader(io.BytesIO(img_bg_data))
                c.drawImage(bg_reader, 0, 0, width=target_page_w * px_to_pt, height=target_page_h * px_to_pt)
            except Exception as e:
                warn(f"Failed to draw PDF background '{page_background_path}': {e}")
        for idx, card in enumerate(batch):
            r = idx // cols
            col = idx % cols
            x_px = margin + col * (card_w + gutter)
            y_px = margin + r * (card_h + gutter)
            card_resized = ImageOps.fit(card, (card_w, card_h), method=Image.LANCZOS)
            buf = io.BytesIO()
            card_resized.save(buf, format="PNG")
            img_data = buf.getvalue()
            x_pt = x_px * px_to_pt
            y_pt = (target_page_h - y_px - card_h) * px_to_pt
            w_pt = card_w * px_to_pt
            h_pt = card_h * px_to_pt
            img_reader = ImageReader(io.BytesIO(img_data))
            c.drawImage(img_reader, x_pt, y_pt, width=w_pt, height=h_pt, mask="auto")
        if cut_lines:
            c.setLineWidth(cut_line_width_pt)
            for col_i in range(1, cols):
                x_px = margin + col_i * (card_w + gutter) - (0 if gutter == 0 else gutter // 2)
                x_pt = x_px * px_to_pt
                c.line(x_pt, 0, x_pt, target_page_h * px_to_pt)
            for row_i in range(1, rows):
                y_px = margin + row_i * (card_h + gutter) - (0 if gutter == 0 else gutter // 2)
                y_pt = (target_page_h - y_px) * px_to_pt
                c.line(0, y_pt, target_page_w * px_to_pt, y_pt)
        c.showPage()

        try:
            # Prefer a blue leather backing specific to treasure cards if present,
            # otherwise fall back to the generic leather.jpg, then parchment.
            blueleather_path = os.path.join(os.path.dirname(__file__), "blueleather.png")
            leather_path = os.path.join(os.path.dirname(__file__), "leather.jpg")
            parchment_path = os.path.join(os.path.dirname(__file__), "parchment.jpg")
            logo_path = os.path.join(os.path.dirname(__file__), "dnd_logo.png")
            cell = Image.new("RGBA", (card_w, card_h), (245, 238, 220, 255))
            bg_used = None
            if os.path.isfile(blueleather_path):
                bg_used = blueleather_path
            elif os.path.isfile(leather_path):
                bg_used = leather_path
            elif os.path.isfile(parchment_path):
                bg_used = parchment_path
            if bg_used:
                try:
                    pimg = Image.open(bg_used).convert("RGBA")
                    pimg = ImageOps.fit(pimg, (card_w, card_h), method=Image.LANCZOS)
                    cell.paste(pimg, (0, 0))
                except Exception as e:
                    warn(f"Failed to load backside background '{bg_used}': {e}")
            if os.path.isfile(logo_path):
                try:
                    limg = Image.open(logo_path).convert("RGBA")
                    limg = limg.rotate(-90, expand=True)
                    max_logo_w = int(card_w * 0.6)
                    max_logo_h = int(card_h * 0.6)
                    limg = ImageOps.contain(limg, (max_logo_w, max_logo_h), method=Image.LANCZOS)
                    lw, lh = limg.size
                    lx = (card_w - lw) // 2
                    ly = (card_h - lh) // 2
                    cell.paste(limg, (lx, ly), limg)
                except Exception as e:
                    warn(f"Failed to load logo for backside: {e}")
            buf_cell = io.BytesIO()
            cell_rgb = cell.convert("RGB")
            cell_rgb.save(buf_cell, format="PNG")
            cell_reader = ImageReader(io.BytesIO(buf_cell.getvalue()))

            for slot_idx in range(per_page):
                r = slot_idx // cols
                col = slot_idx % cols
                x_px = margin + col * (card_w + gutter)
                y_px = margin + r * (card_h + gutter)
                x_pt = x_px * px_to_pt
                y_pt = (target_page_h - y_px - card_h) * px_to_pt
                w_pt = card_w * px_to_pt
                h_pt = card_h * px_to_pt
                c.drawImage(cell_reader, x_pt, y_pt, width=w_pt, height=h_pt, mask="auto")
            if cut_lines:
                c.setLineWidth(cut_line_width_pt)
                for col_i in range(1, cols):
                    x_px = margin + col_i * (card_w + gutter) - (0 if gutter == 0 else gutter // 2)
                    x_pt = x_px * px_to_pt
                    c.line(x_pt, 0, x_pt, target_page_h * px_to_pt)
                for row_i in range(1, rows):
                    y_px = margin + row_i * (card_h + gutter) - (0 if gutter == 0 else gutter // 2)
                    y_pt = (target_page_h - y_px) * px_to_pt
                    c.line(0, y_pt, target_page_w * px_to_pt, y_pt)
            c.showPage()
        except Exception as e:
            warn(f"Failed to generate backside page: {e}")
    c.save()
    info(f"Wrote PDF: {outfile}")


# --------------------------- IO helpers --------------------------------------

def read_treasures(path: Optional[str], demo: bool = False, images_dir: str = "./images") -> List[Treasure]:
    if demo:
        demo_path = os.path.join(os.path.dirname(__file__), "example_treasures.json")
        if not os.path.isfile(demo_path):
            warn("example_treasures.json not found; demo cannot proceed.")
            return []
        data = load_json(demo_path)
    else:
        if not path:
            warn("No input JSON provided.")
            return []
        if not os.path.isfile(path):
            warn(f"Input JSON not found: {path}")
            return []
        data = load_json(path)

    treasures: List[Treasure] = []
    for i, item in enumerate(data):
        name = item.get("name")
        if not name:
            warn(f"Treasure index {i} missing name; skipping.")
            continue
        image = item.get("image")
        if image and not os.path.isabs(image):
            candidate = os.path.join(images_dir, image)
            if os.path.isfile(candidate):
                image = candidate
        treasures.append(
            Treasure(
                id=item.get("id"),
                title=item.get("title"),
                rarity=item.get("rarity"),
                name=name,
                effect=item.get("effect") or "",
                lore=item.get("lore") or "",
                image=image,
                background=item.get("background"),
                fonts_overrides=item.get("fonts") or {},
            )
        )
    return treasures


def save_pngs(cards: List[Image.Image], treasures: List[Treasure], outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []
    for idx, (img, t) in enumerate(zip(cards, treasures)):
        slug = slugify(t.name or f"treasure_{idx}")
        fname = f"{idx:02d}_{slug}.png"
        fpath = os.path.join(outdir, fname)
        img.save(fpath, format="PNG")
        info(f"Saved {fpath}")
        paths.append(fpath)
    return paths


# --------------------------- CLI ---------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DnD Treasure Card Generator")
    parser.add_argument("input", nargs="?", help="Input JSON file containing treasures")
    parser.add_argument("--outdir", default="./out_cards", help="Directory to save PNGs")
    parser.add_argument("--images-dir", default="./images", help="Default images directory for treasure art")
    parser.add_argument("--theme", default="fantasy", help="Apply a curated font theme (e.g., 'fantasy'); looks for TTFs in ./fonts")
    parser.add_argument("--pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--pdf-bg", default=None, help="Optional PDF page background image path (A4 portrait)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--card-width", type=int, default=None, help="Card width (px)")
    parser.add_argument("--card-height", type=int, default=None, help="Card height (px)")
    parser.add_argument("--font-default", default=None, help="Default font path or name")
    parser.add_argument("--font-name", default=None, help="Name/title font path or name")
    parser.add_argument("--font-lore", default=None, help="Lore font path or name")
    parser.add_argument("--font-stats", default=None, help="Stats/effect font path or name")
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="PDF page margin (px)")
    parser.add_argument("--gutter", type=int, default=DEFAULT_GUTTER, help="PDF gutter (px)")
    parser.add_argument("--dpi-cards-per-page", type=int, default=DEFAULT_CARDS_PER_PAGE, help="Cards per page for PDF (default 9)")
    parser.add_argument("--full-bleed", action="store_true", default=True, help="Fill entire PDF page with the 3x3 grid (no margins/gutters)")
    parser.add_argument("--no-full-bleed", action="store_false", dest="full_bleed", help="Disable full-bleed PDF layout (restore margins/gutters)")
    parser.add_argument("--cut-lines", action="store_true", help="Draw cut guide lines between cards on the PDF page")
    parser.add_argument("--cut-line-width", type=float, default=0.5, help="Cut line width in points (PDF units)")
    parser.add_argument("--demo", action="store_true", help="Generate sample cards from example_treasures.json")
    args = parser.parse_args(argv)

    default_card_w = A4_WIDTH_PX_300DPI // DEFAULT_GRID_COLS
    default_card_h = A4_HEIGHT_PX_300DPI // DEFAULT_GRID_ROWS
    card_w = args.card_width or default_card_w
    card_h = args.card_height or default_card_h

    treasures = read_treasures(args.input, demo=args.demo, images_dir=args.images_dir)
    if not treasures:
        warn("No treasures to render.")
        return 1

    theme_cfg = resolve_theme_fonts(args.theme)
    fonts_cfg = FontsConfig(
        default=args.font_default or (theme_cfg.default if theme_cfg else None),
        name=args.font_name or (theme_cfg.name if theme_cfg else None),
        lore=args.font_lore or (theme_cfg.lore if theme_cfg else None),
        stats=args.font_stats or (theme_cfg.stats if theme_cfg else None),
    )
    cfg = CardConfig(width=card_w, height=card_h, dpi=args.dpi, fonts=fonts_cfg)

    cards: List[Image.Image] = []
    for i, t in enumerate(treasures):
        info(f"Rendering treasure card {i+1}/{len(treasures)}: {t.name}")
        img = render_treasure_card(t, cfg)
        img = apply_rarity_border(img, t.rarity)
        cards.append(img)

    save_pngs(cards, treasures, args.outdir)

    if args.pdf:
        info("Composing PDF...")
        compose_pdf(
            cards,
            args.pdf,
            dpi=args.dpi,
            margin=args.margin,
            gutter=args.gutter,
            cards_per_page=args.dpi_cards_per_page,
            page_background_path=args.pdf_bg,
            full_bleed=args.full_bleed,
            cut_lines=args.cut_lines,
            cut_line_width_pt=args.cut_line_width,
        )

    info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
