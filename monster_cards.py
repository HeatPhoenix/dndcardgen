#!/usr/bin/env python3
"""
DnD Monster Flashcard Generator

Generates PNG flashcards for DnD monsters from a JSON input, and optionally
arranges them into an A4 multi-page PDF (3x3 grid). Supports per-section fonts,
per-card backgrounds (color or image), rounded profile image, section headers,
italic lore, and dynamic text wrapping/size reduction.

Usage:
    python monster_cards.py input.json --outdir cards --pdf OUTFILE.pdf --dpi 300

CLI flags:
    --outdir            Directory to save PNGs (default: ./out_cards)
    --pdf               Optional output PDF path; if provided, arrange cards
                        into A4 pages with N cards per page (default 9)
    --dpi               DPI for output images (default: 300)
    --card-width        Card width in pixels (default: A4 width / 3)
    --card-height       Card height in pixels (default: A4 height / 3)
    --font-default      Default font path or system font name
    --font-name         Font for title (name)
    --font-lore         Font for lore text (italic if available)
    --font-stats        Font for stats and section headers
    --theme             Apply a curated font theme (e.g., 'fantasy') by looking for TTFs in ./fonts
    --margin            Page margin (pixels) for PDF (default: 60 at 300dpi)
    --gutter            Gutter between cards for PDF (default: 30 at 300dpi)
    --demo              Generate sample cards from bundled example_monsters.json

Input JSON schema:
    See the user request; object includes fields like name, profile_image,
    lore, culinary_use, statblock, abilities, actions, background, fonts.

Dependencies:
    - Pillow
    - reportlab

"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageEnhance, ImageChops
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# Constants
A4_WIDTH_PX_300DPI = 2480
A4_HEIGHT_PX_300DPI = 3508
DEFAULT_GRID_COLS = 3
DEFAULT_GRID_ROWS = 3
DEFAULT_CARDS_PER_PAGE = DEFAULT_GRID_COLS * DEFAULT_GRID_ROWS
DEFAULT_MARGIN = 90  # pixels at 300dpi (increased)
DEFAULT_GUTTER = 45  # pixels at 300dpi (increased)
FONT_SCALE = 3.0  # Global font scale multiplier per user request

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


def resolve_profile_image(name: str, declared_path: Optional[str], images_dir: str) -> Optional[str]:
    """Resolve a monster's profile image path using declared path or by guessing in images_dir.
    Tries provided path, then images_dir/provided path, then images_dir/<name>.<ext> with common image extensions.
    Returns a filesystem path if found, else None.
    """
    common_exts = ["png", "jpg", "jpeg", "webp"]
    # 1) If declared path exists as-is
    if declared_path and os.path.isfile(declared_path):
        return declared_path
    # 2) If declared path is relative, try under images_dir
    if declared_path:
        candidate = os.path.join(images_dir, declared_path)
        if os.path.isfile(candidate):
            return candidate
    # 3) Try guessed filenames based on name variants under images_dir
    def variants(n: str) -> List[str]:
        s = slugify(n)
        return list(dict.fromkeys([
            n,
            n.strip(),
            n.replace(" ", "_"),
            n.replace(" ", "-") ,
            n.lower(),
            s,
            s.lower(),
        ]))
    bases = variants(name)
    for base in bases:
        for ext in common_exts:
            candidate = os.path.join(images_dir, f"{base}.{ext}")
            if os.path.isfile(candidate):
                return candidate
    return None


def safe_load_image(path_or_url: Optional[str], target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Load an image from a local path (URL not implemented for offline safety). If missing, returns a placeholder silhouette image.
    """
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
    # Simple silhouette: circle + rectangle
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
    # Not a color; possibly a path to background image handled elsewhere
    return fallback


def draw_rounded_image(base: Image.Image, img: Image.Image, box: Tuple[int, int, int, int], radius: int = 20) -> None:
    """Paste an image into the given box with rounded corners, scaling to fit fully (no crop), centered, preserving transparency in letterbox areas."""
    x0, y0, x1, y1 = map(int, box)
    w, h = x1 - x0, y1 - y0
    # Scale to fit inside the box without cropping
    img_contained = ImageOps.contain(img, (w, h), method=Image.LANCZOS)
    iw, ih = img_contained.size
    ox = (w - iw) // 2
    oy = (h - ih) // 2
    # Prepare a transparent canvas and paste the centered image using its alpha
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(img_contained, (ox, oy), img_contained)
    # Build a rounded rectangle alpha mask and combine with current alpha to avoid copying black letterbox
    round_mask = Image.new("L", (w, h), 0)
    mask_draw = ImageDraw.Draw(round_mask)
    mask_draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    current_alpha = canvas.split()[3]
    combined_alpha = ImageChops.multiply(current_alpha, round_mask)
    canvas.putalpha(combined_alpha)
    # Paste onto base using the canvas's own alpha
    base.paste(canvas, (x0, y0), canvas)


def fit_font(fontpath_or_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Try to load the font; fall back to a default PIL font if necessary."""
    if fontpath_or_name:
        # Try as path first
        if os.path.isfile(fontpath_or_name):
            try:
                return ImageFont.truetype(fontpath_or_name, size=size)
            except Exception as e:
                warn(f"Failed to load font from path {fontpath_or_name}: {e}")
        # Try as name via PIL (requires the font file path)
        # Pillow does not resolve system font names universally; we keep it simple.
    # Fallback to DejaVuSans if available, else PIL default
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def find_font_file(candidates: List[str]) -> Optional[str]:
    """Return the first existing font file path from candidates.
    Searches ./fonts relative to this script first, then current working dir.
    """
    search_roots = [os.path.join(os.path.dirname(__file__), "fonts"), os.getcwd()]
    for root in search_roots:
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None


def resolve_theme_fonts(theme: Optional[str]) -> Optional["FontsConfig"]:
    if not theme:
        return None
    theme = theme.lower().strip()
    # Curated open/free fonts (download from Google Fonts) for a fantasy vibe
    # Place TTFs into ./fonts to be auto-detected
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


def measure_wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, line_spacing: int = 4) -> Tuple[int, int, List[str]]:
    """Wraps text into lines that fit max_width; returns (w, h, lines)."""
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
    # Measure
    line_heights = [text_size(line)[1] for line in lines] or [text_size(" ")[1]]
    height = sum(line_heights) + line_spacing * (len(lines) - 1 if lines else 0)
    width = min(max(text_size(line)[0] for line in lines) if lines else 0, max_width)
    return width, height, lines


def shrink_to_fit(draw: ImageDraw.ImageDraw, text: str, font_path: Optional[str], start_size: int, max_width: int, max_height: int, min_size: int = 10) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    size = start_size
    while size >= min_size:
        font = fit_font(font_path, size)
        _, h, lines = measure_wrapped_text(draw, text, font, max_width)
        if h <= max_height:
            return font, lines
        size -= 1
    return fit_font(font_path, min_size), textwrap.wrap(text, width=40)

# --------------------------- Card Renderer -----------------------------------

@dataclass
class FontsConfig:
    default: Optional[str] = None
    name: Optional[str] = None
    lore: Optional[str] = None
    stats: Optional[str] = None


@dataclass
class CardConfig:
    width: int
    height: int
    dpi: int
    fonts: FontsConfig


@dataclass
class Monster:
    id: Optional[str]
    name: str
    profile_image: Optional[str]
    lore: str
    culinary_use: Optional[str]
    statblock: Dict[str, str]
    abilities: List[Dict[str, str]]
    actions: List[Dict[str, str]]
    background: Optional[str]
    fonts_overrides: Dict[str, Optional[str]] = field(default_factory=dict)


def render_card(mon: Monster, cfg: CardConfig) -> Image.Image:
    # We render in landscape, then rotate 90° so the final card fits portrait slots.
    W, H = cfg.width, cfg.height  # portrait dimensions of slot
    LW, LH = H, W  # landscape canvas dimensions

    # Background: color or image (default parchment.jpg if available)
    bg_img: Optional[Image.Image] = None
    bg_color = parse_color(mon.background if mon.background and mon.background.startswith("#") else None, fallback=(245, 238, 220))
    bg_path = None
    if mon.background and not mon.background.startswith("#"):
        bg_path = mon.background
    else:
        # default to parchment.jpg in CWD/script dir if exists
        candidate = os.path.join(os.path.dirname(__file__), "parchment.jpg")
        if os.path.isfile(candidate):
            bg_path = candidate

    base = Image.new("RGBA", (LW, LH), bg_color + (255,))
    if bg_path and os.path.isfile(bg_path):
        try:
            bg_img = Image.open(bg_path).convert("RGBA")
            bg_img = ImageOps.fit(bg_img, (LW, LH), method=Image.LANCZOS)
            base.paste(bg_img, (0, 0))
            # slight texture overlay retained; optionally add semi-transparent panel for text areas later
        except Exception as e:
            warn(f"Failed to load card background {bg_path}: {e}")

    draw = ImageDraw.Draw(base)

    # Padding/layout on landscape
    pad = int(min(LW, LH) * 0.06)
    x, y = pad, pad
    content_w = LW - 2 * pad
    content_h = LH - 2 * pad

    # Columns: image left (about 40%), text right (60%)
    img_col_w = int(content_w * 0.40)
    text_col_x = x + img_col_w + int(pad * 0.5)
    text_col_w = LW - text_col_x - pad

    # Fonts
    fc = cfg.fonts
    name_font_path = mon.fonts_overrides.get("name") or fc.name or fc.default
    lore_font_path = mon.fonts_overrides.get("lore") or fc.lore or fc.default
    stats_font_path = mon.fonts_overrides.get("stats") or fc.stats or fc.default

    # Profile image fills left column, rounded, with drop shadow to the right
    img_box = (x, y, x + img_col_w, y + content_h)
    # Solid panel under image to avoid perceived emptiness around rounded corners
    ImageDraw.Draw(base).rectangle(img_box, fill=(230, 220, 200, 255))
    profile = safe_load_image(mon.profile_image, (img_col_w, content_h))
    # Create a darkened shadow copy and paste it slightly to the right, behind the image
    try:
        shadow = profile.copy().convert("RGBA")
        shadow = ImageEnhance.Brightness(shadow).enhance(0.35)
    except Exception:
        shadow = None
    # Move the drop shadow further to the right per request
    shadow_offset = max(12, int(LW * 0.03))
    if shadow is not None:
        shadow_box = (x + shadow_offset, y, x + shadow_offset + img_col_w, y + content_h)
        draw_rounded_image(base, shadow, shadow_box, radius=24)
    # Now the main rounded image
    draw_rounded_image(base, profile, img_box, radius=24)

    # Title at top of text column (much larger per request, with width/height constraints)
    def shrink_single_line_to_fit(text: str, font_path: Optional[str], start_size: int, max_width: int, max_height: int) -> Tuple[ImageFont.FreeTypeFont, int]:
        size = start_size
        while size > 6:
            f = fit_font(font_path, size)
            bbox = draw.textbbox((0, 0), text, font=f)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w <= max_width and h <= max_height:
                return f, h
            size -= 1
        f = fit_font(font_path, max(6, size))
        h = draw.textbbox((0, 0), text, font=f)[3] - draw.textbbox((0, 0), text, font=f)[1]
        return f, h

    base_title_size = max(36, int(LH * 0.12))
    desired_title_size = int(base_title_size * 6 * FONT_SCALE)
    title_text = mon.name or "Unknown"
    max_title_h = int(content_h * 0.25)
    title_font, th = shrink_single_line_to_fit(title_text, name_font_path, desired_title_size, text_col_w, max_title_h)
    ty = y
    # Fake bold for title
    draw.text((text_col_x, ty), title_text, font=title_font, fill=(30, 20, 10, 255))
    draw.text((text_col_x+1, ty), title_text, font=title_font, fill=(30, 20, 10, 255))
    ty += th + int(pad * 0.25)

    # Section header style (text column width) — headers larger than body
    header_font_size = int(max(26, int(LH * 0.08)) * 3 * FONT_SCALE)
    header_font = fit_font(stats_font_path, header_font_size)
    header_bar_h = max(12, int(LH * 0.02))
    header_gap = int(pad * 0.15)
    # Estimate header text height for allocation using a representative string
    _hb = draw.textbbox((0, 0), "Ag", font=header_font)
    header_text_h_est = _hb[3] - _hb[1]
    header_cost_est = header_bar_h + header_gap + header_text_h_est

    def section_header(text: str, yy: int) -> int:
        # Actual header text height
        tb = draw.textbbox((0, 0), text, font=header_font)
        header_text_h = tb[3] - tb[1]
        draw.rounded_rectangle((text_col_x, yy, text_col_x + text_col_w, yy + header_bar_h), radius=6, fill=(120, 90, 60, 160))
        text_y = yy + max(0, (header_bar_h - header_text_h) // 2)
        # Fake bold by double draw
        draw.text((text_col_x + 6, text_y), text, font=header_font, fill=(40, 30, 20, 255))
        draw.text((text_col_x + 7, text_y), text, font=header_font, fill=(40, 30, 20, 255))
        return yy + header_bar_h + header_gap + header_text_h

    # Fixed block heights for sections to avoid overlap and keep consistency
    # Allocate proportions of text column height remaining after title
    remaining_h = content_h - (ty - y)
    # Section proportions must add to 100%
    # Tuned for balanced layout: Lore 22%, Stats 20%, Abilities 20%, Actions 28%, Culinary 10% = 100%
    lore_pct = 0.22
    stats_pct = 0.20
    abilities_pct = 0.20
    actions_pct = 0.28
    culinary_pct = 0.10
    # Renormalize if culinary is absent (assign its share to Actions)
    if not mon.culinary_use:
        total = lore_pct + stats_pct + abilities_pct + actions_pct
        lore_pct /= total
        stats_pct /= total
        abilities_pct /= total
        actions_pct /= total
        culinary_pct = 0.0
    # Small inner margin between blocks to avoid visual crowding
    inner_margin = max(6, int(LH * 0.012))
    # Compute heights, ensuring total fills remaining space minus margins and header bars
    sections_present = 4 + (1 if mon.culinary_use else 0)
    total_block_margins = inner_margin * 4  # gaps between blocks (Lore|Stats|Abilities|Actions|Culinary)
    total_header_cost = sections_present * header_cost_est
    usable_h = max(0, remaining_h - total_block_margins - total_header_cost)
    lore_h = int(usable_h * lore_pct)
    stats_h = int(usable_h * stats_pct)
    abilities_h = int(usable_h * abilities_pct)
    actions_h = int(usable_h * actions_pct)
    # Assign remainder to culinary to ensure full fill
    assigned = lore_h + stats_h + abilities_h + actions_h
    culinary_h = max(0, usable_h - assigned)
    line_gap = 2

    # Lore block (italic)
    ty = section_header("Lore", ty)
    lore_start = ty
    lore_font, lore_lines = shrink_to_fit(draw, mon.lore or "", lore_font_path, start_size=int(max(22, int(LH * 0.07)) * FONT_SCALE), max_width=text_col_w, max_height=lore_h)
    italic_fill = (50, 40, 30, 255)
    lore_line_h = draw.textbbox((0, 0), "Ag", font=lore_font)[3] - draw.textbbox((0, 0), "Ag", font=lore_font)[1]
    drawn_h = 0
    for line in lore_lines:
        if drawn_h + lore_line_h > lore_h:
            break
        draw.text((text_col_x, ty), line, font=lore_font, fill=italic_fill)
        ty += lore_line_h + line_gap
        drawn_h += lore_line_h + line_gap
    # Advance to end of allocated block to ensure full column fill
    ty = lore_start + lore_h

    # Gap between blocks
    ty += inner_margin
    # Stats block
    ty = section_header("Stats", ty)
    stats_font = fit_font(stats_font_path, int(max(18, int(LH * 0.065)) * FONT_SCALE))
    kv_pad = 6
    col_w = text_col_w // 2
    sx = text_col_x
    sy = ty
    items = list((mon.statblock or {}).items())
    sample_h = draw.textbbox((0, 0), "Ag", font=stats_font)[3] - draw.textbbox((0, 0), "Ag", font=stats_font)[1]
    rows_fit = max(1, (stats_h // (sample_h + kv_pad)))
    max_items = rows_fit * 2
    items = items[:max_items]
    for i, (k, v) in enumerate(items):
        cx = sx if i % 2 == 0 else sx + col_w
        cy = sy + (i // 2) * (sample_h + kv_pad)
        if (i // 2) * (sample_h + kv_pad) + sample_h > stats_h:
            break
        draw.text((cx, cy), f"{k}: {v}", font=stats_font, fill=(35, 25, 15, 255))
    # Advance to end of allocated block to ensure full column fill
    ty = sy + stats_h

    # Gap between blocks
    ty += inner_margin
    # Abilities block
    ty = section_header("Abilities", ty)
    abil_start = ty
    body_font = fit_font(stats_font_path, int(max(18, int(LH * 0.06)) * FONT_SCALE))
    name_color = (30, 20, 10, 255)
    text_color = (50, 40, 30, 255)
    line_h = draw.textbbox((0, 0), "Ag", font=body_font)[3] - draw.textbbox((0, 0), "Ag", font=body_font)[1]
    used_h = 0
    for ab in mon.abilities or []:
        if used_h >= abilities_h:
            break
        name = ab.get("name", "").strip()
        text = ab.get("text", "").strip()
        if name:
            draw.text((text_col_x, ty), name + ": ", font=body_font, fill=name_color)
            draw.text((text_col_x+1, ty), name + ": ", font=body_font, fill=name_color)
        name_w = draw.textbbox((0, 0), name + ": ", font=body_font)
        name_w = (name_w[2] - name_w[0]) if name else 0
        _, _, lines = measure_wrapped_text(draw, text, body_font, text_col_w - name_w)
        for line in lines:
            if used_h + line_h > abilities_h:
                break
            tx_start = text_col_x + name_w if name else text_col_x
            draw.text((tx_start, ty), line, font=body_font, fill=text_color)
            ty += line_h + line_gap
            used_h += line_h + line_gap
        ty += 2
    # Advance to end of allocated block
    ty = abil_start + abilities_h

    # Gap between blocks
    ty += inner_margin
    # Actions block
    ty = section_header("Actions", ty)
    act_start = ty
    used_h = 0
    for ac in mon.actions or []:
        if used_h >= actions_h:
            break
        name = ac.get("name", "").strip()
        text = ac.get("text", "").strip()
        if name:
            draw.text((text_col_x, ty), name + ": ", font=body_font, fill=name_color)
            draw.text((text_col_x+1, ty), name + ": ", font=body_font, fill=name_color)
        name_w = draw.textbbox((0, 0), name + ": ", font=body_font)
        name_w = (name_w[2] - name_w[0]) if name else 0
        _, _, lines = measure_wrapped_text(draw, text, body_font, text_col_w - name_w)
        for line in lines:
            if used_h + line_h > actions_h:
                break
            tx_start = text_col_x + name_w if name else text_col_x
            draw.text((tx_start, ty), line, font=body_font, fill=text_color)
            ty += line_h + line_gap
            used_h += line_h + line_gap
        ty += 2
    # Advance to end of allocated block
    ty = act_start + actions_h

    # Gap between blocks
    ty += inner_margin
    # Culinary Use block
    if mon.culinary_use:
        ty = section_header("Culinary Use", ty)
        # Expand font to fill block height as much as possible
        cul_font, lines = shrink_to_fit(draw, mon.culinary_use, lore_font_path, start_size=int(max(22, int(LH * 0.07)) * FONT_SCALE), max_width=text_col_w, max_height=culinary_h)
        cul_line_h = draw.textbbox((0, 0), "Ag", font=cul_font)[3] - draw.textbbox((0, 0), "Ag", font=cul_font)[1]
        used_h = 0
        for line in lines:
            if used_h + cul_line_h > culinary_h:
                break
            draw.text((text_col_x, ty), line, font=cul_font, fill=text_color)
            ty += cul_line_h + line_gap
            used_h += cul_line_h + line_gap
        # Advance to end of allocated block
        ty = ty - used_h + culinary_h

    # Rotate 90° clockwise to return to portrait slot size
    final_img = base.convert("RGB").rotate(-90, expand=True)
    # Ensure final size matches portrait slot without cropping
    final_img = ImageOps.contain(final_img, (W, H), method=Image.LANCZOS)
    return final_img

# --------------------------- PDF Layout --------------------------------------

def compose_pdf(cards: List[Image.Image], outfile: str, dpi: int, margin: int, gutter: int, cards_per_page: int = DEFAULT_CARDS_PER_PAGE, page_background_path: Optional[str] = None, full_bleed: bool = False, cut_lines: bool = False, cut_line_width_pt: float = 0.5) -> None:
    if not cards:
        warn("No cards to compose into PDF.")
        return
    page_w_px = A4_WIDTH_PX_300DPI
    page_h_px = A4_HEIGHT_PX_300DPI
    # If dpi is not 300, scale page pixels proportionally to maintain physical size
    scale = dpi / 300.0
    target_page_w = int(page_w_px * scale)
    target_page_h = int(page_h_px * scale)

    cols = DEFAULT_GRID_COLS
    rows = DEFAULT_GRID_ROWS
    per_page = min(cards_per_page, cols * rows)

    # Compute card placements
    # Total usable width/height
    if full_bleed:
        margin = 0
        gutter = 0
    usable_w = target_page_w - 2 * margin - (cols - 1) * gutter
    usable_h = target_page_h - 2 * margin - (rows - 1) * gutter
    card_w = usable_w // cols
    card_h = usable_h // rows

    # Prepare reportlab canvas with actual A4 phys size (portrait)
    c = pdf_canvas.Canvas(outfile, pagesize=A4)

    # Convert pixels at given dpi to points for reportlab: points = pixels * 72 / dpi
    px_to_pt = 72.0 / dpi

    for i in range(0, len(cards), per_page):
        batch = cards[i : i + per_page]
        # Optional page background image
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
        # Place cards in grid
        for idx, card in enumerate(batch):
            r = idx // cols
            col = idx % cols
            x_px = margin + col * (card_w + gutter)
            y_px = margin + r * (card_h + gutter)
            # Resize card to fit slot
            card_resized = ImageOps.fit(card, (card_w, card_h), method=Image.LANCZOS)
            # Save to buffer and draw
            buf = io.BytesIO()
            card_resized.save(buf, format="PNG")
            img_data = buf.getvalue()
            # PDF origin is bottom-left, so convert y
            x_pt = x_px * px_to_pt
            y_pt = (target_page_h - y_px - card_h) * px_to_pt
            w_pt = card_w * px_to_pt
            h_pt = card_h * px_to_pt
            img_reader = ImageReader(io.BytesIO(img_data))
            c.drawImage(img_reader, x_pt, y_pt, width=w_pt, height=h_pt, mask='auto')
        # Draw cut lines between cards if requested
        if cut_lines:
            c.setLineWidth(cut_line_width_pt)
            # Vertical lines at column boundaries
            for col_i in range(1, cols):
                x_px = margin + col_i * (card_w + gutter) - (0 if gutter == 0 else gutter // 2)
                x_pt = x_px * px_to_pt
                c.line(x_pt, 0, x_pt, target_page_h * px_to_pt)
            # Horizontal lines at row boundaries
            for row_i in range(1, rows):
                y_px = margin + row_i * (card_h + gutter) - (0 if gutter == 0 else gutter // 2)
                y_pt = (target_page_h - y_px) * px_to_pt
                c.line(0, y_pt, target_page_w * px_to_pt, y_pt)
        c.showPage()

        # Backside page: same grid, per-slot leather background with centered logo
        # Prepare a single cell image to reuse across the grid
        try:
            leather_path = os.path.join(os.path.dirname(__file__), "leather.jpg")
            # Fallbacks if leather is missing
            parchment_path = os.path.join(os.path.dirname(__file__), "parchment.jpg")
            logo_path = os.path.join(os.path.dirname(__file__), "dnd_logo.png")
            cell = Image.new("RGBA", (card_w, card_h), (245, 238, 220, 255))
            # Leather fill (preferred), fallback to parchment
            bg_used = None
            if os.path.isfile(leather_path):
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
            # Centered logo
            if os.path.isfile(logo_path):
                try:
                    limg = Image.open(logo_path).convert("RGBA")
                    # Rotate logo 90° clockwise to match card orientation
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
            # Convert cell to ImageReader once
            buf_cell = io.BytesIO()
            cell_rgb = cell.convert("RGB")
            cell_rgb.save(buf_cell, format="PNG")
            cell_reader = ImageReader(io.BytesIO(buf_cell.getvalue()))

            # Optional: page background on backside not requested; we keep it plain
            for slot_idx in range(per_page):
                r = slot_idx // cols
                col = slot_idx % cols
                x_px = margin + col * (card_w + gutter)
                y_px = margin + r * (card_h + gutter)
                x_pt = x_px * px_to_pt
                y_pt = (target_page_h - y_px - card_h) * px_to_pt
                w_pt = card_w * px_to_pt
                h_pt = card_h * px_to_pt
                c.drawImage(cell_reader, x_pt, y_pt, width=w_pt, height=h_pt, mask='auto')
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

# --------------------------- CLI and Orchestration ---------------------------

def read_monsters(path: Optional[str], demo: bool, images_dir: str) -> List[Monster]:
    data: List[dict]
    if demo:
        demo_path = os.path.join(os.path.dirname(__file__), "example_monsters.json")
        if not os.path.isfile(demo_path):
            warn("example_monsters.json not found; demo cannot proceed.")
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

    monsters: List[Monster] = []
    for i, m in enumerate(data):
        name = m.get("name")
        if not name:
            warn(f"Monster index {i} missing name; skipping.")
            continue
        # Resolve profile image path using default images_dir and declared value
        declared_image = m.get("profile_image")
        resolved_image = resolve_profile_image(name, declared_image, images_dir)
        if declared_image and not resolved_image:
            warn(f"Image not found for '{name}': {declared_image}. Will try placeholder.")
        if declared_image and resolved_image and os.path.normpath(declared_image) != os.path.normpath(resolved_image):
            info(f"Image for '{name}': '{declared_image}' not found; using '{os.path.relpath(resolved_image)}'")
        if not declared_image and resolved_image:
            info(f"Auto-selected image for '{name}': {os.path.relpath(resolved_image)}")
        monsters.append(
            Monster(
                id=m.get("id"),
                name=name,
                profile_image=resolved_image,
                lore=m.get("lore") or "",
                culinary_use=m.get("culinary_use"),
                statblock=m.get("statblock") or {},
                abilities=m.get("abilities") or [],
                actions=m.get("actions") or [],
                background=m.get("background"),
                fonts_overrides=(m.get("fonts") or {}),
            )
        )
    return monsters


def save_pngs(cards: List[Image.Image], monsters: List[Monster], outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []
    for idx, (img, mon) in enumerate(zip(cards, monsters)):
        slug = slugify(mon.name or f"monster_{idx}")
        fname = f"{idx:02d}_{slug}.png"
        fpath = os.path.join(outdir, fname)
        img.save(fpath, format="PNG")
        info(f"Saved {fpath}")
        paths.append(fpath)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DnD Monster Flashcard Generator")
    parser.add_argument("input", nargs="?", help="Input JSON file containing monsters")
    parser.add_argument("--outdir", default="./out_cards", help="Directory to save PNGs")
    parser.add_argument("--images-dir", default="./images", help="Default images directory for monster portraits")
    parser.add_argument("--theme", default=None, help="Apply a curated font theme (e.g., 'fantasy'); looks for TTFs in ./fonts")
    parser.add_argument("--pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--pdf-bg", default=None, help="Optional PDF page background image path (A4 portrait)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--card-width", type=int, default=None, help="Card width (px)")
    parser.add_argument("--card-height", type=int, default=None, help="Card height (px)")
    parser.add_argument("--font-default", default=None, help="Default font path or name")
    parser.add_argument("--font-name", default=None, help="Name/title font path or name")
    parser.add_argument("--font-lore", default=None, help="Lore font path or name")
    parser.add_argument("--font-stats", default=None, help="Stats font path or name")
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="PDF page margin (px)")
    parser.add_argument("--gutter", type=int, default=DEFAULT_GUTTER, help="PDF gutter (px)")
    parser.add_argument("--dpi-cards-per-page", type=int, default=DEFAULT_CARDS_PER_PAGE, help="Cards per page for PDF (default 9)")
    parser.add_argument("--full-bleed", action="store_true", help="Fill entire PDF page with the 3x3 grid (no margins/gutters)")
    parser.add_argument("--cut-lines", action="store_true", help="Draw cut guide lines between cards on the PDF page")
    parser.add_argument("--cut-line-width", type=float, default=0.5, help="Cut line width in points (PDF units)")
    parser.add_argument("--demo", action="store_true", help="Generate sample cards from example_monsters.json")
    args = parser.parse_args(argv)

    # Compute default card size: A4 @300dpi / 3x3 grid
    default_card_w = A4_WIDTH_PX_300DPI // DEFAULT_GRID_COLS
    default_card_h = A4_HEIGHT_PX_300DPI // DEFAULT_GRID_ROWS
    card_w = args.card_width or default_card_w
    card_h = args.card_height or default_card_h

    monsters = read_monsters(args.input, demo=args.demo, images_dir=args.images_dir)
    if not monsters:
        warn("No monsters to render.")
        return 1

    # Theme fonts (if requested) provide defaults which can be overridden by explicit flags
    theme_cfg = resolve_theme_fonts(args.theme)
    fonts_cfg = FontsConfig(
        default=args.font_default or (theme_cfg.default if theme_cfg else None),
        name=args.font_name or (theme_cfg.name if theme_cfg else None),
        lore=args.font_lore or (theme_cfg.lore if theme_cfg else None),
        stats=args.font_stats or (theme_cfg.stats if theme_cfg else None),
    )
    cfg = CardConfig(width=card_w, height=card_h, dpi=args.dpi, fonts=fonts_cfg)

    # Render cards
    cards: List[Image.Image] = []
    for i, mon in enumerate(monsters):
        info(f"Rendering card {i+1}/{len(monsters)}: {mon.name}")
        img = render_card(mon, cfg)
        cards.append(img)

    # Save PNGs
    png_paths = save_pngs(cards, monsters, args.outdir)

    # Compose PDF if requested
    if args.pdf:
        info("Composing PDF...")
        compose_pdf(cards, args.pdf, dpi=args.dpi, margin=args.margin, gutter=args.gutter, cards_per_page=args.dpi_cards_per_page, page_background_path=args.pdf_bg, full_bleed=args.full_bleed, cut_lines=args.cut_lines, cut_line_width_pt=args.cut_line_width)

    info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
