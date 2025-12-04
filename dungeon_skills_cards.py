#!/usr/bin/env python3
"""
Dungeon Skill Flashcard Generator

Generates PNG flashcards for dungeon skills from a JSON input, and optionally
arranges them into an A4 multi-page PDF (3x3 grid). This reuses the layout
and rendering logic from monster_cards.py but simplifies the schema to match
skill cards:

Fields used from JSON per skill object:
    - id            (optional) numeric or string id
    - title         Skill title (e.g., "Monster Chef")
    - lore          Lore blurb (rendered in italics)
    - usage_label   Usage label (rendered in bold, e.g., "Once per floor")
    - body          Main rules text describing how the skill works

Usage:
    python dungeon_skills_cards.py dungeon_skills.json --outdir skill_cards --pdf skills.pdf --dpi 300

See dungeon_skills.json for an example input file.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# Constants shared with monster_cards.py style
A4_WIDTH_PX_300DPI = 2480
A4_HEIGHT_PX_300DPI = 3508
DEFAULT_GRID_COLS = 3
DEFAULT_GRID_ROWS = 3
DEFAULT_CARDS_PER_PAGE = DEFAULT_GRID_COLS * DEFAULT_GRID_ROWS
DEFAULT_MARGIN = 90
DEFAULT_GUTTER = 45
FONT_SCALE = 1.0


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def load_json(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_color(s: Optional[str], fallback=(245, 245, 245)):
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


def fit_font(fontpath_or_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if fontpath_or_name and os.path.isfile(fontpath_or_name):
        try:
            return ImageFont.truetype(fontpath_or_name, size=size)
        except Exception as e:
            warn(f"Failed to load font from path {fontpath_or_name}: {e}")
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        # Fallback: load_default returns an ImageFont.ImageFont, which is
        # sufficient for rendering even though it's not a FreeTypeFont.
        return ImageFont.load_default()  # type: ignore[return-value]


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
    title: Optional[str] = None
    lore: Optional[str] = None
    body: Optional[str] = None


def resolve_theme_fonts(theme: Optional[str]) -> Optional[FontsConfig]:
    """Optional themed fonts, mirroring monster_cards.py's 'fantasy' theme.

    Place the TTF files in ./fonts next to this script. If no themed fonts are
    found, returns None and the caller will fall back to explicit font flags
    or default fonts.
    """
    if not theme:
        return None
    theme = theme.lower().strip()
    THEMES = {
        "fantasy": {
            "title": [
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
            "body": [
                "Cinzel-Regular.ttf",
                "Caudex-Regular.ttf",
                "CormorantGaramond-Regular.ttf",
                "Cardo-Regular.ttf",
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

    def pick(key: str) -> Optional[str]:
        files = cfg.get(key) or []
        return find_font_file(files)

    default = pick("default")
    title = pick("title")
    lore = pick("lore")
    body = pick("body")
    if not any([default, title, lore, body]):
        warn("No themed fonts found in ./fonts. Using built-in defaults instead.")
        return None
    if title:
        info(f"Theme '{theme}': using title font {os.path.basename(title)}")
    if lore:
        info(f"Theme '{theme}': using lore font {os.path.basename(lore)}")
    if body:
        info(f"Theme '{theme}': using body font {os.path.basename(body)}")
    if default:
        info(f"Theme '{theme}': using default font {os.path.basename(default)}")
    return FontsConfig(default=default, title=title, lore=lore, body=body)


@dataclass
class CardConfig:
    width: int
    height: int
    dpi: int
    fonts: FontsConfig


@dataclass
class DungeonSkill:
    id: Optional[str]
    title: str
    lore: str
    usage_label: str
    body: str


def measure_wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int, line_spacing: int = 4):
    def text_size(s: str):
        bbox = draw.textbbox((0, 0), s, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

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


def best_fit_wrapped(draw: ImageDraw.ImageDraw, text: str, font_path: Optional[str], max_width: int, max_height: int, min_size: int, max_size: int):
    lo, hi = min_size, max_size
    best_font = fit_font(font_path, min_size)
    _, _, best_lines = measure_wrapped_text(draw, text, best_font, max_width)
    while lo <= hi:
        mid = (lo + hi) // 2
        font = fit_font(font_path, mid)
        _, h, lines = measure_wrapped_text(draw, text, font, max_width)
        if h <= max_height:
            best_font, best_lines = font, lines
            lo = mid + 1
        else:
            hi = mid - 1
    return best_font, best_lines


def render_skill_card(skill: DungeonSkill, cfg: CardConfig) -> Image.Image:
    # Like monster cards, render in landscape and rotate to portrait at the end
    W, H = cfg.width, cfg.height
    LW, LH = H, W

    bg_color = parse_color(None, fallback=(245, 238, 220))
    base = Image.new("RGBA", (LW, LH), bg_color + (255,))

    # Optional layered background: parchment.jpg under redborder.png
    base_dir = os.path.dirname(__file__)
    parchment_path = os.path.join(base_dir, "parchment.jpg")
    redborder_path = os.path.join(base_dir, "redborder.png")
    # First parchment: scale independently on X and Y to exactly LW x LH, no cropping
    if os.path.isfile(parchment_path):
        try:
            bg_img = Image.open(parchment_path).convert("RGBA")
            bg_img = bg_img.resize((LW, LH), resample=Image.Resampling.LANCZOS)
            base.paste(bg_img, (0, 0))
        except Exception as e:
            warn(f"Failed to load card background '{parchment_path}': {e}")
    # Then red border overlay: also stretched to full canvas, preserving alpha
    if os.path.isfile(redborder_path):
        try:
            border_img = Image.open(redborder_path).convert("RGBA")
            border_img = border_img.resize((LW, LH), resample=Image.Resampling.LANCZOS)
            base.paste(border_img, (0, 0), border_img)
        except Exception as e:
            warn(f"Failed to load card border '{redborder_path}': {e}")

    draw = ImageDraw.Draw(base)

    # Slightly larger padding for a more spacious layout than monster cards
    pad = int(min(LW, LH) * 0.075)
    x, y = pad, pad
    content_w = LW - 2 * pad
    content_h = LH - 2 * pad

    fc = cfg.fonts
    title_font_path = fc.title or fc.default
    lore_font_path = fc.lore or fc.default
    body_font_path = fc.body or fc.default

    # Section: "Dungeon Skill" header centered at very top
    header_font_size = max(18, int(LH * 0.06))
    header_font = fit_font(title_font_path, header_font_size)
    header_text = "Dungeon Skill"
    hb = draw.textbbox((0, 0), header_text, font=header_font)
    hh = hb[3] - hb[1]
    hw = hb[2] - hb[0]
    hx = x + (content_w - hw) // 2
    hy = y
    draw.text((hx, hy), header_text, font=header_font, fill=(40, 25, 10, 255))
    draw.text((hx + 1, hy), header_text, font=header_font, fill=(40, 25, 10, 255))
    ty = hy + hh + int(pad * 0.3)

    # Title just below header
    def ellipsize(draw, text, font, max_width: int) -> str:
        if draw.textbbox((0, 0), text, font=font)[2] <= max_width:
            return text
        ell = "…"
        lo, hi = 0, len(text)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            s = text[:mid] + ell
            if draw.textbbox((0, 0), s, font=font)[2] <= max_width:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return (text[:best] + ell) if best > 0 else ell

    title_size = max(24, int(LH * 0.10))
    title_font = fit_font(title_font_path, int(title_size * FONT_SCALE))
    title_text = ellipsize(draw, skill.title or "Unnamed Skill", title_font, content_w)
    tb = draw.textbbox((0, 0), title_text, font=title_font)
    th = tb[3] - tb[1]
    draw.text((x, ty), title_text, font=title_font, fill=(30, 20, 10, 255))
    draw.text((x + 1, ty), title_text, font=title_font, fill=(30, 20, 10, 255))
    ty += th + int(pad * 0.25)

    # Lore (italic-style body) block – italicize by using lore font and tone
    header_gap = int(pad * 0.2)
    inner_margin = int(LH * 0.015)

    # Lore block height ~22% of content, usage ~12%, body uses the rest
    lore_h = int(content_h * 0.22)
    usage_h = int(content_h * 0.12)
    body_h = max(0, content_h - lore_h - usage_h - 3 * inner_margin)

    lore_font, lore_lines = best_fit_wrapped(
        draw,
        skill.lore or "",
        lore_font_path,
        max_width=content_w,
        max_height=lore_h,
        min_size=12,
        max_size=int(max(18, lore_h)),
    )
    lore_color = (50, 40, 30, 255)
    lore_line_h = draw.textbbox((0, 0), "Ag", font=lore_font)[3] - draw.textbbox((0, 0), "Ag", font=lore_font)[1]
    used = 0
    for line in lore_lines:
        if used + lore_line_h > lore_h:
            break
        draw.text((x, ty), line, font=lore_font, fill=lore_color)
        ty += lore_line_h + 2
        used += lore_line_h + 2

    ty = y + th + header_gap + lore_h
    ty += inner_margin

    # Usage label (bold-ish), with explicit "Usage:" prefix
    usage_font_size = max(18, int(LH * 0.06))
    usage_font = fit_font(body_font_path, usage_font_size)
    raw_usage = skill.usage_label or ""
    usage_text = f"Usage: {raw_usage}" if raw_usage else "Usage:"
    ub = draw.textbbox((0, 0), usage_text, font=usage_font)
    uh = ub[3] - ub[1]
    draw.text((x, ty), usage_text, font=usage_font, fill=(20, 10, 5, 255))
    draw.text((x + 1, ty), usage_text, font=usage_font, fill=(20, 10, 5, 255))
    ty += uh + inner_margin

    # Body text block – dynamic scaling via best_fit_wrapped to fully use space
    body_font, body_lines = best_fit_wrapped(
        draw,
        skill.body or "",
        body_font_path,
        max_width=content_w,
        max_height=body_h,
        min_size=11,
        max_size=int(max(26, body_h)),
    )
    body_color = (35, 25, 15, 255)
    body_line_h = draw.textbbox((0, 0), "Ag", font=body_font)[3] - draw.textbbox((0, 0), "Ag", font=body_font)[1]
    used = 0
    for line in body_lines:
        if used + body_line_h > body_h:
            break
        draw.text((x, ty), line, font=body_font, fill=body_color)
        ty += body_line_h + 3
        used += body_line_h + 3

    final_img = base.convert("RGB").rotate(-90, expand=True)
    final_img = ImageOps.contain(final_img, (W, H), method=Image.Resampling.LANCZOS)
    return final_img


def compose_pdf(
    cards: List[Image.Image],
    outfile: str,
    dpi: int,
    margin: int,
    gutter: int,
    cards_per_page: int = DEFAULT_CARDS_PER_PAGE,
    page_background_path: Optional[str] = None,
    full_bleed: bool = True,
    cut_lines: bool = False,
    cut_line_width_pt: float = 0.5,
) -> None:
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

    # For full-bleed, ignore margin/gutter and fill the page
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

        # Optional page background image
        if page_background_path and os.path.isfile(page_background_path):
            try:
                bg_img = Image.open(page_background_path).convert("RGBA")
                bg_img = ImageOps.fit(bg_img, (target_page_w, target_page_h), method=Image.Resampling.LANCZOS)
                buf_bg = io.BytesIO()
                bg_img.save(buf_bg, format="PNG")
                img_bg_data = buf_bg.getvalue()
                bg_reader = ImageReader(io.BytesIO(img_bg_data))
                c.drawImage(bg_reader, 0, 0, width=target_page_w * px_to_pt, height=target_page_h * px_to_pt)
            except Exception as e:
                warn(f"Failed to draw PDF background '{page_background_path}': {e}")

        # Front side: cards
        for idx, card in enumerate(batch):
            r = idx // cols
            col = idx % cols
            x_px = margin + col * (card_w + gutter)
            y_px = margin + r * (card_h + gutter)
            card_resized = ImageOps.fit(card, (card_w, card_h), method=Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            card_resized.save(buf, format="PNG")
            img_data = buf.getvalue()
            x_pt = x_px * px_to_pt
            y_pt = (target_page_h - y_px - card_h) * px_to_pt
            w_pt = card_w * px_to_pt
            h_pt = card_h * px_to_pt
            img_reader = ImageReader(io.BytesIO(img_data))
            c.drawImage(img_reader, x_pt, y_pt, width=w_pt, height=h_pt, mask="auto")

        # Optional cut lines
        if cut_lines:
            c.setLineWidth(cut_line_width_pt)
            # Vertical lines
            for col_i in range(1, cols):
                x_px = margin + col_i * (card_w + gutter) - (0 if gutter == 0 else gutter // 2)
                x_pt = x_px * px_to_pt
                c.line(x_pt, 0, x_pt, target_page_h * px_to_pt)
            # Horizontal lines
            for row_i in range(1, rows):
                y_px = margin + row_i * (card_h + gutter) - (0 if gutter == 0 else gutter // 2)
                y_pt = (target_page_h - y_px) * px_to_pt
                c.line(0, y_pt, target_page_w * px_to_pt, y_pt)

        c.showPage()

        # Backside page: same grid with leather/parchment background and logo
        try:
            # Prefer brownleather.png for dungeon skill backs; fallback to leather.jpg, then parchment.jpg
            brown_leather_path = os.path.join(os.path.dirname(__file__), "brownleather.png")
            leather_path = os.path.join(os.path.dirname(__file__), "leather.jpg")
            parchment_path = os.path.join(os.path.dirname(__file__), "parchment.jpg")
            logo_path = os.path.join(os.path.dirname(__file__), "dnd_logo.png")
            cell = Image.new("RGBA", (card_w, card_h), (245, 238, 220, 255))

            bg_used = None
            if os.path.isfile(brown_leather_path):
                bg_used = brown_leather_path
            elif os.path.isfile(leather_path):
                bg_used = leather_path
            elif os.path.isfile(parchment_path):
                bg_used = parchment_path
            if bg_used:
                try:
                    pimg = Image.open(bg_used).convert("RGBA")
                    # Stretch backing texture to exactly card_w x card_h, with no cropping
                    pimg = pimg.resize((card_w, card_h), resample=Image.Resampling.LANCZOS)
                    cell.paste(pimg, (0, 0))
                except Exception as e:
                    warn(f"Failed to load backside background '{bg_used}': {e}")

            if os.path.isfile(logo_path):
                try:
                    limg = Image.open(logo_path).convert("RGBA")
                    # Rotate logo 90° clockwise to match card orientation
                    limg = limg.rotate(-90, expand=True)
                    max_logo_w = int(card_w * 0.6)
                    max_logo_h = int(card_h * 0.6)
                    limg = ImageOps.contain(limg, (max_logo_w, max_logo_h), method=Image.Resampling.LANCZOS)
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


def read_skills(path: str) -> List[DungeonSkill]:
    if not os.path.isfile(path):
        warn(f"Input JSON not found: {path}")
        return []
    data = load_json(path)
    skills: List[DungeonSkill] = []
    for i, s in enumerate(data):
        title = s.get("title")
        if not title:
            warn(f"Skill index {i} missing title; skipping.")
            continue
        skills.append(
            DungeonSkill(
                id=str(s.get("id")) if s.get("id") is not None else None,
                title=title,
                lore=s.get("lore") or "",
                usage_label=s.get("usage_label") or "Usage",
                body=s.get("body") or "",
            )
        )
    return skills


def save_pngs(cards: List[Image.Image], skills: List[DungeonSkill], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    paths: List[str] = []
    for idx, (img, sk) in enumerate(zip(cards, skills)):
        slug = slugify(sk.title or f"skill_{idx}")
        fname = f"{idx:02d}_{slug}.png"
        fpath = os.path.join(outdir, fname)
        img.save(fpath, format="PNG")
        info(f"Saved {fpath}")
        paths.append(fpath)
    return paths


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Dungeon Skill Flashcard Generator")
    parser.add_argument("input", help="Input JSON file containing dungeon skills")
    parser.add_argument("--outdir", default="./out_cards_skills", help="Directory to save PNGs")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output images")
    parser.add_argument("--card-width", type=int, default=None, help="Card width (px)")
    parser.add_argument("--card-height", type=int, default=None, help="Card height (px)")
    parser.add_argument("--font-default", default=None, help="Default font path or name")
    parser.add_argument("--font-title", default=None, help="Title font path or name")
    parser.add_argument("--font-lore", default=None, help="Lore font path or name")
    parser.add_argument("--font-body", default=None, help="Body font path or name")
    parser.add_argument("--theme", default=None, help="Optional font theme (e.g., 'fantasy') using TTFs in ./fonts")
    parser.add_argument("--pdf", default=None, help="Optional output PDF path")
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="PDF margin (px)")
    parser.add_argument("--gutter", type=int, default=DEFAULT_GUTTER, help="PDF gutter (px)")
    parser.add_argument("--cards-per-page", type=int, default=DEFAULT_CARDS_PER_PAGE, help="Cards per PDF page")
    parser.add_argument("--pdf-bg", default=None, help="Optional PDF page background image path (A4 portrait)")
    parser.add_argument("--full-bleed", action="store_true", help="Fill entire PDF page with the 3x3 grid (no margins/gutters)")
    parser.add_argument("--cut-lines", action="store_true", help="Draw cut guide lines between cards on the PDF page")
    parser.add_argument("--cut-line-width", type=float, default=0.5, help="Cut line width in points (PDF units)")
    args = parser.parse_args(argv)

    default_card_w = A4_WIDTH_PX_300DPI // DEFAULT_GRID_COLS
    default_card_h = A4_HEIGHT_PX_300DPI // DEFAULT_GRID_ROWS
    card_w = args.card_width or default_card_w
    card_h = args.card_height or default_card_h

    skills = read_skills(args.input)
    if not skills:
        warn("No skills to render.")
        return 1

    # Theme fonts (if requested) provide defaults that can be overridden by flags
    theme_cfg = resolve_theme_fonts(args.theme)
    fonts_cfg = FontsConfig(
        default=args.font_default or (theme_cfg.default if theme_cfg else None),
        title=args.font_title or (theme_cfg.title if theme_cfg else None),
        lore=args.font_lore or (theme_cfg.lore if theme_cfg else None),
        body=args.font_body or (theme_cfg.body if theme_cfg else None),
    )
    cfg = CardConfig(width=card_w, height=card_h, dpi=args.dpi, fonts=fonts_cfg)

    cards: List[Image.Image] = []
    for i, sk in enumerate(skills):
        info(f"Rendering skill card {i+1}/{len(skills)}: {sk.title}")
        img = render_skill_card(sk, cfg)
        cards.append(img)

    save_pngs(cards, skills, args.outdir)

    if args.pdf:
        info("Composing PDF...")
        compose_pdf(
            cards,
            args.pdf,
            dpi=args.dpi,
            margin=args.margin,
            gutter=args.gutter,
            cards_per_page=args.cards_per_page,
            page_background_path=args.pdf_bg,
            full_bleed=args.full_bleed or True,
            cut_lines=args.cut_lines,
            cut_line_width_pt=args.cut_line_width,
        )

    info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
