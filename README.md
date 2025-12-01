# D&D Monster Flashcard Generator

A Python tool for generating beautiful, print-ready flashcards for D&D 5e monsters. Creates individual PNG cards and arranges them into A4 PDFs with professional layouts, custom fonts, and double-sided printing support.

## Features

- **Beautiful Card Design**: Landscape layout with rounded profile images, drop shadows, and themed backgrounds
- **Flexible Typography**: Support for custom fonts with theme presets (e.g., "fantasy" theme)
- **Smart Text Fitting**: Dynamic font sizing ensures all content fits perfectly on each card
- **PDF Output**: Automatically arranges cards into A4 pages (3x3 grid) with double-sided printing support
- **Custom Backgrounds**: Per-card color or image backgrounds with parchment/leather textures
- **Auto Image Resolution**: Automatically finds monster portraits from multiple naming conventions
- **Professional Printing**: Cut lines, full bleed options, and configurable margins/gutters

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository:
```bash
git clone https://github.com/HeatPhoenix/dndcardgen.git
cd dndcardgen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Demo Mode

Generate example cards to see the tool in action:

```bash
python monster_cards.py --demo --pdf demo_output.pdf
```

This creates cards from `example_monsters.json` and saves them to:
- `./out_cards/` - Individual PNG files
- `demo_output.pdf` - Print-ready PDF

### Basic Usage

```bash
python monster_cards.py monsters.json --pdf output.pdf
```

## Usage

### Command Line Options

```bash
python monster_cards.py [INPUT_JSON] [OPTIONS]
```

#### Required Arguments
- `INPUT_JSON` - Path to your monster data JSON file (not needed with `--demo`)

#### Output Options
- `--outdir DIR` - Directory for PNG output (default: `./out_cards`)
- `--pdf FILE` - Generate PDF at specified path (e.g., `monsters.pdf`)
- `--dpi N` - Output resolution in DPI (default: 300)

#### Image Options
- `--images-dir DIR` - Directory containing monster portraits (default: `./images`)

#### Font Options
- `--theme THEME` - Use a curated font theme (e.g., `fantasy`)
- `--font-default PATH` - Default fallback font
- `--font-name PATH` - Font for monster names/titles
- `--font-lore PATH` - Font for lore text (italic preferred)
- `--font-stats PATH` - Font for stats and section headers

#### PDF Layout Options
- `--margin N` - Page margin in pixels (default: 90)
- `--gutter N` - Space between cards in pixels (default: 45)
- `--card-width N` - Custom card width in pixels
- `--card-height N` - Custom card height in pixels
- `--pdf-bg FILE` - Background image for PDF pages
- `--full-bleed` - Remove margins/gutters for full-page coverage
- `--cut-lines` - Draw cutting guide lines between cards
- `--cut-line-width N` - Cut line width in points (default: 0.5)

### Examples

#### Basic PDF Generation
```bash
python monster_cards.py my_monsters.json --pdf cards.pdf
```

#### With Fantasy Theme Fonts
```bash
python monster_cards.py my_monsters.json --theme fantasy --pdf cards.pdf
```

#### High-Resolution Print
```bash
python monster_cards.py my_monsters.json --dpi 600 --pdf high_res.pdf
```

#### Custom Layout with Cut Lines
```bash
python monster_cards.py my_monsters.json --pdf cards.pdf --cut-lines --margin 60 --gutter 30
```

## JSON Input Format

Create a JSON file with an array of monster objects:

```json
[
  {
    "id": "1",
    "name": "Kelpie",
    "profile_image": "kelpie.jpg",
    "size": "Medium",
    "type": "Beast",
    "lore": "Kelpies are territorial water predators that use cunning and deception to lure prey.",
    "culinary_use": "The meat is much like a mix between tuna and horsemeat.",
    "statblock": {
      "Armor Class": "13",
      "Hit Points": "52 (8d10+16)",
      "Speed": "20 ft., swim 40 ft.",
      "STR": "16 (+3)",
      "DEX": "14 (+2)",
      "CON": "14 (+2)",
      "INT": "4 (-3)",
      "WIS": "12 (+1)",
      "CHA": "6 (-2)",
      "Senses": "Darkvision 60 ft.",
      "Challenge": "3 (700 XP)"
    },
    "abilities": [
      {
        "name": "Amorphous",
        "text": "The creature can occupy another creature's space."
      }
    ],
    "actions": [
      {
        "name": "Bite",
        "text": "Melee Weapon Attack: +5 to hit, reach 5 ft., one target. Hit: 12 (2d8 + 3) piercing damage."
      }
    ],
    "background": "#cceef6",
    "fonts": {
      "name": null,
      "lore": null,
      "stats": null
    }
  }
]
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Unique identifier for the monster |
| `name` | string | **Yes** | Monster name (displayed as card title) |
| `profile_image` | string | No | Path to portrait image (auto-detected if missing) |
| `size` | string | No | Size category (e.g., "Medium", "Large") |
| `type` or `creature_type` | string | No | Creature type (e.g., "Beast", "Undead") |
| `lore` | string | No | Descriptive text about the monster |
| `culinary_use` | string | No | Optional cooking/preparation notes |
| `statblock` | object | No | Key-value pairs of stats (AC, HP, abilities, etc.) |
| `abilities` | array | No | Special abilities with `name` and `text` |
| `actions` | array | No | Actions with `name` and `text` |
| `background` | string | No | Hex color (e.g., `#cceef6`) or image path |
| `fonts` | object | No | Per-card font overrides |

## Image Management

### Directory Structure
```
dndcardgen/
├── images/              # Monster portraits
│   ├── kelpie.jpg
│   ├── dragon.png
│   └── owlbear.webp
├── parchment.jpg        # Default card background
├── leather.jpg          # Default PDF backside background
└── dnd_logo.png         # Logo for card backs
```

### Image Resolution

The tool automatically searches for monster images using multiple naming patterns:
- Exact filename from `profile_image` field
- Slugified monster name (spaces → underscores/hyphens)
- Various extensions (.png, .jpg, .jpeg, .webp)

**Example**: For a monster named "Fire Drake", it will try:
- `Fire Drake.png`
- `fire_drake.png`
- `fire-drake.jpg`
- etc.

If no image is found, a placeholder silhouette is generated.

## Font Themes

### Fantasy Theme

Place these TTF files in the `./fonts` directory:

**Title Fonts** (bold, decorative):
- CinzelDecorative-Black.ttf
- CinzelDecorative-Bold.ttf
- UncialAntiqua-Regular.ttf
- MedievalSharp-Regular.ttf

**Lore Fonts** (italic, readable):
- Cardo-Italic.ttf
- EBGaramond-Italic.ttf
- IMFellEnglish-Italic.ttf
- CormorantGaramond-Italic.ttf

**Stats Fonts** (clear, bold):
- Cinzel-Bold.ttf
- Cinzel-Regular.ttf
- Caudex-Bold.ttf
- CormorantGaramond-Bold.ttf

All these fonts are free and available from [Google Fonts](https://fonts.google.com/).

### Using Themes

```bash
# Download fonts to ./fonts/ directory, then:
python monster_cards.py monsters.json --theme fantasy --pdf cards.pdf
```

## PDF Output

### Layout

- **Grid**: 3x3 cards per page (9 cards/page)
- **Double-Sided**: Automatically generates back pages with leather texture and D&D logo
- **Page Size**: A4 (210mm × 297mm)
- **Default Resolution**: 300 DPI (suitable for printing)

### Printing Tips

1. **Two-Sided Printing**: The PDF includes alternating front/back pages
2. **Cut Lines**: Use `--cut-lines` to add guides for cutting
3. **Full Bleed**: Use `--full-bleed` for edge-to-edge printing
4. **High Quality**: Use `--dpi 600` for professional printing services

## Troubleshooting

### Common Issues

**"Image not found" warnings**
- Check that images are in `./images/` directory
- Verify filename matches the monster name or `profile_image` field
- Use `--images-dir` to specify a different directory

**Fonts look wrong**
- Install theme fonts in `./fonts/` directory
- Use `--theme fantasy` after downloading fonts
- Specify custom fonts with `--font-name`, `--font-lore`, etc.

**Text doesn't fit**
- The tool auto-shrinks text, but very long content may be truncated
- Reduce text length in JSON input
- Increase `--card-width` and `--card-height`

**PDF layout issues**
- Adjust `--margin` and `--gutter` for spacing
- Use `--full-bleed` to maximize card size
- Try `--cut-lines` to visualize the grid

## Dependencies

- **Pillow** (PIL Fork): Image processing and rendering
- **ReportLab**: PDF generation

Install with:
```bash
pip install pillow reportlab
```
