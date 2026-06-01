from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import os
from matplotlib import font_manager as _fm

def _setup_japanese_font():
    # ビルド時にダウンロードしたフォント、なければシステムフォントを探す
    candidates = [
        os.path.join(os.path.dirname(__file__), "NotoSansJP.otf"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
    ]
    for path in candidates:
        if os.path.exists(path):
            _fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = _fm.FontProperties(fname=path).get_name()
            return

_setup_japanese_font()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _luminance(color) -> float:
    """WCAG relative luminance from matplotlib color (rgba tuple or hex str)."""
    if isinstance(color, str):
        h = color.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    else:
        r, g, b = float(color[0]), float(color[1]), float(color[2])

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _pick_colors(n: int):
    if n in (5, 7):
        # Blue gradient (light → dark)
        return [plt.cm.Blues(v) for v in np.linspace(0.25, 0.85, n)]
    # Qualitative: muted, no primaries, good mutual contrast
    palette = [
        "#5B8DB8", "#E8956D", "#6BAB7A", "#B5838D", "#9B7BB5",
        "#5BA4A4", "#D4A843", "#8C7B6E", "#84A98C", "#C4826E",
        "#6E8FAE", "#A8B86E",
    ]
    return palette[:n]


def _parse(data: str):
    lines = [l for l in data.strip('\n\r').splitlines() if l.strip()]
    rows = [line.split("\t") for line in lines]
    if len(rows) < 2:
        raise ValueError("データが2行以上必要です（ヘッダー行 + データ行）")

    col_headers = [h.strip() for h in rows[0][1:]]
    row_labels = [r[0].strip() for r in rows[1:]]

    try:
        def to_float(v):
            s = v.replace(",", "").replace(" ", "").replace("　", "").strip()
            return float(s) if s else 0.0

        matrix = np.array(
            [[to_float(v) for v in row[1:]] for row in rows[1:]]
        )
    except ValueError as e:
        raise ValueError(f"数値に変換できないセルがあります: {e}") from e

    if matrix.ndim != 2 or matrix.shape[1] != len(col_headers):
        raise ValueError("行と列の数が一致しません。ヘッダー行を確認してください。")

    return col_headers, row_labels, matrix


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate(
    data: str = Form(...),
    title: str = Form(""),
    xlabel: str = Form(""),
    ylabel: str = Form(""),
    note: str = Form(""),
):
    try:
        col_headers, row_labels, matrix = _parse(data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n_rows, n_cols = matrix.shape
    col_totals = matrix.sum(axis=0)
    grand_total = col_totals.sum()
    col_widths = col_totals / grand_total          # 0–1, sum == 1
    col_props = matrix / col_totals[np.newaxis, :]  # proportions within column

    colors = _pick_colors(n_rows)

    # ---- figure size (auto-scale with number of columns) ------------------
    fig_w = max(12, n_cols * 2.8)
    fig_h = 8.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")

    # ---- draw bars --------------------------------------------------------
    col_positions = []   # (x_left, width) for each column
    x = 0.0
    for j in range(n_cols):
        w = col_widths[j]
        col_positions.append((x, w))
        y = 0.0
        for i in range(n_rows):
            h = col_props[i, j]
            c = colors[i]
            ax.add_patch(
                mpatches.Rectangle((x, y), w, h, fc=c, ec="white", lw=1.5, zorder=2)
            )
            # % inside cell (skip if too small to read)
            if h > 0.04:
                text_color = "white" if _luminance(c) < 0.38 else "#222222"
                ax.text(
                    x + w / 2, y + h / 2,
                    f"{h * 100:.1f}%",
                    ha="center", va="center",
                    fontsize=9, color=text_color, fontweight="bold", zorder=3,
                )
            y += h
        x += w

    # ---- n labels immediately to the left of each column ------------------
    GAP = 0.012
    for j, (cx, cw) in enumerate(col_positions):
        ax.text(
            cx - GAP, 0.5,
            f"n={int(col_totals[j])}",
            ha="right", va="center", fontsize=9, color="#444444", zorder=3,
        )

    # ---- column headers (X axis) ------------------------------------------
    rotate = max(len(h) for h in col_headers) > 8
    for j, (cx, cw) in enumerate(col_positions):
        ax.text(
            cx + cw / 2, -0.035,
            col_headers[j],
            ha="center", va="top", fontsize=10,
            rotation=30 if rotate else 0,
            rotation_mode="anchor",
        )

    # ---- axis labels -------------------------------------------------------
    y_label_text = ylabel if ylabel else "【ここにY軸ラベルを入力】"
    y_label_color = "#111111" if ylabel else "#BBBBBB"
    ax.text(
        -0.13, 0.5, y_label_text,
        ha="center", va="center", rotation=90, fontsize=10,
        color=y_label_color, style="italic" if not ylabel else "normal",
    )

    x_label_text = xlabel if xlabel else "【ここにX軸ラベルを入力】"
    x_label_color = "#111111" if xlabel else "#BBBBBB"
    ax.text(
        0.5, -0.12 if rotate else -0.09,
        x_label_text,
        ha="center", va="top", fontsize=10,
        color=x_label_color, style="italic" if not xlabel else "normal",
    )

    # ---- legend -----------------------------------------------------------
    patches = [mpatches.Patch(fc=colors[i], label=row_labels[i]) for i in range(n_rows)]
    ax.legend(
        handles=patches,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
    )

    # ---- axes limits & frame ----------------------------------------------
    ax.set_xlim(-0.18, 1.03)
    ax.set_ylim(-0.18 if rotate else -0.15, 1.06)
    ax.axis("off")

    # ---- title ------------------------------------------------------------
    title_text = title if title else "【ここにグラフタイトルを入力】"
    title_color = "#111111" if title else "#BBBBBB"
    fig.suptitle(
        title_text, fontsize=14, fontweight="bold",
        color=title_color, style="italic" if not title else "normal", y=0.98,
    )

    # ---- note & grand total -----------------------------------------------
    note_text = note if note else "【出典・注記をここに入力】"
    note_color = "#555555" if note else "#CCCCCC"
    fig.text(0.01, 0.005, note_text, fontsize=8, color=note_color)
    fig.text(0.99, 0.005, f"合計 n={int(grand_total)}", fontsize=8,
             color="#555555", ha="right")

    plt.tight_layout(rect=[0.05, 0.05, 0.88, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="marimekko.png"'},
    )
