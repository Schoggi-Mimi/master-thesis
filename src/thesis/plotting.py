import cv2
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from src.eval.cam_eval_utils import CROP_SIZE

from .cams import norm01
from .config import FIG, GROUP_COLOR

TEXTWIDTH = 6.3          # inches, A4 10pt single column
C_FOCUS = "#C62828"      # the result the reader should look at
C_CONTROL = "#4D4D4D"    # the direct comparison
C_CONTEXT = "#BDBDBD"    # shown for reference, de-emphasised
C_REF = "#1F77B4"        # baselines and reference lines
C_LESION = "#39FF14"

mpl.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")


def g224(m):
    return cv2.resize(np.asarray(m).astype(np.float32), (CROP_SIZE, CROP_SIZE),
                      interpolation=cv2.INTER_NEAREST)


def overlay(ax, rgb, cam, thresh=0.28, gamma=0.8, max_alpha=0.75):
    """Cold regions fully transparent so the skin stays visible."""
    u = cv2.resize(norm01(cam), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
    al = np.clip((u - thresh) / (1 - thresh), 0, 1) ** gamma * max_alpha
    ax.imshow(rgb)
    ax.imshow(u, cmap="turbo", alpha=al, vmin=0, vmax=1, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])


def overlay_signed(ax, rgb, cam, thresh=0.12):
    c = np.asarray(cam, np.float32)
    v = np.percentile(np.abs(c), 98) + 1e-8
    u = cv2.resize(c, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
    al = np.clip((np.abs(u) / v - thresh) / (1 - thresh), 0, 1) ** 0.8 * 0.85
    ax.imshow(rgb)
    ax.imshow(u, cmap="RdBu_r", alpha=al, vmin=-v, vmax=v, interpolation="bilinear")
    ax.set_xticks([])
    ax.set_yticks([])


def outline(ax, mask, color, lw=1.2, ls="solid"):
    ax.contour(g224(mask), [.5], colors=[color], linewidths=lw, linestyles=ls)


def outline_groups(ax, groups, lw=1.4, ls="dashed"):
    for g, m in sorted(groups.items()):
        outline(ax, m, GROUP_COLOR.get(g, "#000"), lw, ls)


def box_compare(ax, data, focus=(), control=(), ref=None, ref_label=None, ylabel=""):
    """data: ordered dict label -> values. Focus red, control dark grey, rest light grey."""
    labs = list(data)
    bp = ax.boxplot([np.asarray(data[k], float)[~np.isnan(data[k])] for k in labs],
                    widths=.6, patch_artist=True, medianprops={"color": "k"})
    for b, k in zip(bp["boxes"], labs):
        b.set_facecolor(C_FOCUS if k in focus else C_CONTROL if k in control else C_CONTEXT)
        b.set_alpha(.75 if k in focus else .55)
    if ref is not None:
        ax.axhline(ref, color=C_REF, ls=":", lw=1.2, label=ref_label)
        ax.legend(frameon=False, loc="best")
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, rotation=25, ha="right")
    ax.set_ylabel(ylabel)

def draw_top(ax, rgb, hot, color="#FF2D2D", alpha=0.40):
    ax.imshow(rgb)
    ov = np.zeros((CROP_SIZE, CROP_SIZE, 4))
    ov[g224(hot) > .5] = [*mpl.colors.to_rgb(color), alpha]
    ax.imshow(ov)
    outline(ax, hot, color, lw=1.1)
    ax.set_xticks([]); ax.set_yticks([])

def draw_top_smooth(ax, rgb, cam, frac=0.10, color="#00E5FF", fill_alpha=0.12):
    u = cv2.resize(norm01(cam), (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_CUBIC)
    thr = np.quantile(u, 1 - frac)
    hot = (u >= thr).astype(float)
    ax.imshow(rgb)
    if fill_alpha > 0:
        ov = np.zeros((CROP_SIZE, CROP_SIZE, 4))
        ov[hot > .5] = [*mpl.colors.to_rgb(color), fill_alpha]
        ax.imshow(ov)
    cs = ax.contour(hot, [.5], colors=[color], linewidths=1.8)
    for c in cs.collections if hasattr(cs, "collections") else [cs]:
        c.set_path_effects([pe.Stroke(linewidth=3.6, foreground="black"), pe.Normal()])
    ax.set_xticks([]); ax.set_yticks([])