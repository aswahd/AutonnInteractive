"""Render measured 3D results and paper figures; no image preprocessing or inference."""

import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from skimage.measure import marching_cubes

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
report = json.loads((args.data / "report.json").read_text())
query = np.load(args.data / "query.npy", mmap_mode="r")[0]
targets = report["targets"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
)
colors = {
    "native_points": "#5974b4",
    "native_lasso": "#34495e",
    "native_mask": "#bd77a1",
    "baseline": "#da9227",
    "kd": "#8c66b3",
    "backprop": "#168b93",
    "coarse": "#868d52",
    "baseline_autozoom": "#da9227",
    "kd_autozoom": "#8c66b3",
    "backprop_autozoom": "#168b93",
}
names = {
    "native_points": "nnInteractive: point",
    "native_lasso": "nnInteractive: lasso",
    "native_mask": "nnInteractive: mask",
    "baseline_autozoom": "Baseline + AutoZoom",
    "kd_autozoom": "KD + AutoZoom",
    "backprop_autozoom": "Backprop + AutoZoom",
    "baseline": "Baseline",
    "kd": "KD",
    "backprop": "Backprop",
    "coarse": "Coarse proto./memory",
}


def save(fig, name):
    fig.savefig(args.output / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(args.output / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def mask(target, method, count=4):
    return np.load(args.data / f"{target}_{method}_k{count}.npy", mmap_mode="r")


def score(target, method, count=4):
    return next(
        r["dice"]
        for r in report["records"]
        if r["target"] == target and r["method"] == method and r["supports"] == count
    )


def scene(image, segmentation=None, color="#168b93", cut=None):
    """Native-index CT cut planes and an unsmoothed binary isosurface."""
    fig = plt.figure(figsize=(3, 3), dpi=180)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    depth, height, width = image.shape
    if cut is None:
        cut = np.array(image.shape) // 2
    low, high = np.percentile(image[::3, ::3, ::3], [1, 99])
    norm = Normalize(low, high)
    # Display texture subsampling and mesh tessellation affect rendering only.
    yy, xx = np.meshgrid(np.arange(0, height, 4), np.arange(0, width, 4), indexing="ij")
    plane = image[int(cut[0]), ::4, ::4]
    ax.plot_surface(
        xx,
        yy,
        np.full_like(xx, cut[0]),
        facecolors=plt.cm.gray(norm(plane)),
        rstride=1,
        cstride=1,
        shade=False,
        alpha=0.65,
        rasterized=True,
        zorder=1,
        linewidth=0,
        antialiased=False,
    )
    zz, xx = np.meshgrid(np.arange(0, depth, 3), np.arange(0, width, 4), indexing="ij")
    plane = image[::3, int(cut[1]), ::4]
    ax.plot_surface(
        xx,
        np.full_like(xx, cut[1]),
        zz,
        facecolors=plt.cm.gray(norm(plane)),
        rstride=1,
        cstride=1,
        shade=False,
        alpha=0.28,
        rasterized=True,
        zorder=1,
        linewidth=0,
        antialiased=False,
    )
    if segmentation is not None and np.any(segmentation):
        vertices, faces, _, _ = marching_cubes(np.pad(np.asarray(segmentation, dtype=np.uint8), 1), 0.5, step_size=2)
        vertices -= 1
        poly = Poly3DCollection(vertices[faces][:, :, [2, 1, 0]], facecolors=color, alpha=0.96, shade=True, zorder=5)
        poly.set_edgecolor("none")
        poly.set_rasterized(True)
        ax.add_collection3d(poly)
    ax.set(xlim=(0, width), ylim=(0, height), zlim=(0, depth))
    ax.set_box_aspect((width, height, depth))
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=180, transparent=False)
    plt.close(fig)
    stream.seek(0)
    return np.asarray(Image.open(stream).convert("RGB"))


target = targets[0]
support = np.load(args.data / f"{target}_supports.npz")
truth = np.load(args.data / f"{target}_truth.npy")
cut = np.median(np.argwhere(truth), axis=0).astype(int)  # Display only, after inference.
support_cut = np.median(np.argwhere(support["masks"][0, 0]), axis=0).astype(int)
panels = [
    scene(support["images"][0, 0], support["masks"][0, 0], "#2d85af", support_cut),
    scene(query, cut=cut),
    scene(query, mask(target, "backprop"), "#d68b22", cut),
    scene(query, mask(target, "backprop_autozoom"), "#168b93", cut),
]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.15))
titles = ["Prompted\nsupport", "Unprompted\nquery", "Automatic ICL\nprediction", "Native AutoZoom\nrefinement"]
for i, (ax, tile, title) in enumerate(zip(axes, panels, titles)):
    ax.imshow(tile)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.axis("off")
    if i >= 2:
        key = "backprop" if i == 2 else "backprop_autozoom"
        ax.text(0.5, -0.01, f"Dice {score(target, key):.3f}", transform=ax.transAxes, ha="center")
fig.subplots_adjust(wspace=0.01, top=0.78, bottom=0.12)
fig.text(
    0.5, 0.01, "Prompted supports + query  →  shared ICL model  →  frozen native refinement", ha="center", fontsize=8
)
save(fig, "workflow")
fig, axes = plt.subplots(1, 4, figsize=(10.6, 3.0))
for k, ax in enumerate(axes):
    cue = support["masks"][k, 0]
    center = np.median(np.argwhere(cue), axis=0).astype(int)
    ax.imshow(scene(support["images"][k, 0], cue, "#2d85af", center))
    ax.set_title(f"Support {k+1}: dense local mask")
    ax.axis("off")
save(fig, "support_bank")

# The native stage grids are displayed without interpolating CT inputs.
attention = np.load(args.data / "attention.npz")
selected = attention["selected"]
z = int(selected[0])
fig, axes = plt.subplots(2, 4, figsize=(7.0, 2.9))
image = attention["query"]
low, high = np.percentile(image, [1, 99])
axes[0, 0].imshow(image[z], cmap="gray", vmin=low, vmax=high)
axes[0, 0].plot(selected[2], selected[1], "x", color="#f9a828", ms=8)
axes[0, 0].set_title("Query / selected voxel", fontsize=8)
masses = [attention[f"stage{s}_foreground_mass"] for s in [3, 4, 5]]
vmax = max(float(x.max()) for x in masses)
for col, (stage, mass) in enumerate(zip([3, 4, 5], masses), 1):
    axes[0, col].imshow(image[z], cmap="gray", vmin=low, vmax=high)
    plane = mass[min(mass.shape[0] - 1, z * mass.shape[0] // 192)]
    im = axes[0, col].imshow(
        plane, extent=(-0.5, 191.5, 191.5, -0.5), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest", alpha=0.75
    )
    axes[0, col].set_title(f"Stage {stage-6}: FG mass", fontsize=8)
weights = attention["stage5_selected_weights"]
for k in range(4):
    cue = support["masks"][k, 0]
    sz = int(cue.sum((1, 2)).argmax())
    img = support["images"][k, 0, sz]
    lo, hi = np.percentile(img, [1, 99])
    axes[1, k].imshow(img, cmap="gray", vmin=lo, vmax=hi)
    axes[1, k].imshow(
        weights[k, min(5, sz * 6 // 192)],
        extent=(-0.5, 191.5, 191.5, -0.5),
        cmap="magma",
        vmin=0,
        vmax=float(weights.max()),
        interpolation="nearest",
        alpha=0.65,
    )
    axes[1, k].contour(cue[sz], levels=[0.5], colors=["#2bd4db"], linewidths=0.6)
    axes[1, k].set_title(f"Support {k+1}; mass {weights[k].sum():.2f}", fontsize=8)
for ax in axes.flat:
    ax.axis("off")
fig.subplots_adjust(wspace=0.03, hspace=0.22, bottom=0.19, top=0.92)
for bounds, maximum, label in [
    ([0.08, 0.10, 0.35, 0.018], vmax, "Foreground attention mass"),
    ([0.57, 0.10, 0.35, 0.018], float(weights.max()), "Selected-voxel attention weight"),
]:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(Normalize(0, maximum), cmap="magma"), cax=fig.add_axes(bounds), orientation="horizontal"
    )
    cbar.ax.tick_params(labelsize=6, pad=1)
    cbar.set_label(label, fontsize=7, labelpad=1)
save(fig, "attention_selected")

methods = ["truth", "native_lasso", "coarse", "baseline", "kd", "backprop", "backprop_autozoom"]
fig, axes = plt.subplots(2, len(methods), figsize=(7.0, 3.0))
for row, target in enumerate(targets):
    reference = np.load(args.data / f"{target}_truth.npy")
    center = np.median(np.argwhere(reference), axis=0).astype(int)
    for col, method in enumerate(methods):
        output = reference if method == "truth" else mask(target, method, 0 if method.startswith("native_") else 4)
        axes[row, col].imshow(
            scene(query, output, "#e0ac30" if method == "truth" else colors.get(method, "#168b93"), center)
        )
        axes[row, col].axis("off")
        if row == 0:
            title = (
                "Reference"
                if method == "truth"
                else names[method]
                .replace(" + ", "\n+ ")
                .replace(": ", "\n")
                .replace("Coarse proto./memory", "Coarse proto./\nmemory")
            )
            axes[row, col].set_title(title, fontsize=7)
        if method != "truth":
            axes[row, col].text(
                0.5,
                0.01,
                f"Dice {score(target, method, 0 if method.startswith('native_') else 4):.3f}",
                ha="center",
                transform=axes[row, col].transAxes,
                fontsize=7,
            )
    axes[row, 0].text(
        -0.04,
        0.5,
        target.replace("_", " ").title(),
        rotation=90,
        va="center",
        transform=axes[row, 0].transAxes,
        fontsize=8,
    )
fig.subplots_adjust(wspace=0.01, hspace=0.02, top=0.86, bottom=0.03)
save(fig, "qualitative")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for method in [
    "native_points",
    "native_lasso",
    "native_mask",
    "baseline_autozoom",
    "kd_autozoom",
    "backprop_autozoom",
    "coarse",
]:
    curves = [r["dice"] for r in report["correction_curves"] if r["method"] == method]
    axes[0].plot(range(6), np.mean(curves, axis=0) * 100, "o-", color=colors[method], ms=3, lw=1.5, label=names[method])
for method in ["baseline", "kd", "backprop"]:
    for refined in [False, True]:
        key = method + "_autozoom" if refined else method
        values = [np.mean([score(t, key, count) for t in targets]) * 100 for count in [1, 2, 3, 4]]
        axes[1].plot(
            [1, 2, 3, 4],
            values,
            "o-" if refined else "s--",
            color=colors[method],
            ms=3,
            label=names[method] + (" + AutoZoom" if refined else ""),
        )
axes[0].set(
    xlabel="Correction rounds after initialization",
    ylabel="Mean Dice (%)",
    title="(a) Simulated interactive correction",
    xticks=range(6),
)
axes[1].set(
    xlabel="Number of support images",
    ylabel="Mean Dice (%)",
    title="(b) Fixed nested support banks",
    xticks=[1, 2, 3, 4],
)
for ax in axes:
    ax.grid(alpha=0.18)
    ax.set_ylim(0, 101)
    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.93)
fig.tight_layout()
save(fig, "curves")

# Preserve exact measured values for reproducibility and manuscript macros.
macros = [r"\newcommand{\BackpropUpdate}{" + str(report["checkpoints"]["backprop"]["update"]) + "}"]
for target, prefix in zip(targets, ["Kidney", "Spleen"]):
    for method, suffix in [("backprop", "Initial"), ("backprop_autozoom", "Refined")]:
        macros.append("\\newcommand{\\" + prefix + suffix + "}{" + f"{score(target, method):.3f}" + "}")
(args.output / "measured.tex").write_text("\n".join(macros) + "\n")
(args.output / "provenance.json").write_text(
    json.dumps(
        {
            **report,
            "rendering": "Real CT cut planes and binary-mask isosurfaces, no mesh smoothing. Rendering texture stride 3–4 and marching-cubes step 2 only; model arrays unchanged. Stored voxel-index geometry; no anatomical orientation or physical-spacing claim. Reference masks select display planes after inference only.",
        },
        indent=2,
    )
    + "\n"
)
print("FIGURES_RENDERED " + str(args.output), flush=True)
