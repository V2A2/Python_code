import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Publication style for Springer LNCS / TACAS
# ============================================================

plt.rcParams.update({
    # Font
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,

    # Axes
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,

    # Tick labels
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

    # Legend
    "legend.fontsize": 8,

    # Vector PDF font embedding
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Saving
    "savefig.bbox": "tight",
})


# ============================================================
# Data
# ============================================================

networks = [
    "2-2",
    "3-5",
    "4-5",
    "5-1",
    "5-4",
    "6-5"
]

# Verification error (%)
# |p_PS - p_true|
ps_error = np.array([
    0.790,
    0.587,
    1.265,
    0.057,
    0.071,
    0.110
])

# |p_GPS - p_true|
gps_error = np.array([
    0.014,
    0.042,
    0.149,
    0.012,
    0.002,
    0.058
])


# ============================================================
# Positions
# ============================================================

x = np.arange(len(networks))

# Small separation between ProbStar and Generalized ProbStar
offset = 0.13

x_ps = x - offset
x_gps = x + offset

# Lower end of stems.
# Must be positive because the y-axis is logarithmic.
baseline = 0.001


# ============================================================
# Figure
# ============================================================

# Designed for approximately one-column placement in LNCS.
fig, ax = plt.subplots(figsize=(4.7, 3.0))


# ============================================================
# Stems
# ============================================================

ax.vlines(
    x_ps,
    baseline,
    ps_error,
    linewidth=1.1,
    alpha=0.75,
    zorder=1
)

ax.vlines(
    x_gps,
    baseline,
    gps_error,
    linewidth=1.1,
    alpha=0.75,
    zorder=1
)


# ============================================================
# Markers
# ============================================================

# ProbStar: circle
ax.scatter(
    x_ps,
    ps_error,
    s=42,
    marker="o",
    label="ProbStar",
    zorder=3
)

# Generalized ProbStar: square
ax.scatter(
    x_gps,
    gps_error,
    s=42,
    marker="s",
    label="Generalized ProbStar",
    zorder=3
)


# ============================================================
# Axes
# ============================================================

ax.set_yscale("log")

ax.set_xlim(-0.55, len(networks) - 0.45)
ax.set_ylim(0.001, 2.0)

ax.set_xticks(x)
ax.set_xticklabels(networks)

ax.set_xlabel("ACAS Xu Network")
ax.set_ylabel("Verification Error (%)")


# ============================================================
# Grid
# ============================================================

# Major log-scale grid only.
# Keeping only major grid lines makes the figure cleaner.
ax.grid(
    axis="y",
    which="major",
    linestyle="--",
    linewidth=0.55,
    alpha=0.45,
    zorder=0
)


# ============================================================
# Numerical annotations
# ============================================================

# ProbStar values
for i, (xp, value) in enumerate(zip(x_ps, ps_error)):

    if i == 0:
        # 0.790: slightly right of the marker center
        xytext = (3, 5)
        ha = "center"
    else:
        xytext = (-3, 5)
        ha = "right"

    ax.annotate(
        f"{value:.3f}",
        xy=(xp, value),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va="bottom",
        fontsize=8
    )

# Generalized ProbStar values
for xg, value in zip(x_gps, gps_error):
    ax.annotate(
        f"{value:.3f}",
        xy=(xg, value),
        xytext=(3, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8
    )


# ============================================================
# Legend
# ============================================================

ax.legend(
    loc="upper right",
    frameon=False,
    handletextpad=0.5,
    borderaxespad=0.3
)


# ============================================================
# Clean publication appearance
# ============================================================

# Remove unnecessary frame lines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Keep ticks simple
ax.tick_params(
    axis="both",
    which="major",
    width=0.8,
    length=3
)

# Minor ticks are useful for reading a logarithmic axis,
# but do not draw minor grid lines.
ax.tick_params(
    axis="y",
    which="minor",
    width=0.5,
    length=2
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout(pad=0.4)


# ============================================================
# Save
# ============================================================

# Vector version for the paper
fig.savefig(
    "p2_verification_error.pdf",
    format="pdf",
    bbox_inches="tight"
)

# High-resolution preview
fig.savefig(
    "p2_verification_error.png",
    format="png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()