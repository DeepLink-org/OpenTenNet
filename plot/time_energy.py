import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

width = 9.0
height = width * 3 / 4
fig, ax = plt.subplots(dpi=400, figsize=(width, height))
fontsize_ = 14
# Sycamore

rect = Rectangle((0, 0), width=600, height=4.3)
ax.add_patch(rect)
rect.set_facecolor("mistyrose")
# ax.text(600/2, 4.3/2, "instance better than sycamore", ha='center', va='center')


ax.scatter(
    600,
    4.3,
    c="brown",
    edgecolors="brown",
    marker="s",
    label="Sycamore,3M uncorrelated samples (Arute et al., 2019)",
)
ax.scatter(
    200,
    4.3 / 3,
    c="olive",
    edgecolors="olive",
    marker="s",
    label="Sycamore,1M uncorrelated samples (Arute et al., 2019)",
)

# sunway
ax.scatter(
    304,
    2955,
    c="none",
    edgecolors="orangered",
    marker="o",
    label="Sunway supercomputer, 1M correlated samples (Liu et al., 2021)",
)

# others
ax.scatter(
    5 * 24 * 3600,
    2520,
    c="none",
    edgecolors="indigo",
    marker="o",
    label="60 GPUs,1M correlated samples (Pan et al., 2022)",
)

# other correlated
ax.scatter(
    15 * 3600,
    2688,
    c="darkolivegreen",
    edgecolors="darkolivegreen",
    marker="o",
    label="512 GPUs, 1M uncorrelated samples (Pan et al., 2022)",
)

# leapfrogging
# ax.scatter(86.4, 13.7, c  ="darkorange", edgecolors = "darkorange", marker = "o",  label='1432 GPUs, 3M uncorrelated samples (Zhao et al., unpublished)')


# ourwork
ax.scatter(
    133.152,
    1.13,
    c="red",
    edgecolors="red",
    marker="o",
    label="96 GPUs, 3M uncorrelated samples (ours, 4T with post selection)",
)
ax.scatter(
    17.18,
    0.29,
    c="blue",
    edgecolors="blue",
    marker="o",
    label="256 GPUs, 3M uncorrelated samples (ours, 32T with post selection)",
)
ax.scatter(
    14.22,
    2.39,
    c="deepskyblue",
    edgecolors="deepskyblue",
    marker="o",
    label="2304 GPUs, 3M uncorrelated samples (ours, 32T no post selection)",
)

ax.scatter(
    514,
    2.55,
    c="red",
    edgecolors="red",
    marker="x",
    label="128 MX C500, 3M uncorrelated samples (ours, 4T post selection)",
)


# ax.set_ylim([0, 3000])
ax.set_xlabel("Time-to-solution (seconds)", fontsize=fontsize_)
ax.set_ylabel("Energy consumption (kWh)", fontsize=fontsize_)
ax.legend(loc=(0.018, 0.46), fontsize=fontsize_ - 1)
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlim([0, 10**6])
plt.yticks(fontsize=fontsize_)
plt.xticks(fontsize=fontsize_)
plt.savefig("time_energy.jpg")
