from openTSNE import TSNE
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

def visualize(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(10, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=True, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="best", bbox_to_anchor=(0.05, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)


tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)

idexp_lm3d_pred_lrs3 = np.load("infer_out/tmp_npys/lrs3_pred_all.npy")
idx = np.random.choice(np.arange(len(idexp_lm3d_pred_lrs3)), 10000)
idexp_lm3d_pred_lrs3 = idexp_lm3d_pred_lrs3[idx]

person_ds = np.load("data/binary/videos/May/trainval_dataset.npy", allow_pickle=True).tolist()
person_idexp_mean = person_ds['idexp_lm3d_mean'].reshape([1,204])
person_idexp_std = person_ds['idexp_lm3d_std'].reshape([1,204])
person_idexp_lm3d_train = np.stack([s['idexp_lm3d_normalized'].reshape([204,]) for s in person_ds['train_samples']])
person_idexp_lm3d_val = np.stack([s['idexp_lm3d_normalized'].reshape([204,]) for s in person_ds['val_samples']])

lrs3_stats = np.load('/home/yezhenhui/datasets/binary/lrs3_0702/stats.npy',allow_pickle=True).tolist()
lrs3_idexp_mean = lrs3_stats['idexp_lm3d_mean'].reshape([1,204])
lrs3_idexp_std = lrs3_stats['idexp_lm3d_std'].reshape([1,204])
person_idexp_lm3d_train = person_idexp_lm3d_train * person_idexp_std + person_idexp_mean
# person_idexp_lm3d_train = (person_idexp_lm3d_train - lrs3_idexp_mean) / lrs3_idexp_std
person_idexp_lm3d_val = person_idexp_lm3d_val * person_idexp_std + person_idexp_mean
# person_idexp_lm3d_val = (person_idexp_lm3d_val - lrs3_idexp_mean) / lrs3_idexp_std
idexp_lm3d_pred_lrs3 = idexp_lm3d_pred_lrs3 * lrs3_idexp_std + lrs3_idexp_mean


idexp_lm3d_pred_vae = np.load("infer_out/tmp_npys/pred_exp_0_vae.npy").reshape([-1,204])
idexp_lm3d_pred_postnet = np.load("infer_out/tmp_npys/pred_exp_0_postnet_hubert.npy").reshape([-1,204])
# idexp_lm3d_pred_postnet = idexp_lm3d_pred_postnet * lrs3_idexp_std + lrs3_idexp_mean

idexp_lm3d_all = np.concatenate([idexp_lm3d_pred_lrs3, person_idexp_lm3d_train,idexp_lm3d_pred_vae, idexp_lm3d_pred_postnet])
idexp_lm3d_all_emb = tsne.fit(idexp_lm3d_all) # array(float64) [B,50]==>[B, 2]
# z_p_emb = tsne.fit(z_p) # array(float64) [B,50]==>[B, 2]
y1 = ["pred_lrs3" for _ in range(len(idexp_lm3d_pred_lrs3))]
y2 = ["person_train" for _ in range(len(person_idexp_lm3d_train))]
y3 = ["vae" for _ in range(len(idexp_lm3d_pred_vae))]
y4 = ["postnet" for _ in range(len(idexp_lm3d_pred_postnet))]
visualize(idexp_lm3d_all_emb, y1+y2+y3+y4)
plt.savefig("infer_out/tmp_npys/lrs3_pred_all_0k.png")