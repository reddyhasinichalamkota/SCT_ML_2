import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load real data
df = pd.read_csv("Mall_Customers.csv")
df.rename(columns={
    "Annual Income (k$)": "income",
    "Spending Score (1-100)": "score",
    "Age": "age",
    "Gender": "gender",
    "CustomerID": "id"
}, inplace=True)

print(df.describe())

#Scale features
X = df[["age", "income", "score"]].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Elbow + Silhouette
inertias, silhouettes = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

#Final model k=5 
K = 5
km_final = KMeans(n_clusters=K, init="k-means++", n_init=20, random_state=42)
df["cluster"] = km_final.fit_predict(X_scaled)
centers_orig = scaler.inverse_transform(km_final.cluster_centers_)

COLORS = ["#3266ad", "#e05b2b", "#2a9d69", "#c84b8e", "#d4a520"]
NAMES  = ["Budget shoppers", "Premium buyers", "Young impulsive",
          "Conservative earners", "Mid-tier balanced"]

profile = df.groupby("cluster")[["age", "income", "score"]].mean().round(1)
profile["size"] = df["cluster"].value_counts().sort_index()

print("\nCluster Profiles:\n", profile)

# FIGURE
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#f4f6fa")

fig.suptitle(
    "K-Means Customer Segmentation  |  Mall Dataset  |  k = 5",
    fontsize=18, fontweight="bold", y=0.995, color="#1a1a2e"
)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.55,
    wspace=0.38,
    left=0.07, right=0.97,
    top=0.97, bottom=0.12
)

#Elbow | Silhouette | Heatmap
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])

# Elbow curve
ax0.plot(K_range, inertias, "o-", color="#3266ad", lw=2.5, ms=7)
ax0.axvline(5, ls="--", color="#e05b2b", alpha=0.7, lw=1.5, label="k = 5")
ax0.fill_between(K_range, inertias, alpha=0.12, color="#3266ad")
ax0.set(title="Elbow Curve", xlabel="Number of Clusters (k)", ylabel="Inertia")
ax0.legend(fontsize=8)
ax0.grid(alpha=0.3, linestyle="--")
ax0.set_facecolor("white")

# Silhouette score
ax1.plot(K_range, silhouettes, "o-", color="#e05b2b", lw=2.5, ms=7)
ax1.axvline(5, ls="--", color="#3266ad", alpha=0.7, lw=1.5, label="k = 5")
ax1.fill_between(K_range, silhouettes, alpha=0.12, color="#e05b2b")
ax1.set(title="Silhouette Score", xlabel="Number of Clusters (k)", ylabel="Score")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3, linestyle="--")
ax1.set_facecolor("white")

# Feature heatmap — short x-tick labels to avoid overlap
xtick_labels = [f"C{i+1}\n(n={int(profile['size'].iloc[i])})" for i in range(K)]
heat_data = profile[["age", "income", "score"]].T
sns.heatmap(
    heat_data,
    annot=True, fmt=".1f",
    cmap="YlOrRd",
    xticklabels=xtick_labels,
    yticklabels=["Age", "Income (k$)", "Score"],
    linewidths=0.6,
    ax=ax2,
    cbar_kws={"shrink": 0.75, "pad": 0.02}
)
ax2.set_title("Feature Heatmap by Cluster", fontsize=11, fontweight="bold")
ax2.tick_params(axis="x", labelsize=8, rotation=0)
ax2.tick_params(axis="y", labelsize=8, rotation=0)

#Income vs Score | Age vs Income | Age vs Score
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])

for c in range(K):
    sub = df[df["cluster"] == c]
    ax3.scatter(sub["income"], sub["score"], color=COLORS[c],
                label=f"C{c+1}: {NAMES[c]}", alpha=0.75,
                edgecolors="white", linewidths=0.5, s=80)
    ax4.scatter(sub["age"], sub["income"], color=COLORS[c],
                alpha=0.75, edgecolors="white", linewidths=0.5, s=70)
    ax5.scatter(sub["age"], sub["score"], color=COLORS[c],
                alpha=0.75, edgecolors="white", linewidths=0.5, s=70)

# Centroids on income vs score plot
ax3.scatter(centers_orig[:, 1], centers_orig[:, 2],
            c="black", marker="X", s=230, zorder=5,
            label="Centroids", edgecolors="white", linewidths=0.8)
ax3.set(title="Income vs Spending Score",
        xlabel="Annual Income (k$)", ylabel="Spending Score")
ax3.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
ax3.grid(alpha=0.25, linestyle="--")
ax3.set_facecolor("white")

ax4.set(title="Age vs Annual Income",
        xlabel="Age (years)", ylabel="Annual Income (k$)")
ax4.grid(alpha=0.25, linestyle="--")
ax4.set_facecolor("white")

ax5.set(title="Age vs Spending Score",
        xlabel="Age (years)", ylabel="Spending Score")
ax5.grid(alpha=0.25, linestyle="--")
ax5.set_facecolor("white")

#Avg Age | Avg Income | Avg Score (bar charts)
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

bar_specs = [
    (ax7, "age",    "Average Age per Cluster",            "Years"),
    (ax8, "income", "Average Annual Income per Cluster",  "k$"),
    (ax9, "score",  "Average Spending Score per Cluster", "/ 100"),
]

for ax_b, feat, title, unit in bar_specs:
    vals  = [profile[feat].iloc[c] for c in range(K)]
    xlabs = [f"C{i+1}" for i in range(K)]
    bars  = ax_b.bar(xlabs, vals, color=COLORS,
                     edgecolor="white", linewidth=0.8, width=0.6)
    ax_b.set(title=title, ylabel=unit)
    ax_b.set_ylim(0, max(vals) * 1.18)
    ax_b.grid(axis="y", alpha=0.3, linestyle="--")
    ax_b.set_facecolor("white")
    for bar, v in zip(bars, vals):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(vals) * 0.015,
            f"{v:.0f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#222"
        )

#Cluster legend at bottom
handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS[c], markersize=11,
               label=f"C{c+1}: {NAMES[c]}")
    for c in range(K)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=5,
    fontsize=9.5,
    framealpha=0.95,
    edgecolor="#ccc",
    bbox_to_anchor=(0.5, 0.01),
    handletextpad=0.4,
    columnspacing=1.2
)

plt.savefig("mall_kmeans_complete.png", dpi=150,
            bbox_inches="tight", facecolor="#f4f6fa")
plt.show()

#Export CSV 
df["segment"] = df["cluster"].map(dict(enumerate(NAMES)))
df.to_csv("customers_clustered.csv", index=False)
print("\nSaved → mall_kmeans_complete.png  &  customers_clustered.csv")