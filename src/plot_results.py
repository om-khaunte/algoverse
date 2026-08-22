import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_distance_vs_auroc(registry_path="results/registry.csv", distance_path="data/phylo_distance.csv"):
    """
    Reads experiment results and phylogenetic distances,
    plots zero-shot AUROC vs phylogenetic distance.
    STUB: uses placeholder data - fill in with real merge/plot logic
    once registry.csv and phylo_distance.csv are both finalized.
    """
    os.makedirs("figures", exist_ok=True)

    fake_distances = [0.1, 0.3, 0.5, 0.7, 0.9]
    fake_aurocs = [0.85, 0.78, 0.70, 0.65, 0.55]

    plt.figure(figsize=(6, 5))
    plt.scatter(fake_distances, fake_aurocs)
    plt.xlabel("Phylogenetic Distance")
    plt.ylabel("Zero-shot AUROC")
    plt.title("Transfer Performance vs. Phylogenetic Distance (STUB - fake data)")
    plt.savefig("figures/distance_vs_auroc_stub.png")
    plt.show()


if __name__ == "__main__":
    plot_distance_vs_auroc()
