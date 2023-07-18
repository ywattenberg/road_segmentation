import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def plot_test_scores(csv_dir="smp"):
    fig = plt.figure()
    for file in sorted(os.listdir(csv_dir)):
        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(csv_dir, file))

        plt.plot(df["epoch"], df["DiceLoss"], label=file)
    plt.legend()
    fig.savefig("smp/test_scores.png")


if __name__ == "__main__":
    plot_test_scores()
