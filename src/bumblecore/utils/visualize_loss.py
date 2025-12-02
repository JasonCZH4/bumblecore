import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def smooth(scalars: list[float]) -> list[float]:
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

class LossPlotter:
    
    @staticmethod
    def plot_loss(
        x_values, 
        y_values,
        xlabel,
        ylabel,
        save_dir,
        filename,
        smooth_plot
    ):
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)

        plt.figure()

        if smooth_plot:
            plt.plot(x_values, y_values, color="#0095ff", alpha=0.4, label='Original')
            smoothed_losses = smooth(y_values.tolist())
            plt.plot(x_values, smoothed_losses, color='#0095ff', label='Smoothed')
            plt.legend()
        else:
            plt.plot(x_values, y_values, color="#0095ff", label='Original')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        full_filename = os.path.join(save_dir, filename)
        plt.savefig(full_filename, dpi=300, bbox_inches='tight')
        plt.close()