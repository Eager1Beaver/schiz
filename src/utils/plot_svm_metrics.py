import os
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


class SVMVisualizer:

    def __init__(self, results_json_path, save_figs_path: str = "paper/figs"):
        """
        Initializes the visualization class and loads classification results.

        Parameters:
            results_json_path (str): Path to the JSON file containing classification results.
        """
        self.results_json_path = results_json_path
        self.results = self._load_results()
        self.save_figs_path = save_figs_path

    def _load_results(self):
        """Loads classification results from the JSON file."""
        if not os.path.exists(self.results_json_path):
            raise FileNotFoundError(
                f"Results file not found: {self.results_json_path}")

        with open(self.results_json_path, "r") as f:
            return json.load(f)

    def plot_confusion_matrix(self):
        """Plots the confusion matrix."""
        y_test = np.array(self.results["y_test"])
        test_preds = np.array(self.results["test_preds"])
        conf_matrix = np.array(self.results["confusion_matrix"])

        # Get unique class names if labels are more than just 0 and 1
        #unique_classes = sorted(set(y_test))

        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=unique_classes,
                    yticklabels=unique_classes)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        fig.savefig(self.save_figs_path + "cm" + ".png")
        #plt.show()

    def plot_roc_curve(self):
        """Plots the ROC curve."""
        y_test = np.array(self.results["y_test"])
        test_scores = np.array(
            self.results["test_scores"])  # Scores should be probabilities

        # Handle binary classification vs. multi-class
        if test_scores.ndim == 1:
            fpr, tpr, _ = roc_curve(y_test, test_scores)
        else:
            fpr, tpr, _ = roc_curve(
                y_test, test_scores[:, 1])  # Only take positive class scores

        roc_auc = auc(fpr, tpr)

        fig = plt.figure(figsize=(6, 5))
        plt.plot(fpr,
                 tpr,
                 color='blue',
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        fig.savefig(self.save_figs_path + "roc" + ".png")


def metrics_comparison_plot(df, col, save_path, color) -> None:
    """Creates horizontal bar plot of metrics for all the experiments
    
    Parameters:
        df (DataFrame): Dataframe containing the metrics
        col (str): Column containing metric of interest
        save_path (str): File path to save the plot.
        color (str): Seaborn colour palette to use for plot
    Returns:
        None
    """
    # Sort dataframe in descending order
    df_sorted = df.sort_values(by=col, ascending=False)
    # Plot the results
    fig = sns.barplot(x=col,
                      y='Preprocessing/ Augmentation Technique',
                      data=df_sorted,
                      color=color,
                      orient='h')

    #add plot title
    plt.title(f'{col} Comparison Plot', fontsize=13)
    #add axis labels
    plt.xlabel('Preprocessing and Augmentation Experiments')
    plt.ylabel(f'{col}')
    fig.get_figure().savefig(save_path)
