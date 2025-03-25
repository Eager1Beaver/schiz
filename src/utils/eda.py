import os
import torch
import pandas as pd
from logger import logger
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any
from utils.preprocess_validation import calculate_metrics


class SchizophreniaEDA:
    """Performs exploratory data analysis (EDA) on the clinical data and calculates image metrics"""

    def __init__(self,
                 path_to_clinical_data: str = "data/clinical_data.csv",
                 path_to_raw_images: Optional[str] = "data/raw_pt",
                 output_path: str = "data/eda") -> None:
        """
        Initializes the SchizophreniaEDA class with paths to clinical and raw imaging data.

        Parameters:
            path_to_clinical_data (str): Path to the CSV file containing clinical data.
            path_to_raw_images (Optional[str]): Directory containing raw MRI images in `.pt` format.
            
        Returns:
            None.
        """
        self.df: pd.DataFrame = pd.read_csv(path_to_clinical_data)
        self.schiz_df: pd.DataFrame = self._add_schiz_column()
        self._create_age_groups()
        self.path_to_raw_images: Optional[str] = path_to_raw_images
        self.output_path = output_path

    def _add_schiz_column(self) -> pd.DataFrame:
        """
        Adds a binary schizophrenia column to the clinical dataset.

        Returns:
            pd.DataFrame: Dataframe with an additional `schiz` column.
        """
        df = self.df.copy()
        df["schiz"] = df["dx"].apply(
            lambda x: "Schizophrenia"
            if "Schizophrenia" in x else "No Schizophrenia")
        return df

    def _create_age_groups(self) -> None:
        """categorical age groups for better visualization"""
        bins = [0, 20, 30, 40, 50, 60, float('inf')]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69"]
        self.schiz_df["age_group"] = pd.cut(self.schiz_df["age"],
                                            bins=bins,
                                            labels=labels,
                                            right=False)

    def plot_age_distribution(self, save_path='schiz_age.png') -> None:
        """
        Plots the age group distribution by schizophrenia diagnosis.

        Parameters:
            save_path (str): File path to save the plot.
            
        Returns:
            None.
        """
        age_group_dx_bin = self.schiz_df.groupby(
            ["age_group", "schiz"],
            observed=False).size().unstack(fill_value=0)
        fig = age_group_dx_bin.plot(kind='bar', stacked=False, figsize=(10, 6))
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Diagnosis', bbox_to_anchor=[1, 0.9])
        plt.tight_layout()
        plt.show()
        fig.get_figure().savefig(self.output_path + "/" + save_path)

    def plot_gender_distribution(self, save_path='schiz_gen.png') -> None:
        """
        Plots the schizophrenia diagnosis distribution by gender.

        Parameters:
            save_path (str): File path to save the plot.
            
        Returns:
            None.
        """
        schiz_bin_gender = self.schiz_df.groupby(
            ["sex", "schiz"], observed=False).size().unstack(fill_value=0)
        fig = schiz_bin_gender.plot(kind='bar',
                                    stacked=False,
                                    colormap="Paired",
                                    figsize=(10, 6))
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.legend(title='Diagnosis', bbox_to_anchor=[1, 0.9])
        plt.tight_layout()
        plt.show()
        fig.get_figure().savefig(self.output_path + "/ " +save_path)

    def get_age_statistics(self) -> Dict[str, pd.Series]:
        """
        Computes age statistics for schizophrenia and non-schizophrenia groups.

        Returns:
            Dict[str, pd.Series]: Dictionary with age statistics for each group.
        """
        df_schiz = self.schiz_df[self.schiz_df["schiz"] == "Schizophrenia"]
        df_ns = self.schiz_df[self.schiz_df["schiz"] == "No Schizophrenia"]
        return {
            "Schizophrenia": df_schiz["age"].describe(),
            "No Schizophrenia": df_ns["age"].describe()
        }

    def save_age_statistics(self, filename: str = "age_statistics.csv"):
        """
        Computes age statistics and saves them to a CSV file.

        Args:
            filename (str): The name of the CSV file to save the statistics.
        """
        age_stats = self.get_age_statistics()

        # Convert dictionary to DataFrame
        df_stats = pd.DataFrame(age_stats)

        # Transpose so that the statistics are rows and groups are columns
        df_stats.to_csv(self.output_path + "/" + filename, index=True)

        print(f"Statistics saved to {filename}")

    def assess_raw_images(self) -> List[Dict[str, Any]]:
        """
        Assesses the quality of raw MRI images using predefined metrics.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing metrics for each image.
        """
        if not self.path_to_raw_images or not os.path.exists(
                self.path_to_raw_images):
            raise FileNotFoundError(
                f"Path {self.path_to_raw_images} does not exist.")

        raw_metrics_all_images = []
        for file_name in os.listdir(self.path_to_raw_images):
            if not file_name.endswith(".pt"):
                continue  # Skip non-PT files

            file_path = os.path.join(self.path_to_raw_images, file_name)
            try:
                loaded_tensor = torch.load(file_path)
                tensor_numpy = loaded_tensor.numpy()
                raw_metrics = calculate_metrics(None, tensor_numpy)
                raw_metrics["file_path"] = file_path
                raw_metrics_all_images.append(raw_metrics)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        return raw_metrics_all_images

    def save_raw_images_metrics(self,
                                save_path: str = "raw_image_metrics.csv"
                                ) -> None:
        """
        Saves computed MRI image quality metrics to a CSV file.

        Parameters:
            save_path (str): File path to save the CSV file.
            
        Returns:
            None.
        """
        raw_metrics_all_images = self.assess_raw_images()
        if raw_metrics_all_images:
            df_raw_metrics = pd.DataFrame(raw_metrics_all_images)
            save_path = self.output_path + "/" + save_path
            df_raw_metrics.to_csv(save_path, index=False)
            logger.info(f"Raw image metrics saved to {save_path}")
        else:
            logger.info("No metrics computed. Check the input directory.")
