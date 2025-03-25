import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SVMClassifier:
    """
    A Support Vector Machine (SVM) classifier with built-in preprocessing (StandardScaler).

    This class encapsulates the SVM model, allowing easy training, prediction, 
    and saving/loading of trained models.

    Attributes:
        pipeline (sklearn.pipeline.Pipeline): A pipeline containing StandardScaler and SVC.
    """

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 100,
                 gamma: float = 0.0001,
                 random_state: int = 42) -> None:
        """
        Initializes the SVM classifier with specified hyperparameters.

        Parameters:
            kernel (str): Specifies the kernel type to be used in the SVM (default: 'rbf').
            C (float): Regularization parameter (default: 100).
            gamma (float): Kernel coefficient (default: 0.0001).
            random_state (int): Controls randomness (default: 42).
        
        Returns:
            None
        """
        self.pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel,
                probability=True,
                C=C,
                gamma=gamma,
                random_state=random_state))

    def train(self, train_features: np.ndarray,
              train_labels: np.ndarray) -> None:
        """
        Trains the SVM classifier on the given training data.

        Parameters:
            train_features (np.ndarray): Feature matrix of shape (num_samples, num_features).
            train_labels (np.ndarray): Target labels of shape (num_samples,).
            
        Returns:
            None
        """
        self.pipeline.fit(train_features, train_labels)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for the given test features.

        Parameters:
            test_features (np.ndarray): Feature matrix of shape (num_samples, num_features).

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.pipeline.predict(test_features)

    def predict_proba(self, test_features: np.ndarray) -> np.ndarray:
        """
        Returns probability estimates for the given test features.

        Parameters:
            test_features (np.ndarray): Feature matrix of shape (num_samples, num_features).

        Returns:
            np.ndarray: Probability estimates for each class.
        """
        return self.pipeline.predict_proba(test_features)

    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model to a file.

        Parameters:
            file_path (str): Path to save the model.
            
        Returns:
            None
        """
        joblib.dump(self.pipeline, file_path)

    @classmethod
    def load_model(cls, file_path: str) -> "SVMClassifier":
        """
        Loads a trained model from a file.

        Parameters:
            file_path (str): Path to the saved model file.

        Returns:
            SVMClassifier: A new instance of SVMClassifier with the loaded model.
        """
        instance = cls()
        instance.pipeline = joblib.load(file_path)
        return instance
