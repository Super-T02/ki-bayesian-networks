from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, multilabel_confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

class Evaluation():
    
    MATCH_LESS_EQUAL = 0
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        
        self.accuracy = None
        self.mean_absolute_error = None
        self.confusion_matrix = None
        self.calc_stats()       
               
    def calc_stats(self):
        """Calculates the accuracy, mean absolute error and confusion matrix for a model."""
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.mean_absolute_error = mean_absolute_error(self.y_true, self.y_pred)
        self.confusion_matrix = multilabel_confusion_matrix(self.y_true, self.y_pred)
        self.f1_score = f1_score(self.y_true, self.y_pred, average="macro")
        self.precision = precision_score(self.y_true, self.y_pred, average="macro")
        self.recall = recall_score(self.y_true, self.y_pred, average="macro")
    
    def print_stats(self):
        """Prints the accuracy and mean absolute error for a model.
        Args:
            y_true (list): The true labels.
            y_pred (list): The predicted labels.
        """
        print("Accuracy:", self.accuracy)
        print("Mean absolute error:", self.mean_absolute_error)
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1 score:", self.f1_score)
        
    def plot(self):
        """Plots the true and predicted labels.
        Args:
            y_true (list): The true labels.
            y_pred (list): The predicted labels.
        """
        # Set up plot
        N = self.confusion_matrix.shape[0]
        x = np.array([i for i in range(0, N)])
        y_fp, y_fn, y_tn, y_tp  = [], [], [], []

        # Get the values for each grade
        for i, cm in enumerate(self.confusion_matrix):
            rates = {
                "TP": cm[1][1],
                "TN": cm[0][0],
                "FP": cm[0][1],
                "FN": cm[1][0]
            }
            y_fp.append(rates["FP"])
            y_fn.append(rates["FN"])
            y_tp.append(rates["TP"])
            y_tn.append(rates["TN"])
                
        # Build the plot
        plt.figure(figsize=(10, 5))

        plt.plot(x, y_fp, color="red", label="False Positives (FP)")
        plt.plot(x, y_tp, color="green" , label="True Positives (TP)")
        plt.plot(x, y_fn, color="pink", label="False Negatives (FN)")
        plt.plot(x, y_tn, color="blue", label="True Negatives (TN)")
        plt.legend(loc="best")
        plt.xlabel("Grade")
        plt.ylabel("X Rate")
        plt.xticks(x)
        plt.show()