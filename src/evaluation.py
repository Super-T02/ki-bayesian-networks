from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class Evaluation():
    
    MATCH_LESS_EQUAL = 0
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true
        self.y_pred = y_pred
        
        self.accuracy = None
        self.mean_absolute_error = None
        self.tp_fp_tn_fn = None
        self.calc_stats()       
               
    def calc_stats(self):
        """Calculates the accuracy, mean absolute error and confusion matrix for a model."""
        self.accuracy = accuracy_score(self.y_true, self.y_pred)
        self.mean_false_error = self.mean_error_false(self.y_true, self.y_pred)
        self.mean_absolute_error = mean_absolute_error(self.y_true, self.y_pred)
        self.tp_fp_tn_fn = self.calc_tp_fp_tn_fn(self.y_true, self.y_pred)
        self.f1_score = f1_score(self.y_true, self.y_pred, average="macro")
        self.precision = precision_score(self.y_true, self.y_pred, average="macro")
        self.recall = recall_score(self.y_true, self.y_pred, average="macro")
        
    def calc_tp_fp_tn_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the True Positives, False Positives, True Negatives and False Negatives for each grade.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The True Positives, False Positives, True Negatives and False Negatives for each grade.
        """
        GRADE_MAX = 20
        matrix = np.zeros((GRADE_MAX + 1, 2, 2))
        
        for i in range(0, GRADE_MAX + 1):
            # Iterate over all predictions
            for j in range(len(y_true)):
                if y_true[j] == i: # If the true label is the current grade
                    if y_true[j] == y_pred[j]: # If the prediction is correct --> TP
                        matrix[i][1][1] += 1
                    else: # If the prediction is wrong --> FN
                        matrix[i][1][0] += 1
                else: # If the true label is not the current grade
                    if i != y_pred[j]: # If the prediction is correct --> TN
                        matrix[i][0][0] += 1
                    else: # If the prediction is wrong --> FP
                        matrix[i][0][1] += 1

        return matrix
    
    def mean_error_false(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the mean error for false predictions.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            float: The mean error for false predictions.
        """
        distances = []
        for i in range(len(y_true)):
            if y_true[i] != y_pred[i]:
                distances.append(abs(y_true[i] - y_pred[i]))
        return np.mean(distances)
                
    
    def print_stats(self):
        """Prints the accuracy and mean absolute error for a model.
        Args:
            y_true (list): The true labels.
            y_pred (list): The predicted labels.
        """
        print("Accuracy:", self.accuracy)
        print("Mean false error:", self.mean_false_error)
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
        N = self.tp_fp_tn_fn.shape[0]
        x = np.array([i for i in range(0, N)])
        y_fp, y_fn, y_tn, y_tp  = [], [], [], []

        # Get the values for each grade
        for i, cm in enumerate(self.tp_fp_tn_fn):
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
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x, y_fp, color="red", label="False Positives (FP)")
        plt.plot(x, y_tp, color="green" , label="True Positives (TP)")
        plt.legend(loc="best")
        plt.xlabel("Grade")
        plt.ylabel("X Rate")
        plt.xticks(x)
        
        plt.subplot(1, 2, 2)
        plt.plot(x, y_fn, color="red", label="False Negatives (FN)")
        plt.plot(x, y_tn, color="green", label="True Negatives (TN)")
        plt.legend(loc="best")
        plt.xlabel("Grade")
        plt.ylabel("X Rate")
        plt.xticks(x)
        plt.show()
        
    def plot_confusion_matrix(self) -> None:
        """Plots the confusion matrix for a model."""
        GRADE_MAX = 20
        labels = [i for i in range(0, GRADE_MAX + 1)]
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        # create empty figure with a specified size
        fig, ax = plt.subplots(figsize=(20, 10))
        
        plt.title("Confusion Matrix")
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=True, fmt='g')
        #plt.savefig(filename)
        plt.show()