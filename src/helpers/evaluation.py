from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class Evaluation():
    
    MATCH_LESS_EQUAL = 0
    GRADE_MAX = 17
    GRADE_MIN = 7
    
    def __init__(self, result: pd.DataFrame):
        if "G3" not in result.columns or "G3_pred" not in result.columns:
            raise ValueError("The result DataFrame must contain the columns G3 and G3_pred.")
        
        self.y_true = result["G3"].to_numpy().astype(str)
        self.y_pred = result["G3_pred"].to_numpy().astype(str)
        self.y_true_int = self.map_grades(self.y_true.copy())
        self.y_pred_int = self.map_grades(self.y_pred.copy())
        
        self.accuracy = None
        self.mean_absolute_error = None
        self.tp_fp_tn_fn = None
        self.calc_stats()       
    
    def map_grades(self, list: np.ndarray) -> np.ndarray:
        """Maps the grades to the correct values.
        <7 --> 0
        7 - 17 --> 1 - 11
        >17 --> 12

        Args:
            list (np.ndarray): The list of grades.

        Returns:
            np.ndarray: The list of mapped grades.
        """
        for i in range(len(list)):
            if list[i] == '<7':
                list[i] = 0
            elif list[i] == '>17':
                list[i] = 12
            else:
                list[i] = int(list[i]) - 6
        return list.astype(np.int64)
               
    def calc_stats(self):
        """Calculates the accuracy, mean absolute error and confusion matrix for a model."""
        self.accuracy = accuracy_score(self.y_true_int, self.y_pred_int)
        self.mean_false_error = self.mean_error_false(self.y_true_int, self.y_pred_int)
        self.mean_absolute_error = mean_absolute_error(self.y_true_int, self.y_pred_int)
        self.tp_fp_tn_fn = self.calc_tp_fp_tn_fn(self.y_true_int, self.y_pred_int)
        self.f1_score = f1_score(self.y_true_int, self.y_pred_int, average="macro")
        self.precision = precision_score(self.y_true_int, self.y_pred_int, average="macro")
        self.recall = recall_score(self.y_true_int, self.y_pred_int, average="macro")
        
    def calc_tp_fp_tn_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the True Positives, False Positives, True Negatives and False Negatives for each grade.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The True Positives, False Positives, True Negatives and False Negatives for each grade.
        """
        matrix = np.zeros((self.GRADE_MAX - self.GRADE_MIN + 3, 2, 2))
        
        for i in range(0, self.GRADE_MAX - self.GRADE_MIN + 3):
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
        x = [f"<{self.GRADE_MIN}"] + [f'{i}' for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)] + [f">{self.GRADE_MAX}"]
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
        labels = [f"<{self.GRADE_MIN}"] + [f'{i}' for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)] + [f">{self.GRADE_MAX}"]
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        _, ax = plt.subplots(figsize=(20, 10))
        
        plt.title("Confusion Matrix")
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=True, fmt='g')
        #plt.savefig(filename)
        plt.show()