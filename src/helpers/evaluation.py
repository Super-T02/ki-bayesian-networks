from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class Evaluation():
    
    GRADE_MAX = 20
    GRADE_MIN = 0
    OFFSET = 0
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> "Evaluation":
        self.y_true = y_true
        self.y_pred = y_pred
        
        self.accuracy = None
        self.mean_absolute_error = None
        self.tp_fp_tn_fn = None
        self.calc_stats(self.y_true, self.y_pred)
    
    def calc_stats(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Calculates the accuracy, mean absolute error and confusion matrix for a model.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        """
        self.accuracy = accuracy_score(y_true, y_pred)
        self.mean_false_error = self.mean_error_false(y_true, y_pred)
        self.mean_absolute_error = mean_absolute_error(y_true, y_pred)
        self.tp_fp_tn_fn = self.calc_tp_fp_tn_fn(y_true, y_pred)
        self.f1_score = f1_score(y_true, y_pred, average="macro")
        self.precision = precision_score(y_true, y_pred, average="macro")
        self.recall = recall_score(y_true, y_pred, average="macro")
        
    def calc_tp_fp_tn_fn(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the True Positives, False Positives, True Negatives and False Negatives for each grade.

        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.

        Returns:
            np.ndarray: The True Positives, False Positives, True Negatives and False Negatives for each grade.
        """
        matrix = np.zeros((self.GRADE_MAX - self.GRADE_MIN + 1 + self.OFFSET, 2, 2))
        
        for i in range(0, self.GRADE_MAX - self.GRADE_MIN + 1 + self.OFFSET):
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
        print("Accuracy:", self.accuracy.round(3))
        print("Mean false error:", self.mean_false_error.round(2))
        print("Mean absolute error:", self.mean_absolute_error.round(2))
        print("Precision:", self.precision)
        print("Recall:", self.recall)
        print("F1 score:", self.f1_score)
        
    def plot(self) -> None:
        raise NotImplementedError("The plot method must be implemented in the subclass.")
    
    def plot_confusion_matrix(self) -> None:
        raise NotImplementedError("The plot_confusion_matrix method must be implemented in the subclass.")
    
    def _show_plot(self, x: list) -> None:
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
        
    def _show_plot_confusion_matrix(self, cm: np.ndarray, labels: list) -> None:
        """Shows the confusion matrix for a model.

        Args:
            cm (np.ndarray): The confusion matrix.
            labels (list): Labels for the axes.
        """
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        _, ax = plt.subplots(figsize=(20, 10))
        
        plt.title("Confusion Matrix")
        sns.heatmap(cm, ax=ax, cmap="Blues", annot=True, fmt='g')
        #plt.savefig(filename)
        plt.show()
        
class MultiClassEvaluation(Evaluation):
    
    MATCH_LESS_EQUAL = 0
    GRADE_MAX = 20
    GRADE_MIN = 0
    OFFSET = 0
    
    def __init__(self, result: pd.DataFrame, match: str = 'G3', is_string: bool = False, min_grade: int = 0, max_grade: int = 20, offset: int = 0):
        if f"{match}" not in result.columns or f"{match}_pred" not in result.columns:
            raise ValueError(f"The result DataFrame must contain the columns {match} and {match}_pred.")
        
        self.GRADE_MIN = min_grade
        self.GRADE_MAX = max_grade
        self.OFFSET = offset
        self.is_string = is_string
        
        y_true = result[f"{match}"].values
        y_pred = result[f"{match}_pred"].values
        
        if is_string:
            self.y_str_true = [str(i) for i in y_true.copy()]
            self.y_str_pred = [str(i) for i in y_pred.copy()]
            y_true = self.map_grades(y_true)
            y_pred = self.map_grades(y_pred)
            
        
        super().__init__(y_true, y_pred)
    
    def map_grades(self, grades: np.ndarray) -> np.ndarray:
        """Maps the grades to the correct values.
        Example:
        <7 --> 0
        7 - 17 --> 1 - 11
        >17 --> 12

        Args:
            grades (np.ndarray): The list of grades.

        Returns:
            np.ndarray: The list of mapped grades.
        """
        grades = grades.copy()
        for i in range(len(grades)):
            if grades[i] == f'<{self.GRADE_MIN}':
                grades[i] = 0
            elif grades[i] == f'>{self.GRADE_MAX}':
                grades[i] = self.GRADE_MAX - self.GRADE_MIN + 2
            else:
                grades[i] = int(grades[i]) - self.GRADE_MIN + 1
        return grades.astype(np.int64)
        
    def plot(self):
        """Plots the true and predicted labels."""
        # Set up plot
        if self.is_string:
            x = [f"<{self.GRADE_MIN}"] + [f'{i}' for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)] + [f">{self.GRADE_MAX}"]
        else:
            x = [i for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)]
        self._show_plot(x)
        
        
    def plot_confusion_matrix(self) -> None:
        """Plots the confusion matrix for a model."""
        if self.is_string:
            labels = [f"<{self.GRADE_MIN}"] + [f'{i}' for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)] + [f">{self.GRADE_MAX}"]
            cm = confusion_matrix(self.y_str_true, self.y_str_pred, labels=labels)
        else:
            labels = [i for i in range(self.GRADE_MIN, self.GRADE_MAX + 1)]
            cm = confusion_matrix(self.y_true, self.y_pred)
        self._show_plot_confusion_matrix(cm, labels)

class BinaryEvaluation(Evaluation):
    
    MATCH_LESS_EQUAL = 0
    GRADE_MAX = 1
    GRADE_MIN = 0
    OFFSET = 0
    
    def __init__(self, result: pd.DataFrame, match: str = 'G3'):
        if f"{match}" not in result.columns or f"{match}_pred" not in result.columns:
            raise ValueError(f"The result DataFrame must contain the columns {match} and {match}_pred.")
        super().__init__(result[f"{match}"].to_numpy(), result[f"{match}_pred"].to_numpy())
    
    def plot(self):
        """Plots the true and predicted labels."""
        x = [0, 1]
        self._show_plot(x)
        
    def plot_confusion_matrix(self) -> None:
        """Plots the confusion matrix for a model."""
        labels = [0, 1]
        cm = confusion_matrix(self.y_true, self.y_pred, labels=labels)
        self._show_plot_confusion_matrix(cm, labels)