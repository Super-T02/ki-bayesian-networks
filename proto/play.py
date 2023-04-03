import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bnlearn as bn


DATA_PATH = "data/"

data = pd.read_csv(DATA_PATH + "student-por_2.csv", sep=";")
print(data.head())