import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import dataframe_image as dfi
import networkx as nx
from pgmpy.models import BayesianNetwork

class Framework():
    
    @staticmethod
    def draw_model(model: BayesianNetwork, show: bool = True, save: bool = False, filename: str = "model.png"):
        """Draws a model as a networkx graph and saves it to a file.
        Args:
            model (Model): The model to draw.
            show (bool): Whether to show the model in a window.
            save (bool): Whether to save the model to a file.
            filename (str): The filename to save the model to.
        """
        G = nx.DiGraph(model.edges())
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=40)
        if save:
            plt.savefig(filename)
        if show:
            plt.show()
            
    @staticmethod
    def add_edges(model: BayesianNetwork, edges: list):
        """Adds edges to a model.
        Args:
            model (Model): The model to add edges to.
            edges (list): A list of edges to add.
        """
        for edge in edges:
            model.add_edge(*edge)