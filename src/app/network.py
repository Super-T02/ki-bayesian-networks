from framework import Framework
from app.preprocessing import PreprocessingCSV
from pgmpy.models import BayesianNetwork

import pandas as pd

class Network:
    """The final Bayesian network model."""
    
    G_STATES = [i for i in range(0, 21, 1)]
    STATES = {
        "G1": G_STATES, 
        "G2": G_STATES, 
        "G3": G_STATES,
        "school": [0, 1],
        "sex": [0, 1],
        "age": [i for i in range(15, 23, 1)],
        "address": [0, 1],
        "famsize": [0, 1],
        "Pstatus": [0, 1],
        "Pedu": [i for i in range(0, 5, 1)],
        "Pjob_at_home": [0, 1],
        "Pjob_teacher": [0, 1],
        "Pjob_health": [0, 1],
        "Pjob_service": [0, 1],
        "Pjob_other": [0, 1],
        "reason": ["home", "reputation", "course", "other"],
        "guardian": ["mother", "father", "other"],
        "traveltime": [i for i in range(1, 5, 1)],
        "studytime": [i for i in range(1, 5, 1)],
        "failures": [i for i in range(0, 5, 1)],
        "sup": [i for i in range(0, 4, 1)],
        "activities": [0, 1],
        "nursery": [0, 1],
        "higher": [0, 1],
        "internet": [0, 1],
        "romantic": [0, 1],
        "famrel": [i for i in range(1, 6, 1)],
        "social": [i for i in range(1, 6, 1)],
        "alc": [i for i in range(1, 6, 1)],
        "health": [i for i in range(1, 6, 1)],
        "absences": [i for i in range(0, 94, 1)],
    }
        
    def __init__(self, file_path: str, n_jobs: int = -1) -> "Network":
        """Initializes the network."""
        self._njobs = n_jobs
        self.network = BayesianNetwork()
        self._state_names = self.STATES
        data = pd.read_csv(file_path, sep=";")
        pre = PreprocessingCSV(data)
        pre.process()
        self.train_data = pre.processed_data 
        self.train_data = self.train_data.drop(self.train_data[self.train_data['G3'] == 0].index)
        self.train_data = self.train_data.drop(self.train_data[self.train_data['G3'] == 1].index)
        self.train_data = self.prepare_data(self.train_data)
    
    def prepare_data(self, data: pd.DataFrame, *args) -> pd.DataFrame:
        """Converts the data to a format that can be used for training and drops not needed columns.

        Args:
            data (pd.DataFrame): The data to be prepared.
            args (list): The additional columns to be dropped.
            
        Returns:
            pd.DataFrame: The prepared data.
        """
        for column in [*args, 'Unnamed: 33']:
            if column in data.columns:
                del data[column]
        return data
    
    def create(self) -> None:
        """Ctrates the network."""
        for col in self.train_data.columns:
            self.network.add_node(col)
        
        # Edges for all sub networks
        edges = [
            # Medu and Fedu
            ('Pjob_at_home', 'Pedu'),
            ('Pjob_at_home', 'school'),
            ('Pjob_teacher', 'Pedu'),
            ('Pjob_teacher', 'school'),
            ('Pjob_health', 'Pedu'),
            ('Pjob_health', 'school'),
            ('Pjob_services', 'Pedu'),
            ('Pjob_other', 'Pedu'),
            ('Pedu', 'failures'),
            ('Pedu', 'higher'),
            ('Pedu', 'internet'),
            ('Pedu', 'school'),
            ('Pedu', 'sup'),
            ('Pedu', 'nursery'),
            ('Pedu', 'address'),
            
            # Family
            ('nursery', 'famsize'),
            ('famsize', 'Pstatus'),
            ('guardian', 'Pstatus'),
            ('health', 'famrel'),
            ('famrel', 'social'),
            ('sex', 'sup'),
            ('sup', 'higher'),
            ('sup', 'studytime'),
            ('school', 'sup'),
            # ('sup', 'age'),
            ('age', 'sup'),
            ('sup', 'higher'),
            
            
            # Free time
            ('sex', 'alc'),
            ('alc', 'social'),
            ('alc', 'absences'),
            ('alc', 'studytime'),
            ('alc', 'health'),
            ('social', 'failures'),
            ('social', 'studytime'),
            ('activities', 'social'),
            ('activities', 'failures'),
            ('activities', 'studytime'),
            ('Pstatus', 'activities'),
            ('absences', 'failures'),
            ('absences', 'school'),
            ('absences', 'higher'),
            # ('absences', 'age'),
            ('age', 'absences'),
            
            # School
            ('internet', 'address'),
            ('internet', 'school'),
            ('address', 'school'),
            ('school', 'traveltime'),
            ('address', 'traveltime'),
            
            # Rest
            ('failures', 'higher'),
            ('age', 'higher'),
            ('romantic', 'age'),
            ('guardian', 'age'),
            ('studytime', 'higher'),
        ]
        
        Framework.add_edges(self.network, edges)
    
    def fit(self) -> None:
        """Fit the network to the data."""
        self.network.fit(self.train_data, state_names=self._state_names, n_jobs=self._njobs)
    
    def get(self) -> BayesianNetwork:
        """Returns the network."""
        return self.network  
        
class NetworkG1G2(Network):
    """The final Bayesian network model with missing G1 and G2."""
    
    def __init__(self, file_path: str, n_jobs: int = -1) -> "Network":
        super().__init__(file_path, n_jobs)
        
    def create(self) -> None:
        """Creates the network."""
        super().create()
        edges = [
            ('Pedu', 'G3'),
            ('school', 'G3'),
            ('failures', 'G3'),
            ('higher', 'G3'),
            ('age', 'G3'),
            ('alc', 'G3'),
            ('studytime', 'G3'),
        ]
        Framework.add_edges(self.network, edges)
        
class NetworkG1OrG2(Network):
    """The final Bayesian network model with missing G1 or G2."""
    def __init__(self, file_path: str, n_jobs: int = -1) -> "Network":
        super().__init__(file_path, n_jobs)
        
    def create(self) -> None:
        """Creates the network."""
        super().create()
        edges = [
            ('G1', 'G3'),
            ('G2', 'G3'),
            
            ('reason', 'G1'),
            ('address', 'G1'),
            ('Pedu', 'G1'),
            ('school', 'G1'),
            ('failures', 'G1'),
            ('higher', 'G1'),
            # ('age', 'G1'),
            ('alc', 'G2'),
            ('studytime', 'G1'),
            ('reason', 'G2'),
            ('address', 'G2'),
            ('Pedu', 'G2'),
            ('school', 'G2'),
            ('failures', 'G2'),
            ('higher', 'G2'),
            # ('age', 'G2'),
            ('alc', 'G2'),
            ('studytime', 'G2'),
        ]
        Framework.add_edges(self.network, edges)