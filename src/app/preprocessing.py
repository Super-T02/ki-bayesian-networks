import abc
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

class Preprocessing(abc.ABC):
    """The abstract base class for all preprocessing classes."""
    
    def __init__(self, data: pd.DataFrame) -> "Preprocessing":
        self._original_data: pd.DataFrame = data.copy()
        self._processed_data: pd.DataFrame | None = None
    
    @property
    def original_data(self) -> pd.DataFrame:
        """The original data."""
        return self._original_data
    
    @property
    def processed_data(self) -> pd.DataFrame | None:
        """The processed data."""
        return self._processed_data
        
    @abc.abstractmethod
    def process(self) -> "Preprocessing":
        """Processes the data."""
        raise NotImplementedError
    
    def reset(self) -> None:
        """Resets the processed data to None."""
        self._processed_data = None
    
    def _extract_existing_cols(self, data: pd.DataFrame, required_cols: list[str]) -> list[str]:
        """Extracts the existing columns from the given list.

        Args:
            data (pd.DataFrame): Data with the existing columns.
            required_cols (list[str]): Required columns.

        Returns:
            list[str]: Existing columns of the required columns.
        """
        cols = []
        for col in required_cols:
            if col in data.columns:
                cols.append(col)
        return cols
    
class BinaryPreprocessing(Preprocessing):
    """Preprocessing binary data."""
    STRUCTURE = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', '_weight']
    
    def __init__(self, data: pd.DataFrame) -> "Preprocessing":
        super().__init__(data)
        
    def process(self) -> "BinaryPreprocessing":
        data = self._original_data.copy()
        
        # Delete columns which are not defined
        for column in data.columns:
            if column not in self.STRUCTURE:
                del data[column]     
        
        # Create binary columns
        data = self._create_binary_columns(data)   
        
        # Transform age and absences
        data = self._bound_age_absences(data)     
        
        self._processed_data = data
        
        return self
    
    def _create_binary_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create binary columns for the given data and columns that exist.

        Args:
            data (pd.DataFrame): Data to create binary columns for.

        Returns:
            pd.DataFrame: Data with binary columns.
        """
        data = data.copy()
        
        if 'school' in data.columns:
            data['school'] = data['school'].map({'GP': 0, 'MS': 1})
        if 'sex' in data.columns:
            data['sex'] = data['sex'].map({'F': 0, 'M': 1})
        if 'address' in data.columns:
            data['address'] = data['address'].map({'U': 0, 'R': 1})
        if 'famsize' in data.columns:
            data['famsize'] = data['famsize'].map({'LE3': 0, 'GT3': 1})
        if 'Pstatus' in data.columns:
            data['Pstatus'] = data['Pstatus'].map({'T': 0, 'A': 1})
        if 'schoolsup' in data.columns:
            data['schoolsup'] = data['schoolsup'].map({'yes': 1, 'no': 0})
        if 'famsup' in data.columns:
            data['famsup'] = data['famsup'].map({'yes': 1, 'no': 0})
        if 'paid' in data.columns:
            data['paid'] = data['paid'].map({'yes': 1, 'no': 0})
        if 'activities' in data.columns:
            data['activities'] = data['activities'].map({'yes': 1, 'no': 0})
        if 'nursery' in data.columns:
            data['nursery'] = data['nursery'].map({'yes': 1, 'no': 0})
        if 'higher' in data.columns:
            data['higher'] = data['higher'].map({'yes': 1, 'no': 0})
        if 'internet' in data.columns:
            data['internet'] = data['internet'].map({'yes': 1, 'no': 0})
        if 'romantic' in data.columns:
            data['romantic'] = data['romantic'].map({'yes': 1, 'no': 0})
        # # !NOT SURE IF THIS IS CORRECT!
        # if 'failures' in data.columns:
        #     data['failures'] = data['failures'].apply(lambda x: 1 if x > 0 else 0)
        
        return data
    
    def _bound_age_absences(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bounds the age and absences to the desired values.
        Absences have the upper bound of 15 and age of 20.

        Args:
            data (pd.DataFrame): Data to bound the values.

        Returns:
            pd.DataFrame: Data with bounded values.
        """
        data = data.copy()
        
        if 'absences' in data.columns and type(data['absences'][0]) == np.int64:
            data['absences'] = data['absences'].apply(lambda x: '>15' if x > 15 else f'{x}')
        if 'age' in data.columns and type(data['age'][0]) == np.int64:
            data['age'] = data['age'].apply(lambda x: '>20' if x > 20 else f'{x}')
        
        return data
        
class TunedPreprocessing(BinaryPreprocessing):
    """The preprocessing for the compressed network."""
    
    TRANSFORM_TO_NUMERICAL = ['Mjob', 'Fjob']
    
    def __init__(self, data: pd.DataFrame) -> "TunedPreprocessing":
        """Initializes the preprocessing for the compressed network.

        Args:
            data (pd.DataFrame): Data to be processed.

        Returns:
            PreprocessingCompressed: The preprocessing object.
        """
        super().__init__(data)
        
    def process(self) -> "TunedPreprocessing":
        """Processes the data."""
        # Delete not needed columns and create binary columns
        data = super().process()._processed_data.copy()
        
        # Create numerical columns
        data = self._create_numerical_columns(data)
        
        # Merge columns to desired structure
        data = self._merge_columns(data)
        
        self._processed_data = data
        
        return self
        
    def _create_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform columns from categorical to numerical.

        Args:
            data (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        data = data.copy()
        
        # Transform existing columns only to avoid errors
        cols = []
        for col in self.TRANSFORM_TO_NUMERICAL:
            if col in data.columns:
                cols.append(col)
        
        # Create numerical columns    
        converter = make_column_transformer((OneHotEncoder(), cols), remainder='passthrough', verbose_feature_names_out=False)
        data = converter.fit_transform(data)
        data = pd.DataFrame(data, columns=converter.get_feature_names_out())
        
        return data
        
    def _merge_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Merges the columns to the desired structure.

        Args:
            data (pd.DataFrame): Data to merge the columns.

        Returns:
            pd.DataFrame: Data with merged columns.
        """
        data = data.copy()
        # Support
        cols = self._extract_existing_cols(data, ['schoolsup', 'famsup', 'paid'])
        if len(cols) > 0:
            data['sup'] = data.apply(lambda row: row[cols].sum(), axis=1)
            data = data.drop(cols, axis=1)

        # Alcohol
        cols = self._extract_existing_cols(data, ['Dalc', 'Walc'])
        if len(cols) > 0:
            data['alc'] = data[cols].median(axis=1).astype(np.float64).round(0).astype(int)
            data = data.drop(cols, axis=1)

        # Social
        cols = self._extract_existing_cols(data, ['goout', 'freetime'])
        if len(cols) > 0:
            data['social'] = data[cols].median(axis=1).astype(np.float64).round(0).astype(int)
            data = data.drop(cols, axis=1)

        # Parent education
        cols = self._extract_existing_cols(data, ['Medu', 'Fedu'])
        if len(cols) > 0:
            data['Pedu'] = data[cols].median(axis=1).astype(np.float64).round(0).astype(int)
            data = data.drop(cols, axis=1)

        # Jobs
        data = self._merge_job(data, ['Mjob_teacher', 'Fjob_teacher'], 'Pjob_teacher')
        data = self._merge_job(data, ['Mjob_health', 'Fjob_health'], 'Pjob_health')
        data = self._merge_job(data, ['Mjob_services', 'Fjob_services'], 'Pjob_services')
        data = self._merge_job(data, ['Mjob_at_home', 'Fjob_at_home'], 'Pjob_at_home')
        data = self._merge_job(data, ['Mjob_other', 'Fjob_other'], 'Pjob_other')
        
        return data
    
    def _merge_job(self, data: pd.DataFrame, old_names: list[str], new_name: str) -> pd.DataFrame:
        """Maps the given jobs to the new job name.
        

        Args:
            data (pd.DataFrame): Dataframe to be transformed.
            old_names (list[str]): Old job names (max two names).
            new_name (str): New job name.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        data = data.copy()
        
        if len(old_names) > 2:
            raise ValueError('Only two old names are allowed.')
        
        cols = self._extract_existing_cols(data, old_names)
        if len(cols) == 2:
            data[new_name] = data.apply(lambda row: 1. if row[cols[0]] == 1 or row[cols[1]] == 1 else 0., axis=1)
            data = data.drop(cols, axis=1)
        elif len(cols) == 1:
            data[new_name] = data[cols[0]].astype(np.float64)
            data = data.drop(cols, axis=1)
        else:
            pass
            
        return data
    
class BinaryOutPutPreprocessing(Preprocessing):
    
    def __init__(self, data: pd.DataFrame) -> "BinaryOutPutPreprocessing":
        super().__init__(data)
        
    def process(self) -> "BinaryOutPutPreprocessing":
        """Processes the data.

        Returns:
            BinaryOutPutPreprocessing: Self.
        """
        data = self.original_data.copy()
        
        data['G3'] = data['G3'].apply(lambda x: 1 if x > 10 else 0)
    
        self._processed_data = data
        
        return self
        
class BoundOutPutPreprocessing(Preprocessing):
    
    def __init__(self, data: pd.DataFrame) -> "BoundOutPutPreprocessing":
        super().__init__(data)
        
    def process(self) -> "BoundOutPutPreprocessing":
        """Processes the data.

        Returns:
            BoundOutPutPreprocessing: Self
        """
        data = self.original_data.copy()
        data = self._bound_cols(data)
        self._processed_data = data
        
        return self
        
    def _bound_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        """Bounds the grades to the lower (7) and upper bound (17).
        
        Args:
            data (pd.DataFrame): To be bounded.
            
        Returns:
            pd.DataFrame: Data with bounded cols.        
        """
        data = data.copy()
        
        data = self._bound_grade('G1', data)
        data = self._bound_grade('G2', data)
        data = self._bound_grade('G3', data)
    
        return data
    
    def _bound_grade(self, name: str, data: pd.DataFrame) -> pd.DataFrame:
        """Bounds the grade to the lower (7) and upper bound (17).

        Args:
            name (str): Name of the column.
            data (pd.DataFrame): Data to be bounded.

        Returns:
            pd.DataFrame: Data with bounded column.
        """
        if name in data.columns and (type(data[name][0]) == int or type(data[name][0]) == np.int64):
            data[name] = data[name].apply(lambda x: '<7' if x < 7 else '>17' if x > 17 else f'{x}')
        return data