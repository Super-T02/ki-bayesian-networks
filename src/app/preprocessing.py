import abc
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

class Preprocessing(abc.ABC):
    """The abstract base class for all preprocessing classes."""
    
    def __init__(self, data: pd.DataFrame) -> "Preprocessing":
        self._original_data: pd.DataFrame = data.copy()
        self._processed_data: pd.DataFrame | None = None
        
    @property
    def processed_data(self) -> pd.DataFrame | None:
        """The processed data."""
        return self._processed_data
        
    @abc.abstractmethod
    def process(self) -> None:
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
        
class PreprocessingCSV(Preprocessing):
    """The preprocessing class for CSV files."""
    
    CSV_STRUCTURE = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']
    TRANSFORM_TO_NUMERICAL = ['Medu', 'Fedu']
    
    def __init__(self, path: str, *args, **kwargs) -> "PreprocessingCSV":
        """Load the csv file and initialize the PreprocessingCSV class.

        Args:
            path (str): Path to the csv file.

        Returns:
            PreprocessingCSV: PreprocessingCSV class.
        """
        data = pd.read_csv(path, *args, **kwargs)
        super().__init__(data)
        
    def process(self) -> None:
        """Processes the data."""
        data = self._original_data.copy()
        
        # Delete columns which are not defined
        for column in data.columns:
            if column not in self.CSV_STRUCTURE:
                del data[column]     
        
        # Create binary columns
        data = self._create_binary_columns(data)
        
        # Create numerical columns
        data = self._create_numerical_columns(data)
        
        # Merge columns to desired structure
        data = self._merge_columns(data)
        
        return data
        
    def _create_binary_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create binary columns for the given data and columns that exist.

        Args:
            data (pd.DataFrame): Data to create binary columns for.

        Returns:
            pd.DataFrame: Data with binary columns.
        """
        
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
        
        return data
        
    def _create_numerical_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform columns from categorical to numerical.

        Args:
            data (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        
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
        # Support
        cols = self._extract_existing_cols(data, ['schoolsup', 'famsup', 'paid'])
        if len(cols) > 0:
            df['sup'] = df.apply(lambda row: row[cols].sum(), axis=1)
            data = data.drop(cols, axis=1)

        # Alcohol
        cols = self._extract_existing_cols(data, ['Dalc', 'Walc'])
        if len(cols) > 0:
            df['alc'] = df[cols].median(axis=1).round(0)
            df = df.drop(cols, axis=1)

        # Social
        cols = self._extract_existing_cols(data, ['goout', 'freetime'])
        if len(cols) > 0:
            df['social'] = df[cols].median(axis=1).round(0)
            df = df.drop(cols, axis=1)

        # Parent education
        cols = self._extract_existing_cols(data, ['Medu', 'Fedu'])
        if len(cols) > 0:
            df['Pedu'] = df[cols].median(axis=1).round(0)
            df = df.drop(cols, axis=1)

        # Jobs
        self._merge_job(data, ['Mjob_at_home', 'Fjob_at_home'], 'Pjob_at_home')
        self._merge_job(data, ['Mjob_health', 'Fjob_health'], 'Pjob_health')
        self._merge_job(data, ['Mjob_services', 'Fjob_services'], 'Pjob_services')
        self._merge_job(data, ['Mjob_teacher', 'Fjob_teacher'], 'Pjob_teacher')
        self._merge_job(data, ['Mjob_other', 'Fjob_other'], 'Pjob_other')
        
        return df
    
    def _merge_job(self, data: pd.DataFrame, old_names: list[str], new_name: str) -> pd.DataFrame:
        """Maps the given jobs to the new job name.

        Args:
            data (pd.DataFrame): Dataframe to be transformed.
            old_names (list[str]): Old job names (max two names).
            new_name (str): New job name.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        if len(old_names) > 2:
            raise ValueError('Only two old names are allowed.')
        
        cols = self._extract_existing_cols(data, old_names)
        if len(cols) == 2:
            data[new_name] = data.apply(lambda row: 1. if row[cols[0]] == 1 or row[cols[1]] == 1 else 0., axis=1)
            data = data.drop(cols, axis=1)
        elif len(cols) == 1:
            data[new_name] = data[cols[0]]
            data = data.drop(cols, axis=1)
        
        # if old_names[0] in cols and old_names[1] in cols:
        #     data[new_name] = data.apply(lambda row: 1. if row[old_names[0]] == 1 or row[old_names[1]] == 1 else 0., axis=1)
        #     data = data.drop(old_names, axis=1)
        # elif old_names[0] in cols and old_names[1] not in cols:
        #     data[new_name] = data[old_names[0]]
        #     data = data.drop(old_names[0], axis=1)
        # elif old_names[0] not in cols and old_names[1] in cols:
        #     data[new_name] = data[old_names[1]]
        #     data = data.drop(old_names[1], axis=1)
            
        return data