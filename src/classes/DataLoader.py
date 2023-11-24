import pandas as pd


class DataLoader:

    def __init__(self, path_to_file, independent_column_names, dependent_column_name):
        if not path_to_file:
            raise Exception("Must provide path to csv file containing data")
        self.path_to_file = path_to_file

        if not independent_column_names:
            raise Exception(
                "Must provide the names of columns (in order of desired dimensions) that contain the independent variables")
        self.independent_column_names = independent_column_names

        if not dependent_column_name:
            raise Exception(
                "Must provide the names of the column which contains the dependent variable")
        self.dependent_column_name = dependent_column_name

        try:
            self.df = pd.read_csv(filepath_or_buffer=path_to_file)
        except:
            raise Exception(f"Error loading data file: {path_to_file}, please ensure file exists and path is correct!")

        ind_data = []
        for col_name in independent_column_names:
            ind_data.append(self.df[col_name].to_numpy())
        dep_data = self.df[dependent_column_name].to_numpy()
        self.ind_data = ind_data
        self.dep_data = dep_data
