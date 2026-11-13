class DataCleaner:

    def clean(self, df):

        df = df.drop_duplicates()

        df.columns = df.columns.str.strip()

        object_columns = df.select_dtypes(include=["object"]).columns

        for column in object_columns:
            df[column] = df[column].str.strip()

        df.replace("", None, inplace=True)

        df = df.dropna()

        return df