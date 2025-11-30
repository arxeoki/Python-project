import pandas as pd

def load_data(path):
    """
    Load and clean a CSV dataset.

    This function reads a CSV file, displays basic information, removes missing
    values and duplicated rows, and returns a cleaned pandas DataFrame.

    Parameters
    ----------
    path : str
        File path to the CSV dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with no NaN or duplicate rows.
    """

    df = pd.read_csv(path)
    print(df.head())
    df.info()
    print(df.describe())

    rows_before = len(df)
    dup_count = df.duplicated().sum()

    df = df.dropna()
    df = df.drop_duplicates()

    rows_after = len(df)
    print(f"\nRows before : {rows_before}")
    print(f"Rows after : {rows_after}")
    print(f"Rows removed : {rows_before - rows_after}")

    return df
