import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, "results")

def outliers(df, valid_ranges):
    """
    Detect outliers using both IQR and predefined valid ranges, and return a cleaned dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing numeric columns to check for outliers.
    valid_ranges : dict
        Dictionary mapping column names to (min, max) valid ranges.

    Returns
    -------
    tuple
        iqr_outliers : pandas.Series
            Number of IQR-based outliers per numeric column.
        range_outliers : pandas.Series
            Number of outliers based on valid ranges for each listed column.
        cleaned_df : pandas.DataFrame
            Dataframe with rows removed if they violate valid ranges.
    """

    valid_cols = [col for col in valid_ranges if col in df.columns]
    num_cols = df[valid_cols].select_dtypes(include='number').columns

    Q1 = df[num_cols].quantile(0.25)
    Q3 = df[num_cols].quantile(0.75)
    IQR = Q3 - Q1

    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    outlier_mask = (df[num_cols] < lower_fence) | (df[num_cols] > upper_fence)
    iqr_outliers = outlier_mask.sum()

    range_outliers = {}
    for col in valid_cols:
        low, high = valid_ranges[col]
        range_outliers[col] = ((df[col] < low) | (df[col] > high)).sum()
    range_outliers = pd.Series(range_outliers)

    clean_mask = pd.Series(True, index=df.index)
    for col in valid_cols:
        low, high = valid_ranges[col]
        clean_mask &= (df[col] >= low) & (df[col] <= high)

    cleaned_df = df[clean_mask].copy()
    cleaned_df.to_csv(f"{results_dir}/cleaned_dataset.csv", index=False)

    return iqr_outliers, range_outliers, cleaned_df




