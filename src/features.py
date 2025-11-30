import numpy as np
import pandas as pd

def feature_engineering(df):

    """
    Generate categorical features and metabolic indices for diabetes analysis.

    This function creates grouped versions of diet score, age, BMI, and physical
    activity, and computes four indicators. Zero values in required indicator columns are replaced
    with NaN to avoid invalid logarithmic operations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing necessary clinical and lifestyle variables.

    Returns
    -------
    pandas.DataFrame
        Dataframe with added engineered feature columns.
    """

    df['diet_score_level'] = pd.cut(df['diet_score'], bins=np.linspace(0, 10, 11), labels=[1,2,3,4,5,6,7,8,9,10], include_lowest=True).astype(int)
    df['age_group'] = pd.cut(df['age'], bins=[0, 20, 30, 40, 50, 60, 70, 120], labels=['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+'], include_lowest=True)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'], include_lowest=True)
    df['activity_level'] = pd.cut(df['physical_activity_minutes_per_week'], bins=[0,180,480,960], labels=['Low','Moderate','High'], include_lowest=True)

    trig = df['triglycerides'].replace(0, np.nan)
    glu_f = df['glucose_fasting'].replace(0, np.nan)
    ins = df['insulin_level'].replace(0, np.nan)
    hdl = df['hdl_cholesterol'].replace(0, np.nan)

    df['tyg_index'] = np.log((trig * glu_f) / 2.0)
    df['homa_ir'] = (glu_f * ins) / 405.0
    df['quicki_index'] = 1.0 / (np.log(ins) + np.log(glu_f))
    df['aip'] = np.log10(trig / hdl)

    return df
