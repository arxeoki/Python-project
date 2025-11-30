from data_loading import load_data
from features import feature_engineering
from outliers import outliers
from eda import *

df = load_data("data/diabetes_dataset.csv")

valid_ranges = {
    'alcohol_consumption_per_week':       (0, 28),
    'physical_activity_minutes_per_week': (0, 900),
    'sleep_hours_per_day':                (3, 12),
    'screen_time_hours_per_day':          (0, 16),
    'bmi':                                (10, 50),
    'waist_to_hip_ratio':                 (0.5, 2.0),
    'systolic_bp':                        (80, 220),
    'diastolic_bp':                       (50, 130),
    'heart_rate':                         (30, 200),
    'cholesterol_total':                  (100, 400),
    'hdl_cholesterol':                    (20, 100),
    'ldl_cholesterol':                    (0, 250),
    'triglycerides':                      (30, 1000),
    'glucose_fasting':                    (50, 200),
    'glucose_postprandial':               (70, 300),
    'insulin_level':                      (2, 50),
    'hba1c':                              (4, 14),
    'diabetes_risk_score':                (0, 100),
}

iqr_outliers, range_outliers, df = outliers(df, valid_ranges)

print("\nIQR outlier counts (per column):")
print(iqr_outliers[iqr_outliers > 0])

print("\nValues outside realistic ranges:")
print(range_outliers[range_outliers > 0])

df = feature_engineering(df)

gender_diabetes_bar(df)
ethnicity_diabetes_bar(df)
age_bmi_pie(df)
diet_diabetes_bar(df)
glucose_stage_box(df)
hba1c_fasting_scatter(df)
bmi_diabetes_bar(df)
activity_bmi_bar(df)
family_bar(df) 
age_diabetes_line(df)
corr_matrix(df)
top_corr(df, target_col='diabetes_risk_score', top_k=5)
