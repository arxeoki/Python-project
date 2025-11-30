import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, "results")

def ethnicity_diabetes_bar(df):
    """
    Plot the proportion of each diabetes stage within each ethnicity.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns 'ethnicity' and 'diabetes_stage'.
    """

    ratio = df.groupby(['ethnicity', 'diabetes_stage'], observed=False).size().reset_index(name='count')
    ratio['ratio'] = ratio.groupby('ethnicity')['count'].transform(lambda x: x / x.sum())
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ratio, x='ethnicity', y='ratio', hue='diabetes_stage', palette='Set2')
    plt.title('Proportion of diabetes stage for each ethnicity')
    plt.ylabel('Ratio')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ethnicity_diabetes_bar.png")
    plt.show()

def gender_diabetes_bar(df):
    """
    Plot the proportion of each diabetes stage within each gender.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with columns 'gender' and 'diabetes_stage'.
    """

    ratio = df.groupby(['gender', 'diabetes_stage'], observed=False).size().reset_index(name='count')
    ratio['ratio'] = ratio.groupby('gender')['count'].transform(lambda x: x / x.sum())
    plt.figure(figsize=(10, 6))
    sns.barplot(data=ratio, x='gender', y='ratio', hue='diabetes_stage', palette='Set2')
    plt.title("Proportion of diabetes stage for each gender")
    plt.ylabel("Ratio")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/gender_diabetes_bar.png")
    plt.show()

def age_bmi_pie(df):
    """
    Plot pie charts for BMI category distribution and age group distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'bmi_category' and 'age_group' columns.
    """

    bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    age_order = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    bmi_counts = df['bmi_category'].value_counts().reindex(bmi_order, fill_value=0)
    age_counts = df['age_group'].value_counts().reindex(age_order, fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].pie(bmi_counts.values, labels=bmi_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title("BMI")
    axes[1].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title("Age")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/age_bmi_pie.png")
    plt.show()

def bmi_diabetes_bar(df):
    """
    Plot diabetes diagnosis proportiob across BMI categories.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'bmi_category' and 'diagnosed_diabetes'.
    """

    bmi_counts = df.groupby(['bmi_category', 'diagnosed_diabetes'], observed=False).size().reset_index(name='count')
    bmi_counts['percentage'] = bmi_counts.groupby('bmi_category', observed=False)['count'].transform(lambda x: 100 * x / x.sum())

    plt.figure(figsize=(8, 5))
    sns.barplot(data=bmi_counts, x='bmi_category', y='percentage', hue='diagnosed_diabetes', palette={0: 'green', 1: 'red'})
    plt.title("Diabetes ratio across BMI")
    plt.xlabel("BMI")
    plt.ylabel("Proportion %")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/bmi_diabetes_bar.png")
    plt.show()

def diet_diabetes_bar(df):
    """
    Plot diabetes diagnosis proportion by diet score level.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'diet_score_level' and 'diagnosed_diabetes'.
    """

    grouped = df.groupby(['diet_score_level', 'diagnosed_diabetes'], observed=False).size().reset_index(name='count')
    total_per_level = grouped.groupby('diet_score_level', observed=False)['count'].transform('sum')
    grouped['percentage'] = grouped['count'] * 100 / total_per_level
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped, x='diet_score_level', y='percentage', hue='diagnosed_diabetes')
    plt.title("Diabetes by diet score level")
    plt.xlabel("Diet score level")
    plt.ylabel("Proportion for each level %")
    plt.legend(title="Diagnosed diabetes (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/diet_diabetes_bar.png")
    plt.show()

def glucose_stage_box(df):
    """
    Plot boxplots of fasting and postprandial glucose by diabetes stage.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'diabetes_stage', 'glucose_fasting', 'glucose_postprandial'.
    """

    stage_order = ["No Diabetes", "Pre-Diabetes", "Gestational", "Type 1", "Type 2"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    sns.boxplot(data=df, x="diabetes_stage", y="glucose_fasting", order=stage_order, palette="Set2", ax=axes[0])
    axes[0].set_title("Fasting Glucose by Diabetes Stage")
    axes[0].set_xlabel("Diabetes Stage")
    axes[0].set_ylabel("Fasting Glucose (mg/dL)")
    sns.boxplot(data=df, x="diabetes_stage", y="glucose_postprandial", order=stage_order, palette="Set2", ax=axes[1])
    axes[1].set_title("Postprandial Glucose by Diabetes Stage")
    axes[1].set_xlabel("Diabetes Stage")
    axes[1].set_ylabel("Postprandial Glucose (mg/dL)")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/glucose_stage_box.png")
    plt.show()

def hba1c_fasting_scatter(df):
    """
    Scatter plot of HbA1c versus fasting glucose, colored by diagnosis.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'glucose_fasting', 'hba1c', 'diagnosed_diabetes'.
    """

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='glucose_fasting', y='hba1c', hue='diagnosed_diabetes', palette={0: 'green', 1: 'red'}, alpha=1)
    plt.title("HbA1c vs Fasting glucose")
    plt.xlabel("Fasting glucose mg/dL")
    plt.ylabel("HbA1c %")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/hba1c_fasting_scatter.png")
    plt.show()

def family_bar(df):
    """
    Plot diabetes stage proportions by family history of diabetes.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'family_history_diabetes' and 'diabetes_stage'.
    """

    stage_counts = df.groupby(['family_history_diabetes', 'diabetes_stage'], observed=False).size().reset_index(name='count')
    stage_counts['percentage'] = stage_counts.groupby('family_history_diabetes')['count'].transform(lambda x: 100 * x / x.sum())
    plt.figure(figsize=(10, 6))
    sns.barplot(data=stage_counts, x='family_history_diabetes', y='percentage', hue='diabetes_stage', palette='Set2')
    plt.title("Diabetes stage ratio by family history")
    plt.xlabel("Family history")
    plt.ylabel("Percentage %")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/family_bar.png")
    plt.show()

def age_diabetes_line(df):
    """
    Plot percentage of diagnosed diabetes cases by age group.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'age_group' and 'diagnosed_diabetes'.
    """

    age_order = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    grouped = df.groupby('age_group', observed=False)['diagnosed_diabetes'].mean()
    age_percent = grouped.reindex(age_order) * 100
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=age_percent.index, y=age_percent.values, marker='o', linewidth=2, color='red')
    plt.title("Percentage of diagnosed cases by age")
    plt.xlabel("Age")
    plt.ylabel("Diagnosed %")
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/age_diabetes_line.png")
    plt.show()

def activity_bmi_bar(df):
    """
    Plot diagnosis rate by physical activity level and BMI group.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with 'activity_level', 'bmi', and 'diagnosed_diabetes'.
    """

    df['bmi_group'] = np.where(df['bmi'] < 25, 'BMI < 25', 'BMI > 25')
    grouped = df.groupby(['activity_level', 'bmi_group'], observed=False)['diagnosed_diabetes'].mean().reset_index()
    grouped['percentage'] = grouped['diagnosed_diabetes'] * 100
    plt.figure(figsize=(8, 6))
    sns.barplot(data=grouped, x='activity_level', y='percentage', hue='bmi_group', palette=['#5ab4ff', '#ff4c4c'])
    plt.title('Diagnosis rate by physical activity and bmi')
    plt.ylabel('Diagnosis rate %')
    plt.xlabel('Physical activity level')
    plt.ylim(0, 100)
    plt.legend(title='BMI Group')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/activity_bmi_bar.png")
    plt.show()

def corr_matrix(df):
    """
    Compute and display the full correlation matrix of all numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataset containing numeric and non-numeric features.

    Returns
    -------
    None
        Displays a heatmap of the correlation matrix.
    """

    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", linewidths=0.5, annot=True, fmt=".2f")
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{results_dir}/corr_matrix.png")
    plt.show()


def top_corr(df, target_col, top_k):
    """
    Print the strongest positive and negative correlations with a target column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset with numeric columns.
    target_col : str
        Column for which correlations will be ranked.
    top_k : int
        Number of top positive and negative correlations to display.

    Returns
    -------
    None
        Prints correlation values.
    """

    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    target_corr = corr[target_col].drop(labels=[target_col])
    top_pos = target_corr.sort_values(ascending=False).head(top_k)
    top_neg = target_corr.sort_values().head(top_k)
    print(f"\nTop {top_k} Positive correlations with '{target_col}':"); print(top_pos)
    print(f"\nTop {top_k} Negative correlations with '{target_col}':"); print(top_neg)


