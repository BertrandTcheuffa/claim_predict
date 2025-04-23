import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(
    page_title="Fraud Prediction Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Welcome to our Claim Prediction Project")
st.sidebar.title("Navigation")
pages=["Home","Project Introduction","Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("",pages)

df=pd.read_csv("insurance_claims.csv", parse_dates=["policy_bind_date", "incident_date"], index_col="policy_number")

if page == pages[1]:
    st.write('## Project Goal')
    st.markdown("""
    Develop a predictive model capable of detecting potentially fraudulent claims by anaöyzing a dataset of insurance claims
    """)
    st.write('## Economical Context')
    st.markdown("""
    - Successfully achieving the project will help insurance companies reduce financial losses.
    - It will improve the efficiency of their claim validation processes.
    - Recommendation of fraud investigation threshold > 50EUR.
    """)
    st.write('## Data Framework')
    st.markdown("""
    - The dataset 'insurance_claims.csv' is freely available at this link: https://data.mendeley.com/datasets/992mh7dk9y/2
    - The dataset - 1000 observations / 40 features / Target variable: 'fraud_reported' 
    - No external business expert is involved.
    """)
    st.write('## Technical Challenges')
    st.markdown("""
    - Transform the business problem to a machine learning problem.
    - Perform data cleaning.
    - Deal with unbalanced variables.
    - Identify variables with the most influence on the target variable.
    - Data Normalization / Standardization.
    - Evaluate different models and find the best solution. 
    """)
    st.write('## Data pre-processing and feature engineering')
    st.markdown("""
    - Data cleaning: The dataset underwent a cleaning process.
    - Daa consistency: Measures were taken to ensure consistent data across all column.
    - Outlier detection: Outliers were identified and examined.
    - Plausibility check: The data was reviewed for plausibility to ensure realistic values.
    - Correlation analysis: Correlation among variables was displayed via heatmap.
    """)

if page == pages[2]:
    st.write('## Data Exploration')
    st.write('### Head of dataframe')
    st.dataframe(df.head())
    st.write('### Statistics')
    st.dataframe(df.describe())
    st.write('### Statistical Tests')
    
if page == pages[3]:
    st.write('## Data Vizualization')
    st.write('### Some Plots')
    colors = plt.cm.magma([0.2, 0.5, 0.7, 0.9])
    fig, ax = plt.subplots(3, 3, figsize=(20, 15))
    fig.tight_layout(pad=5)

# 1. insured sex distribution
    ax[0, 0].pie(
        df['insured_sex'].value_counts(normalize=True),
        labels=['Female', 'Male'],
        explode=[0.05, 0],
        autopct="%0.2f%%",
        colors=["silver", "wheat"]
    )
    ax[0, 0].set_title('Distribution Male/Female', fontsize=13)
    
# 2. age distribution
    sns.histplot(
        ax=ax[0, 1],
        x="age",
        data=df,
        color="brown",
        stat="density",
        kde=True
    )   
    ax[0, 1].set_title('Distribution of the Age', fontsize=13)
    ax[0, 1].spines['right'].set_visible(False)
    ax[0, 1].spines['top'].set_visible(False)

# 3.Authorities contacted by fraud reported (violin plot)
    pd.crosstab(df['authorities_contacted'], df['fraud_reported']).plot(
        kind='bar', 
        stacked=True, 
        ax=ax[0, 2], 
        color=colors  # Set the colors manually
    )
    ax[0, 2].set_title('Authorities Contacted by Fraud Reported', fontsize=13)  # Title
    ax[0, 2].spines['right'].set_visible(False)
    ax[0, 2].spines['top'].set_visible(False)
    ax[0, 2].set_xlabel('Authorities Contacted')
    ax[0, 2].set_ylabel('Count')  # Y-axis label
    # Rotate x-axis labels
    ax[0, 2].set_xticklabels(ax[0, 2].get_xticklabels(), rotation=30, ha='right')

# 4. witness distribution for fraud reported = yes, colorpalette magma
    sns.countplot(
        x='witnesses',data=df[df['fraud_reported'] == 'Y'], palette='magma', ax=ax[1, 0])
    ax[1, 0].set_title('Witnesses Distribution for Fraud Reported = Yes', fontsize=13)
    ax[1, 0].spines['right'].set_visible(False)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].set_xlabel('Number of Witnesses')
    ax[1, 0].set_ylabel('Count')




# 5. Stacked bar chart incident_severity distribution to fraud reported
    pd.crosstab(df['incident_severity'], df['fraud_reported']).plot(
        kind='bar', 
        stacked=True, 
        ax=ax[1, 1], 
        color=colors  # Set the colors manually
    )
    ax[1, 1].set_title('Incident Severity Distribution by Fraud Reported', fontsize=13)  # Title
    ax[1, 1].spines['right'].set_visible(False)
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].set_xlabel('Incident Severity')
    ax[1, 1].set_ylabel('Count')  # Y-axis label
    # Rotate x-axis labels
    ax[1, 1].set_xticklabels(ax[1, 1].get_xticklabels(), rotation=10, ha='right')



# 6. fraud reported distribution
    ax[1, 2].pie(
        df["fraud_reported"].value_counts(normalize=True),
        labels=df['fraud_reported'].value_counts().index.astype(str),
        explode=[0.05] * len(df['fraud_reported'].value_counts()),
        autopct="%0.2f%%",
        colors=["palevioletred", "darksalmon", "lightcoral"]
    )
    ax[1, 2].set_title('Distribution of Fraud Reported', fontsize=13)


# 7. auto_make distribution hue=fraud_reported, x axis auto make labels rotated
    sns.countplot(x='auto_make',data=df, hue='fraud_reported', palette='coolwarm',  # Updated color palette
    ax=ax[2, 0])
    ax[2, 0].set_title('Auto Make Distribution by Fraud Reported', fontsize=13)
    ax[2, 0].spines['right'].set_visible(False)
    ax[2, 0].spines['top'].set_visible(False)
    ax[2, 0].yaxis.set_ticks_position('left')
    ax[2, 0].xaxis.set_ticks_position('bottom')
    ax[2, 0].set_xticklabels(ax[2, 0].get_xticklabels(), rotation=45)
    ax[2, 0].set_xlabel('Auto Make')
    ax[2, 0].set_ylabel('Count')


# 8. incident_type distribution to fraud reported, rotate xlabels
    sns.countplot(x='incident_type',data=df, hue='fraud_reported', palette='magma', ax=ax[2, 1])
    ax[2, 1].set_title('Incident Type Distribution by Fraud Reported', fontsize=13)     # Title
    ax[2, 1].spines['right'].set_visible(False)
    ax[2, 1].spines['top'].set_visible(False)
    ax[2, 1].set_xlabel('Incident Type')
    ax[2, 1].set_xticklabels(ax[2, 1].get_xticklabels(), rotation=45)
    ax[2, 1].set_ylabel('Count')    # Y

# 9. chart insured_hobbies to fraud reported, rotate xlabels
# Filter the data for fraud_reported == 'Y' and get the top 10 hobbies
    top_hobbies = (
        df[df['fraud_reported'] == 'Y']['insured_hobbies']
        .value_counts()
        .head(10)
        .index)
    # Filter the original DataFrame to include only the top 10 hobbies
    filtered_df = df[df['insured_hobbies'].isin(top_hobbies)]

# Sort the filtered DataFrame by the count of fraud_reported='Y' in descending order
    sorted_hobbies = (
        filtered_df[filtered_df['fraud_reported'] == 'Y']['insured_hobbies']
        .value_counts()
        .index
    )
    filtered_df['insured_hobbies'] = pd.Categorical(
        filtered_df['insured_hobbies'], categories=sorted_hobbies, ordered=True
    )
# Plot the countplot for the filtered data
    sns.countplot(
        y='insured_hobbies',
        data=filtered_df,
        hue='fraud_reported',
        palette='coolwarm',
        ax=ax[2, 2])
    ax[2, 2].set_title('Top 10 Insured Hobbies Distribution by Fraud Reported', fontsize=13)  # Title
    ax[2, 2].spines['right'].set_visible(False)
    ax[2, 2].spines['top'].set_visible(False)
    ax[2, 2].set_xlabel('Count')
    ax[2, 2].set_ylabel('Insured Hobbies')  # Y-axis label
    ax[2, 2].set_xticklabels(ax[2, 2].get_xticklabels(), rotation=90)
    st.pyplot(fig)
    
    palette = sns.color_palette("hls", 12)
    fig2, axes = plt.subplots(3, 3,figsize=(20, 15))
    sns.histplot(x=df['injury_claim'], ax=axes[0, 0], bins=50, kde=True, color=palette[0], stat="percent")
    sns.histplot(x=df['property_claim'], ax=axes[0, 1], bins=50, kde=True, color=palette[1], stat="percent")
    sns.histplot(x=df['vehicle_claim'], ax=axes[0, 2], bins=50, kde=True, color=palette[2], stat="percent")
    sns.histplot(x=df['number_of_vehicles_involved'], ax=axes[1, 0], kde=True, color=palette[3], stat="percent")
    sns.histplot(x=df['age'], ax=axes[1, 1], bins=50, kde=True, color=palette[4], stat="percent")
    sns.histplot(x=df['policy_deductable'], ax=axes[1, 2], kde=True, color=palette[5], stat="percent")
    sns.histplot(x=df['policy_annual_premium'], ax=axes[2, 0], kde=True, color=palette[6], stat="percent")
    sns.histplot(x=df['capital-gains'], ax=axes[2, 1], kde=True, color=palette[7], stat="percent")
    sns.histplot(x=df['capital-loss'], ax=axes[2, 2], kde=True, color=palette[8], stat="percent")

    plt.tight_layout()
    plt.show()
    st.pyplot(fig2)
    
    
    col_quan = df.select_dtypes(include=['float64', 'int64']).columns
    fig3=plt.figure(figsize=(20, 15))
    sns.heatmap(np.round(df[col_quan].corr(),2), annot=True, center =0)
    st.pyplot(fig3)