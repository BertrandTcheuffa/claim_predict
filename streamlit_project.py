import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with wide layout and custom theme
st.set_page_config(layout="wide")

# Custom CSS to style the sidebar with a blue gradient and improve radio buttons
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#2C3E50, #4CA1AF);
        color: white;
    }
    
    /* Increase radio button label size and change color */
    [data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF;
        font-size: 18px !important;
        font-weight: 600;
    }
    
    /* Make the "Go to:" text larger and more visible */
    .sidebar .sidebar-content label {
        color: #FFFFFF !important;
        font-size: 20px !important;
        font-weight: bold;
    }
    
    /* Make all sidebar text more visible */
    [data-testid="stSidebar"] p {
        color: #FFFFFF;
        font-size: 18px !important;
    }
    
    [data-testid="stSidebarNav"] {
        background-color: rgba(0, 0, 0, 0);
    }
    </style>
""", unsafe_allow_html=True)

df=pd.read_csv("insurance_claims.csv")

st.title("Insurance Claim Prediction ")
st.sidebar.title("Table of contents")
pages=["Business problem", "Data Visualization","Statistical Methods", "Feature Engineering",  "Modelling", "Real-life Sample Application"]
st.sidebar.write("\n\nCreated by:")
st.sidebar.write("Bertrand Tcheuffa  \n Nathalie Mugrauer  \n Quy-Manh Jurca-Tsan \n")
st.sidebar.write("\n\n\n")
# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page=st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label
  
if page == pages[0] : 
  st.write("### Introduction")
  st.write("Business Problem: Predict whether an insurance claim is fraudulent or not. ")
  st.write("Type of Machine Learning Problem: Supervised learning --> Binary Classification ")
  st.write("Brief overview of the dataset:  \n")
  st.dataframe(df.head(8))
  st.write("- The dataset contains 40 columns and 1000 rows.")
  st.write("- The dataset is made up of categorical and numerical features.")
  st.write("- Our target variable, 'fraud_reported',  indicates if a claim is fraudulent or not.")
  st.write(" Considering the relative large dimensionality of our dataset, after carefully exploring the data , we will review each feature and its relevance to the target variable. Based on this, we will select the most relevant features for our model.")
#############################################################################################################################################
#############################################################################################################################################

if page == pages[1] : 
  st.write("### Data Visualization")
  st.write("In this section, we will visualize the data to better understand the relationships between features and the target variable.")
  st.write("- We observe a distribution of gender that is fairly balanced. The age distribution is slightly skewed to the right, with a peak around 30-40 years. ")
  st.write("- The 'authorities_contacted' feature shows that most claims are reported to the police, while the 'incident_severity' feature indicates that most incidents are minor.") 
  st.write("- The 'witnesses' feature shows that most claims have two witnesses.") 
  st.write("- We observe our target variable, 'fraud_reported', is imbalanced, with a higher number of non-fraudulent claims.")
  st.write("- For the hobbies distribution we observe  that the hobbies chess and crossfit are the most common among fraudulent claims.")
  

  # New blue-green color palette
  colors = plt.cm.Blues([0.3, 0.5, 0.7, 0.9])
  accent_colors = plt.cm.GnBu([0.4, 0.6, 0.8, 1.0])
  
  # Increase global matplotlib font sizes
  plt.rcParams.update({
      'font.size': 16,
      'axes.titlesize': 20,
      'axes.labelsize': 20,
      'xtick.labelsize': 17,
      'ytick.labelsize': 17,
      'legend.fontsize': 17,
  })
  
  # Create figure with better size for readability
  fig, ax = plt.subplots(3, 3, figsize=(36, 36), dpi=200)
  fig.tight_layout(pad=7)  # Increase padding between subplots

  # 1. insured sex distribution
  ax[0, 0].pie(
      df['insured_sex'].value_counts(normalize=True),
      labels=['Female', 'Male'],
      explode=[0.05, 0],
      autopct="%0.2f%%",
      colors=["#4682B4", "#1E90FF"],  # Steel blue and dodger blue
      textprops={'fontsize': 18, 'fontweight': 'bold'}  # Make pie chart text more readable
  )
  ax[0, 0].set_title('Distribution Male/Female', fontsize=18, fontweight='bold')

  # 2. age distribution
  sns.histplot(
      ax=ax[0, 1],
      x="age",
      data=df,
      color="#0077b6",  # Dark blue
      stat="density",
      kde=True
  )
  ax[0, 1].set_title('Distribution of the Age', fontsize=18, fontweight='bold')
  ax[0, 1].spines['right'].set_visible(False)
  ax[0, 1].spines['top'].set_visible(False)
  ax[0, 1].set_xlabel('Age', fontsize=18, fontweight='bold')
  ax[0, 1].set_ylabel('Density', fontsize=18, fontweight='bold')

  # 3.Authorities contacted by fraud reported
  pd.crosstab(df['authorities_contacted'], df['fraud_reported']).plot(
      kind='bar', 
      stacked=True, 
      ax=ax[0, 2], 
      color=colors  # Blue colors
  )
  ax[0, 2].set_title('Authorities Contacted by Fraud Reported', fontsize=18, fontweight='bold')
  ax[0, 2].spines['right'].set_visible(False)
  ax[0, 2].spines['top'].set_visible(False)
  ax[0, 2].set_xlabel('Authorities Contacted', fontsize=18, fontweight='bold')
  ax[0, 2].set_ylabel('Count', fontsize=18, fontweight='bold')
  ax[0, 2].set_xticklabels(ax[0, 2].get_xticklabels(), rotation=30, ha='right', fontsize=14)
  ax[0, 2].legend(fontsize=14)

  # 4. witness distribution for fraud reported = yes
  sns.countplot(
      x='witnesses', data=df[df['fraud_reported'] == 'Y'], palette='Blues', ax=ax[1, 0])
  ax[1, 0].set_title('Witnesses Distribution for Fraud Reported = Yes', fontsize=18, fontweight='bold')
  ax[1, 0].spines['right'].set_visible(False)
  ax[1, 0].spines['top'].set_visible(False)
  ax[1, 0].set_xlabel('Number of Witnesses', fontsize=18, fontweight='bold')
  ax[1, 0].set_ylabel('Count', fontsize=18, fontweight='bold')

  # 5. Stacked bar chart incident_severity distribution to fraud reported
  pd.crosstab(df['incident_severity'], df['fraud_reported']).plot(
      kind='bar', 
      stacked=True, 
      ax=ax[1, 1], 
      color=accent_colors  # Green-blue colors
  )
  ax[1, 1].set_title('Incident Severity Distribution by Fraud Reported', fontsize=18, fontweight='bold')
  ax[1, 1].spines['right'].set_visible(False)
  ax[1, 1].spines['top'].set_visible(False)
  ax[1, 1].set_xlabel('Incident Severity', fontsize=18, fontweight='bold')
  ax[1, 1].set_ylabel('Count', fontsize=18, fontweight='bold')
  ax[1, 1].set_xticklabels(ax[1, 1].get_xticklabels(), rotation=10, ha='right', fontsize=14)
  ax[1, 1].legend(fontsize=14)

  # 6. fraud reported distribution
  ax[1, 2].pie(
      df["fraud_reported"].value_counts(normalize=True),
      labels=df['fraud_reported'].value_counts().index.astype(str),
      explode=[0.05] * len(df['fraud_reported'].value_counts()),
      autopct="%0.2f%%",
      colors=["#48d1cc", "#20b2aa", "#5f9ea0"],  # Turquoise shades
      textprops={'fontsize': 18, 'fontweight': 'bold'}  # Make pie chart text more readable
  )
  ax[1, 2].set_title('Distribution of Fraud Reported', fontsize=18, fontweight='bold')

  # 7. auto_make distribution hue=fraud_reported
  sns.countplot(
      x='auto_make',
      data=df, 
      hue='fraud_reported', 
      palette='Blues_r',  # Reversed blues
      ax=ax[2, 0]
  )
  ax[2, 0].set_title('Auto Make Distribution by Fraud Reported', fontsize=18, fontweight='bold')
  ax[2, 0].spines['right'].set_visible(False)
  ax[2, 0].spines['top'].set_visible(False)
  ax[2, 0].yaxis.set_ticks_position('left')
  ax[2, 0].xaxis.set_ticks_position('bottom')
  ax[2, 0].set_xticklabels(ax[2, 0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
  ax[2, 0].set_xlabel('Auto Make', fontsize=18, fontweight='bold')
  ax[2, 0].set_ylabel('Count', fontsize=18, fontweight='bold')
  ax[2, 0].legend(fontsize=14)

  # 8. incident_type distribution to fraud reported
  sns.countplot(
      x='incident_type',
      data=df, 
      hue='fraud_reported', 
      palette='GnBu', 
      ax=ax[2, 1]
  )
  ax[2, 1].set_title('Incident Type Distribution by Fraud Reported', fontsize=18, fontweight='bold')
  ax[2, 1].spines['right'].set_visible(False)
  ax[2, 1].spines['top'].set_visible(False)
  ax[2, 1].set_xlabel('Incident Type', fontsize=18, fontweight='bold')
  ax[2, 1].set_xticklabels(ax[2, 1].get_xticklabels(), rotation=45, ha='right', fontsize=14)
  ax[2, 1].set_ylabel('Count', fontsize=18, fontweight='bold')
  ax[2, 1].legend(fontsize=14)

  # 9. top hobbies distribution
  top_hobbies = (
      df[df['fraud_reported'] == 'Y']['insured_hobbies']
      .value_counts()
      .head(10)
      .index
  )
  filtered_df = df[df['insured_hobbies'].isin(top_hobbies)]
  sorted_hobbies = (
      filtered_df[filtered_df['fraud_reported'] == 'Y']['insured_hobbies']
      .value_counts()
      .index
  )
  filtered_df['insured_hobbies'] = pd.Categorical(
      filtered_df['insured_hobbies'], categories=sorted_hobbies, ordered=True
  )
  
  sns.countplot(
      y='insured_hobbies',
      data=filtered_df,
      hue='fraud_reported',
      palette='PuBuGn',  # Purple-Blue-Green palette
      ax=ax[2, 2]
  )
  ax[2, 2].set_title('Top 10 Insured Hobbies Distribution by Fraud Reported', fontsize=18, fontweight='bold')
  ax[2, 2].spines['right'].set_visible(False)
  ax[2, 2].spines['top'].set_visible(False)
  ax[2, 2].set_xlabel('Count', fontsize=18, fontweight='bold')
  ax[2, 2].set_ylabel('Insured Hobbies', fontsize=18, fontweight='bold')
  ax[2, 2].set_xticklabels(ax[2, 2].get_xticklabels(), rotation=90, fontsize=14)
  ax[2, 2].legend(fontsize=14)

  # Set overall figure style
  plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Blues(np.linspace(0, 1, 10)))
  
  # Show the plot in Streamlit
  st.pyplot(fig)

#####################################################################################################################
#####################################################################################################################