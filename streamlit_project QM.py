import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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

df=pd.read_csv("insurance_claims_geodata.csv",  parse_dates=["policy_bind_date", "incident_date"],index_col="policy_number")
tmp=df
df.rename(columns={'lng':'lon'}, inplace=True)
    
total_fraud_amount = df['total_claim_amount'][df['fraud_reported'] == 'Y'].sum()
total_amount=df['total_claim_amount'].sum()


st.title("Insurance Claim Prediction ")
st.sidebar.title("Table of contents")
pages=["Business problem", "Data Visualization","Statistical Methods", "Feature Engineering",  "Modelling", "Real-life Sample Application"]

st.sidebar.write("\n\n\n")
st.sidebar.write("\n\n\n")
st.sidebar.write("\n\nCreated by:")
st.sidebar.write("Bertrand Tcheuffa  \n Nathalie Mugrauer  \n Quy-Manh Jurca-Tsan \n")
st.sidebar.write("\n\n\n")
# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page=st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label
  
if page == pages[0] : 
    st.header("Business Problem Introduction")
    st.divider()
    st.markdown("## ❓ To fraud, or not to fraud: that is the question")
    st.markdown("### 📉 Insurance fraud undermines underwriting profitability, so prompt claim assessment is critical.")
    st.markdown("### ⚙️ Any suspicion of fraud must be underpinned by robust data analysis and pattern detection before opening an investigation.")
    st.markdown("### 💵 Efficient and accurate identification of fraudulent cases minimizes investigation costs and prevents unwarranted payouts.")

    st.divider()
    st.header("Economic KPI Dashboard")
    st.divider()
    col1,col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Total Claim Amount</h3>
            <h1 style="font-size:48px; color:#000000;">${total_amount:,.0f}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Total Fraud Claim Amount</h3>
            <h1 style="font-size:48px; color:red">${total_fraud_amount:,.0f}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col3:
        total_legit_amount=total_amount-total_fraud_amount
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Total Not Fraud Claim Amount</h3>
            <h1 style="font-size:48px; color:blue;">${total_legit_amount:,.0f}</h1>
        </div>
    """, unsafe_allow_html=True)

    bar_data = (
        df
        .groupby([df['incident_date'].dt.date, 'fraud_reported'])
        .size()
        .reset_index(name='count')
    )

    col1, col2,col3 = st.columns([1, 2, 1]) 

    with col1:
        st.subheader("Distribution Fraud")

        verteilung = df['fraud_reported'].value_counts(normalize=True) * 100
        verteilung = verteilung.reset_index()
        verteilung.columns = ['Fraud/No Fraud', 'Frequency']

        fig_pie = px.pie(
            verteilung,
            names='Fraud/No Fraud',
            values='Frequency',
            color='Fraud/No Fraud',
            #color_discrete_map={"Y": '#e74c3c', "N":  '#2e86de'},
            color_discrete_map={"Y": 'red', "N":  'blue'}
        )

        fig_pie.update_layout(showlegend=False)

        st.plotly_chart(fig_pie)

    with col2:
        st.subheader("Distribution of reported incidents")

        # Gruppieren nach Datum & Fraud
        bar_data = df.groupby([df['incident_date'].dt.date, 'fraud_reported']).size().reset_index(name='count')

        fig_bar = px.bar(
            bar_data,
            x='incident_date',
            y='count',
            color='fraud_reported',
            barmode='stack',
            labels={'incident_date': 'Date', 'count': 'Counts'},
            color_discrete_sequence=px.colors.qualitative.D3,
            height=500,
            color_discrete_map={"Y": 'red', "N":  'blue'},
        )
        fig_bar.update_layout(
        height=500,
        showlegend=False 
        )   
        st.plotly_chart(fig_bar, use_container_width=True, showlegend=False)
        
    with col3:
        st.subheader("Incident Locations")

        fig_map = px.scatter_map(
            df,
            lat="lat",
            lon="lon",
            color="fraud_reported",
            hover_name="fraud_reported",
            hover_data={"incident_date": True, "lat": False, "lon": False},
            color_discrete_sequence=px.colors.qualitative.D3,
            zoom=4,
            height=400,
            color_discrete_map={"Y": 'red', "N":  'blue'}
        )

        fig_map.update_layout(mapbox_style="open-street-map")
        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        st.plotly_chart(fig_map, use_container_width=True)

    
    st.divider()
    st.header("Overview of the dataset")
    st.dataframe(df.head(8))
    
    st.divider()
    st.header("Technical KPI Dashboard") 
    st.divider()
    col1,col2, col3, col4, col5, col6= st.columns(6)
  # The above code is creating a dashboard layout using Streamlit in Python. It consists of multiple
  # columns (`col1` to `col6`) where each column displays different information about a dataset
  # (`tmp`).
    with col1:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Table</h3>
            <h1 style="font-size:48px; color:#000000;">{1}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Samples</h3>
            <h1 style="font-size:48px; color:#000000;">{tmp.shape[0]}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Features</h3>
            <h1 style="font-size:48px; color:#000000;">{tmp.shape[1]-2}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">NaN-Values</h3>
            <h1 style="font-size:48px; color:#000000;">{tmp.isna().sum().sum()}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Num. Features</h3>
            <h1 style="font-size:48px; color:#000000;">{len(tmp.select_dtypes(['float64', 'int64']).columns)}</h1>
        </div>
    """, unsafe_allow_html=True)
    with col6:
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Cat. Features</h3>
            <h1 style="font-size:48px; color:#000000;">{len(tmp.select_dtypes(['object']).columns)}</h1>
        </div>
    """, unsafe_allow_html=True)

      
  
    
    st.divider()
    st.header("Objective")
    st.divider()
    st.markdown("### ➡️Our target variable, 'fraud_reported', indicates if a claim is fraudulent or not.")
    st.markdown("### ➡️Type of Machine Learning Problem: Supervised learning." )
    st.markdown("### ➡️Binary Classification Problem ")
    st.divider()
    st.markdown("### 🧠🔁📉Build accurate different machine learning models that spots fraudulent insurance claims.")  
    st.markdown("### 🚨🕵️‍♂️🚨Ensure model stability and reliability by achieving consistently prediction scores across different subsets and unseen data.")  
    st.markdown("### ⚡🕵️‍♀️📈Apply the full data science workflow - from understanding and preprocessing to feature engineering and model evaluation.") 
    st.write(" Considering the relative large dimensionality of our dataset, after carefully exploring the data , we will review each feature and its relevance to the target variable. Based on this, we will select the most relevant features for our model.")
#############################################################################################################################################
#############################################################################################################################################

if page == pages[1] : 
  st.write("### Data Visualization")
  st.markdown("### In this section, we will visualize the data to better understand the relationships between features and the target variable.")
  
  tab1, tab2, tab3, tab4 =st.tabs(["📊 Overview Chart","⚙️ Distribution Categorical", "🧮 Distribution Numerical", "📚 Heatmap"])
  
  with tab1:
    col_img, col_text = st.columns([2, 1])
    with col_img:
        st.header("📊 Overview Chart")
        uploaded_overview =   st.image("meine_figur.png")
    with col_text:
        st.header("Observation")
     
        st.markdown("### - We observe a distribution of gender that is fairly balanced.")
        st.markdown("### - The 'authorities_contacted' feature shows that most claims are reported to the police.") 
        st.markdown("### - While the 'incident_severity' feature indicates that most incidents are minor.") 
        st.markdown("### - We observe our target variable, 'fraud_reported', is imbalanced, with a higher number of non-fraudulent claims.")
        st.markdown("### - For the hobbies distribution we observe  that the hobbies chess and crossfit are the most common among fraudulent claims.")
        
  with tab2:
    col_img2, col_text2 = st.columns([2, 1])  
    with col_img2:
        st.header("⚙️ Distribution Categorical")
        st.image("meine_figur3.png")
    with col_text2:
        st.header("Observation")
        
        st.markdown("### - Approximately 9.1% of the NaN-values in the column ”authorities_contacted” needed to be handled.")
        st.markdown("### - In addition to missing values, we also encountered placeholder values represented as ❔ in three columns.") 
        st.markdown("### - ”police_report_available”, ”property_damage”, ”collision_type”") 
        
    
  with tab3:
    col_img3, col_text3 = st.columns([2, 1])  
    with col_img3:
        st.header("🧮 Distribution Numerical")
        st.image("meine_figur2.png")
    with col_text3:
        st.header("Observation")
        
        st.markdown("### - We observe a distribution of gender that is fairly balanced.")
        st.markdown("### - The 'authorities_contacted' feature shows that most claims are reported to the police.") 
        st.markdown("### - While the 'incident_severity' feature indicates that most incidents are minor.") 
        st.markdown("### - We observe our target variable, 'fraud_reported', is imbalanced, with a higher number of non-fraudulent claims.")
        st.markdown("### - For the hobbies distribution we observe  that the hobbies chess and crossfit are the most common among fraudulent claims.")
  with tab4:
    col_img4, col_text4 = st.columns([2, 1]) 
    with col_img4:
        st.header("📚 Heatmap")
        st.image("meine_figur4.png")
    with col_text4:
        st.header("Observation")
        
        st.markdown("### - In this visualization, lighter colors indicate strong correlations, while darker shades represent weaker or no correlation.")
        st.markdown("### - Two distinct clusters of high correlation are clearly visible.") 
        st.markdown("### - We observe strong correlations between different types of claims—particularly between ”vehicle_claim” and ”total_claim_amount”.") 
        st.markdown("### - The second cluster, located in the opposite corner of the heatmap, reveals a similarly strong correlation between ”months_as_customer” and ”age”.")

  
# #New blue-green color palette
#   colors = plt.cm.Blues([0.3, 0.5, 0.7, 0.9])
#   accent_colors = plt.cm.GnBu([0.4, 0.6, 0.8, 1.0])
  
#   # Increase global matplotlib font sizes
#   plt.rcParams.update({
#       'font.size': 16,
#       'axes.titlesize': 20,
#       'axes.labelsize': 20,
#       'xtick.labelsize': 17,
#       'ytick.labelsize': 17,
#       'legend.fontsize': 17,
#   })
  
#   #Create figure with better size for readability
#   fig, ax = plt.subplots(2, 3, figsize=(36, 20), dpi=200)
#   fig.tight_layout(pad=7)  # Increase padding between subplots

#   # 1. insured sex distribution
#   ax[0, 0].pie(
#       df['insured_sex'].value_counts(normalize=True),
#       labels=['Female', 'Male'],
#       explode=[0.05, 0],
#       autopct="%0.2f%%",
#       colors=["#4682B4", "#1E90FF"],  # Steel blue and dodger blue
#       textprops={'fontsize': 18, 'fontweight': 'bold'}  # Make pie chart text more readable
#   )
#   ax[0, 0].set_title('Distribution Male/Female', fontsize=18, fontweight='bold')

#   # 2. age distribution
# #   sns.histplot(
# #       ax=ax[0, 1],
# #       x="age",
# #       data=df,
# #       color="#0077b6",  # Dark blue
# #       stat="density",
# #       kde=True
# #   )
# #   ax[0, 1].set_title('Distribution of the Age', fontsize=18, fontweight='bold')
# #   ax[0, 1].spines['right'].set_visible(False)
# #   ax[0, 1].spines['top'].set_visible(False)
# #   ax[0, 1].set_xlabel('Age', fontsize=18, fontweight='bold')
# #   ax[0, 1].set_ylabel('Density', fontsize=18, fontweight='bold')

#   # 3.Authorities contacted by fraud reported
#   pd.crosstab(df['authorities_contacted'], df['fraud_reported']).plot(
#       kind='bar', 
#       stacked=True, 
#       ax=ax[0, 1], 
#       color=colors  # Blue colors
#   )
#   ax[0, 1].set_title('Authorities Contacted by Fraud Reported', fontsize=18, fontweight='bold')
#   ax[0, 1].spines['right'].set_visible(False)
#   ax[0, 1].spines['top'].set_visible(False)
#   ax[0, 1].set_xlabel('Authorities Contacted', fontsize=18, fontweight='bold')
#   ax[0, 1].set_ylabel('Count', fontsize=18, fontweight='bold')
#   ax[0, 1].set_xticklabels(ax[0, 1].get_xticklabels(), rotation=30, ha='right', fontsize=14)
#   ax[0, 1].legend(fontsize=14)

#   # 4. witness distribution for fraud reported = yes
# #   sns.countplot(
# #       x='witnesses', data=df[df['fraud_reported'] == 'Y'], palette='Blues', ax=ax[1, 0])
# #   ax[1, 0].set_title('Witnesses Distribution for Fraud Reported = Yes', fontsize=18, fontweight='bold')
# #   ax[1, 0].spines['right'].set_visible(False)
# #   ax[1, 0].spines['top'].set_visible(False)
# #   ax[1, 0].set_xlabel('Number of Witnesses', fontsize=18, fontweight='bold')
# #   ax[1, 0].set_ylabel('Count', fontsize=18, fontweight='bold')

#   # 5. Stacked bar chart incident_severity distribution to fraud reported
#   pd.crosstab(df['incident_severity'], df['fraud_reported']).plot(
#       kind='bar', 
#       stacked=True, 
#       ax=ax[0, 2], 
#       color=accent_colors  # Green-blue colors
#   )
#   ax[0, 2].set_title('Incident Severity Distribution by Fraud Reported', fontsize=18, fontweight='bold')
#   ax[0, 2].spines['right'].set_visible(False)
#   ax[0, 2].spines['top'].set_visible(False)
#   ax[0, 2].set_xlabel('Incident Severity', fontsize=18, fontweight='bold')
#   ax[0, 2].set_ylabel('Count', fontsize=18, fontweight='bold')
#   ax[0, 2].set_xticklabels(ax[0, 2].get_xticklabels(), rotation=10, ha='right', fontsize=14)
#   ax[0, 2].legend(fontsize=14)

#   # 6. fraud reported distribution
#   ax[1, 1].pie(
#       df["fraud_reported"].value_counts(normalize=True),
#       labels=df['fraud_reported'].value_counts().index.astype(str),
#       explode=[0.05] * len(df['fraud_reported'].value_counts()),
#       autopct="%0.2f%%",
#       colors=["#48d1cc", "#20b2aa", "#5f9ea0"],  # Turquoise shades
#       textprops={'fontsize': 18, 'fontweight': 'bold'}  # Make pie chart text more readable
#   )
#   ax[1, 1].set_title('Distribution of Fraud Reported', fontsize=18, fontweight='bold')

# #   # 7. auto_make distribution hue=fraud_reported
# #   sns.countplot(
# #       x='auto_make',
# #       data=df, 
# #       hue='fraud_reported', 
# #       palette='Blues_r',  # Reversed blues
# #       ax=ax[2, 0]
# #   )
# #   ax[2, 0].set_title('Auto Make Distribution by Fraud Reported', fontsize=18, fontweight='bold')
# #   ax[2, 0].spines['right'].set_visible(False)
# #   ax[2, 0].spines['top'].set_visible(False)
# #   ax[2, 0].yaxis.set_ticks_position('left')
# #   ax[2, 0].xaxis.set_ticks_position('bottom')
# #   ax[2, 0].set_xticklabels(ax[2, 0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
# #   ax[2, 0].set_xlabel('Auto Make', fontsize=18, fontweight='bold')
# #   ax[2, 0].set_ylabel('Count', fontsize=18, fontweight='bold')
# #   ax[2, 0].legend(fontsize=14)

#   # 8. incident_type distribution to fraud reported
#   sns.countplot(
#       x='incident_type',
#       data=df, 
#       hue='fraud_reported', 
#       palette='GnBu', 
#       ax=ax[1, 0]
#   )
#   ax[1, 0].set_title('Incident Type Distribution by Fraud Reported', fontsize=18, fontweight='bold')
#   ax[1, 0].spines['right'].set_visible(False)
#   ax[1, 0].spines['top'].set_visible(False)
#   ax[1, 0].set_xlabel('Incident Type', fontsize=18, fontweight='bold')
#   ax[1, 0].set_xticklabels(ax[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=14)
#   ax[1, 0].set_ylabel('Count', fontsize=18, fontweight='bold')
#   ax[1, 0].legend(fontsize=14)

#   # 9. top hobbies distribution
#   top_hobbies = (
#       df[df['fraud_reported'] == 'Y']['insured_hobbies']
#       .value_counts()
#       .head(10)
#       .index
#   )
#   filtered_df = df[df['insured_hobbies'].isin(top_hobbies)]
#   sorted_hobbies = (
#       filtered_df[filtered_df['fraud_reported'] == 'Y']['insured_hobbies']
#       .value_counts()
#       .index
#   )
#   filtered_df['insured_hobbies'] = pd.Categorical(
#       filtered_df['insured_hobbies'], categories=sorted_hobbies, ordered=True
#   )
  
#   sns.countplot(
#       y='insured_hobbies',
#       data=filtered_df,
#       hue='fraud_reported',
#       palette='PuBuGn',  # Purple-Blue-Green palette
#       ax=ax[1, 2]
#   )
#   ax[1, 2].set_title('Top 10 Insured Hobbies Distribution by Fraud Reported', fontsize=18, fontweight='bold')
#   ax[1, 2].spines['right'].set_visible(False)
#   ax[1, 2].spines['top'].set_visible(False)
#   ax[1, 2].set_xlabel('Count', fontsize=18, fontweight='bold')
#   ax[1, 2].set_ylabel('Insured Hobbies', fontsize=18, fontweight='bold')
#   ax[1, 2].set_xticklabels(ax[1, 2].get_xticklabels(), rotation=90, fontsize=14)
#   ax[1, 2].legend(fontsize=14)

#   # Set overall figure style
#   plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Blues(np.linspace(0, 1, 10)))
  
#   fig.savefig('meine_figur.png',              # Dateiname
#             format='png',                   # Dateiformat (optional, wird aber empfohlen)
#             dpi=300,                        # Auflösung in dots per inch
#             bbox_inches='tight',           # Ränder zuschneiden
#             transparent=False)        
  
#   #Show the plot in Streamlit
#   st.pyplot(fig)

#####################################################################################################################
#####################################################################################################################
if page == pages[4] :
    st.header("Modelling")
    st.divider()
    st.markdown("### In this section, we will present the different steps we achieved to identify the best suitable ML models for the fraud prediction task.")
    st.markdown("### We begin by outlining the preliminary steps, proceed to detail the strategy employed for model selection, and conclude with a concise summary of the modelling process.")
    st.divider()
    st.header("Preliminary steps:")
    
    with st.expander("Show Preliminary steps"):
        tab1, tab2, tab3, tab4, tab5, tab6 =st.tabs(["✂️ Cardinality Reduction","🎯 Target Variable Encoding", "➗ Data Set Train/Test Splitting", " ⚖️ Feature Normalisation/Encoding", " ❓ Missing Value Handling", "📏 Evaluation Metric Definition"])
    
        with tab1:
    
            st.markdown("### The statistical tests conducted identified the following features, presented in the table below, as having the strongest influence on the target variable: ")
            df = df[['insured_hobbies', 'incident_type', 'collision_type',
        'incident_severity', 'authorities_contacted', 'incident_state',
        'property_damage', 'vehicle_claim'
        ]]
        
            st.dataframe(df.head())

        with tab2:
            st.markdown("### ➡️ Feature encoding transforms raw data into numerical values suitable for input to most machine learning algorithms")
            st.markdown("### ➡️ Label Encoding was used to mark the fraud cases as '1' and the non-fraud cases as '0'" )   

        with tab3:
            col_img3, col_text3 = st.columns([2, 1])  
            with col_img3:
                labels = ['Train', 'Test']
                sizes = [80, 20]
                colors = ['#66b3ff', '#ff9999']

                def plot_pie():
                    fig, ax = plt.subplots(figsize=(1.5, 1.5))  # Smaller plot size
                    wedges, texts, autotexts = ax.pie(sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90,
                    textprops={'fontsize': 4} )
                    ax.axis('equal')
                    st.pyplot(fig)

                plot_pie()

            with col_text3:
                st.markdown("### ➡️ The train / test split was performed before the feature encoding and normalization ")
                st.markdown("### ➡️ It helps avoiding data leakage during the the transformation ")
                st.markdown("### ➡️ Missing value handling, feature normalization and encoding were done on each set separately ")

        with tab4:
            st.header("Categorical features encoding with the OneHotEncoder")
            col_table4, col_text4 = st.columns([2, 1])  
            with col_table4:
                st.image("Categorical variables.png")
            with col_text4:
                st.markdown("### ➡️ Handling 'incident_severity' as ordinal variable worsen the f1-score. ")
                st.markdown("### ➡️ 'incident_severity' and all other categorical features were handled as nominal variables and therefore encoded with the OneHotEncoder  ")
        
            st.header("Numerical features normalization with the MinMaxScaler")
            col_table5, col_text5 = st.columns([2, 1])  
            with col_table5:
                st.image("Numerical variable.png")
            with col_text5:
                st.markdown("### ➡️ 'vehicule_claim' was normalized with the MinMaxScaler, due to it's non-normal distribution ")
                st.markdown("### ➡️ 'vehicule_claim was the most relevant numerical feature selected for the task  ")
    
        with tab5:
            st.markdown("### ➡️ 9.1 % of the values in 'authorities_contacted' were identified as missing values ")
            st.markdown("### ➡️ They were replaced with the mode, using the SimpleImputer  ")
    
        with tab6:
            st.markdown("### ➡️ Fraud cases that are misclassified can cause a big financial damage by claim settlement.")
            st.markdown("### ➡️ It is therefore important to minimize the false negative rate ")
            st.markdown("### ➡️ Non-fraud cases, that are classified as fraud (False Positive) are also to be minimized ")
            st.markdown("### ➡️ Wrongly assigned penalties due to a misprediction may result in costly lawsuits against the company ")
            st.markdown("### ➡️ The f1-score, with a good balance between precision and recall, was chosen as the metric  for the task \n")
            st.markdown(" \n")
            col1, col2, col3 = st.columns([1, 2, 1])  # 2 is wider, acts as center
            with col2:
                st.image("F1-score definition.png", use_container_width=True)

    st.divider()
    st.header("Our strategy:")
    with st.expander("Show our strategy"):
        st.markdown("### 1️⃣ Modelisation with imbalanced data")
        st.markdown("### 2️⃣ Resampling strategy selection (RandomOverSampler/SMOTETomek)")
        st.markdown("### 3️⃣ Modelisation with resampled data")
        st.markdown("### 4️⃣ Hyperparameter optimization (GridSearch/ensemble methods) ")
        st.markdown("### 5️⃣ Model robustness testing (Stratified 50-Fold cross validation)")
        #st.markdown("### 6️⃣ Best models selection ")
    
    st.divider()
    st.header("Our results:")
    with st.expander("Show our results"):
        st.markdown("### 🏆 The best scorer models")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1️⃣ - Linear Discriminant Analysis (LDA)")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2️⃣ - Logistic Regression")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3️⃣ - AdaBoost")
        st.markdown("### 🔝 The LDA most influent features")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 1️⃣ - 'insured_hobbies_chess' ")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2️⃣ - 'insured_hobbies_cross_fit'")
        st.markdown("### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3️⃣ - 'incident_severity_Major_Damage'")

        tab7, tab8 =st.tabs(["🏆 The Best Scorer Models ","🔝 The LDA Most Influent Features"])
    
        with tab7:
            col_img7, col_text7 = st.columns([2, 1])
            with col_img7:
                st.image("Best scorer models.png")
            with col_text7:
                st.markdown("### - The Linear Discriminant Analysis model, when combined with SMOTETomek resampling, achieved one of the highest performance scores ")           
                st.markdown("### - The variation in weighted f1-score across models remained minimal ")
                st.markdown("### - Analysis of the confusion matrices enabled a better differentiation among model performances.")
        with tab8:
                st.header("Hypotheses:")
                st.markdown("### - People committing fraud may misreport hobbies (Chess or cross fit) to appear more trustworthy")           
                st.markdown("### - The LDA model may be picking up spurious correlations due to small sample size (1000 samples) ")
                st.markdown("### - Fraudulent profiles might exhibit non-representative hobby distributions compared to genuine profiles")
    st.divider()