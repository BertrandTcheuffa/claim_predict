import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from PIL import Image
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import chi2_contingency, ttest_ind, ks_2samp, mannwhitneyu

# Load the trained model and preprocessors
try:
    lda_model = joblib.load('lda_model.pkl')
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
    categorical_imputer = joblib.load('categorical_imputer.joblib')
except FileNotFoundError as e:
    st.error(f"Error loading model or preprocessor: {e}. Please ensure 'lda_model.pkl', 'onehot_encoder.joblib', 'scaler.joblib', and 'categorical_imputer.joblib' are in the same directory as your script.")
    st.stop()

# Define the categorical and numerical columns used for training
categorical_cols_trained = ['insured_hobbies', 'incident_type', 'collision_type', 'authorities_contacted', 'incident_state', 'incident_severity', 'property_damage']
numerical_cols_trained = ['vehicle_claim']
all_hobbies_trained = ['sleeping', 'reading', 'board-games', 'bungie-jumping', 'base-jumping', 'skydiving', 'golf', 'hiking', 'camping', 'dancing', 'kayaking', 'yachting', 'paintball', 'movies', 'video-games', 'кроссфит', 'караоке']

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

# Load and prepare data
df = pd.read_csv("insurance_claims_geodata.csv")

# Convert date columns to datetime if they exist
try:
    df['incident_date'] = pd.to_datetime(df['incident_date'])
except Exception as e:
    st.warning(f"Warning: Could not convert incident_date to datetime. Error: {e}")

# Create tmp dataframe for dashboard
tmp = df.copy()

# Calculate economic KPI values
try:
    # Sum up the total claim amounts
    total_amount = df['total_claim_amount'].sum() if 'total_claim_amount' in df.columns else 0
    # Sum up fraudulent claim amounts
    total_fraud_amount = df[df['fraud_reported'] == 'Y']['total_claim_amount'].sum() if 'total_claim_amount' in df.columns and 'fraud_reported' in df.columns else 0
except Exception as e:
    total_amount = 0
    total_fraud_amount = 0
    st.warning(f"Warning: Could not calculate claim amounts. Error: {e}")

st.title("Insurance Claim Prediction ")
st.sidebar.title("Insurance Claim Prediction")
pages = ["Business problem", "Data Visualization", "Statistical Methods", "Feature Engineering", "Modelling", "Sample Application", "Conclusion", "Appendix - Extras"]

# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page = st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label

# Inject CSS to push the footer to the bottom
st.markdown(
    """
    <style>
        .sidebar {
            display: flex;
            flex-direction: column;
        }
        .sidebar > div:last-child {
            margin-top: auto;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Container for the bottom elements (ensure this is the LAST thing added to the sidebar)
with st.sidebar.container():
    st.markdown("---")  # Optional separator
    st.write("Created by:")
    st.write("Bertrand Tcheuffa")
    st.write("Nathalie Mugrauer")
    st.write("Quy-Manh Jurca-Tsan")


####################################################################################################################################################################
####################################################################################################################################################################
# Main content area based on the selected page
if page == "Business problem":
    st.header("Business Problem Introduction")
    st.divider()
    col1, col2 = st.columns([1,8])
    with col1:
        try:
            img = Image.open("shakespeare.jpg")
            img_klein = img.resize((200, 200))
            st.image(img_klein)
        except FileNotFoundError:
            st.warning("Image file 'shakespeare.jpg' not found")
    with col2:
        st.markdown("## To fraud, or not to fraud: that is the question")
        st.markdown("##### 📉 Insurance fraud undermines underwriting profitability, so prompt claim assessment is critical.")
        st.markdown("##### ⚙️ Any suspicion of fraud must be underpinned by robust data analysis and pattern detection before opening an investigation.")
        st.markdown("##### 💵 Efficient and accurate identification of fraudulent cases minimizes investigation costs and prevents unwarranted payouts.")
    
    st.divider()
    st.header("Technical KPI Dashboard") 
    st.divider()
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Create technical KPI dashboard
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
    st.header("Economic KPI Dashboard")
    st.divider()
    col1, col2, col3 = st.columns(3)
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
        total_legit_amount = total_amount - total_fraud_amount
        st.markdown(f"""
        <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
            <h3 style="margin-bottom:5px;">Total Not Fraud Claim Amount</h3>
            <h1 style="font-size:48px; color:blue;">${total_legit_amount:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)

    # Create data for plots
    try:
        bar_data = (
            df
            .groupby([df['incident_date'].dt.date, 'fraud_reported'])
            .size()
            .reset_index(name='count')
        )

        col1, col2, col3 = st.columns([1, 2, 1]) 

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
                color_discrete_map={"Y": 'red', "N": 'blue'}
            )

            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie)

        with col2:
            st.subheader("Distribution of reported incidents")

            # Group by date & fraud
            fig_bar = px.bar(
                bar_data,
                x='incident_date',
                y='count',
                color='fraud_reported',
                barmode='stack',
                labels={'incident_date': 'Date', 'count': 'Counts'},
                color_discrete_sequence=px.colors.qualitative.D3,
                height=500,
                color_discrete_map={"Y": 'red', "N": 'blue'},
            )
            fig_bar.update_layout(
                height=500,
                showlegend=False 
            )   
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col3:
            st.subheader("Incident Locations")

            # Check if lat and lon columns exist in the dataset
            if 'lat' in df.columns and 'lng' in df.columns:
                fig_map = px.scatter_map(
                    df,
                    lat="lat",
                    lon="lng",
                    color="fraud_reported",
                    hover_name="fraud_reported",
                    hover_data={"incident_date": True, "lat": False, "lng": False},
                    color_discrete_sequence=px.colors.qualitative.D3,
                    zoom=4,
                    height=400,
                    color_discrete_map={"Y": 'red', "N": 'blue'}
                )

                fig_map.update_layout(mapbox_style="open-street-map")
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Location data (lat, lng) not found in dataset")
    except Exception as e:
        st.error(f"Error creating plots: {e}")
    
    st.divider()
    st.header("Overview of the dataset")
    st.dataframe(df.head(8))
    
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
####################################################################################################################################################################
####################################################################################################################################################################
elif page == "Data Visualization":
    st.header("Data Visualization")
    st.markdown("### In this section, we will visualize the data to better understand the relationships between features and the target variable.")
  
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview Chart", "⚙️ Distribution Categorical", "🧮 Distribution Numerical", "📚 Heatmap"])
  
    with tab1:
        col_img, col_text = st.columns([2, 1])
        with col_img:
            st.header("📊 Overview Chart")
            try:
                uploaded_overview = st.image("meine_figur.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur.png' not found")
        with col_text:
            st.header("Observation")
        
            st.markdown("#### - We observe a distribution of gender that is fairly balanced.")
            st.markdown("#### - The `authorities_contacted` feature shows that most claims are reported to the police.") 
            st.markdown("#### - While the `incident_severity` feature indicates that most incidents are minor.") 
            st.markdown("#### - We observe our target variable, `fraud_reported`, is imbalanced, with a higher number of non-fraudulent claims.")
            st.markdown("#### - For the hobbies distribution we observe that the `hobbies chess` and `crossfit` are the most common among fraudulent claims.")
        
    with tab2:
        col_img2, col_text2 = st.columns([2, 1])  
        with col_img2:
            st.header("⚙️ Distribution Categorical")
            try:
                st.image("meine_figur3.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur3.png' not found")
        with col_text2:
            st.header("Observation")
            
            st.markdown("#### - Approximately 9.1% of the NaN-values in the column `authorities_contacted` needed to be handled.")
            st.markdown("#### - In addition to missing values, we also encountered placeholder values represented as ❔ in three columns.") 
            st.markdown("#### - `police_report_available`, `property_damage`, `collision_type`") 
        
    
    with tab3:
        col_img3, col_text3 = st.columns([2, 1])  
        with col_img3:
            st.header("🧮 Distribution Numerical")
            try:
                st.image("meine_figur2.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur2.png' not found")
        with col_text3:
            st.header("Observation")
            
            st.markdown("#### - We found that only the feature `policy_annual_premium` showed approximate normal distributions.")
            st.markdown("#### - All other numerical features did not follow a normal distribution.")

    with tab4:
        col_img4, col_text4 = st.columns([2, 1]) 
        with col_img4:
            st.header("📚 Heatmap")
            try:
                st.image("meine_figur4.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur4.png' not found")
        with col_text4:
            st.header("Observation")
            
            st.markdown("#### - In this visualization, dark colors indicate strong correlations, while lighter shades represent weaker or no correlation.")
            st.markdown("#### - Two distinct clusters of high correlation are clearly visible.") 
            st.markdown("#### - We observe strong correlations between different types of claims—particularly between `vehicle_claim` and `total_claim_amount`.") 
            st.markdown("#### - The second cluster, located in the opposite corner of the heatmap, reveals a similarly strong correlation between `months_as_customer` and `age`.")
####################################################################################################################################################################
####################################################################################################################################################################
elif page == "Statistical Methods":
    st.header("Data Exploration:")
    st.markdown("##### Before proceeding with statistical analyses, we should establish a strategy for handling missing (NaN) values. \
                In addition to missing values, we also encountered placeholder values represented as `?` in three columns. \
                This character may either be treated as a missing value and replaced accordingly, or considered as a separate category.\
                In addition, it is important to examine whether the occurrence of `?` shows any systematic relationship with fraudulent activity.")
    
    tab1, tab2, tab3 = st.tabs(["NaN - authorities_contacted", "❔ - police_report_available, property_damage", "❔ - collision_type"])
    
    with tab1:
        st.subheader("NaN - authorities_contacted")
        col_img, col_text = st.columns([2, 1])
        with col_img:
            try:
                st.image("meine_figur5.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur5.png' not found")
        with col_text:
            st.header("Observation")
         
            st.markdown("#### - We do not want to remove these values, as they represent almost 10% of the data and our overall dataset is relatively small.")
            st.markdown("#### - Deleting them could lead to a loss of important information.") 
            st.markdown("#### - The distribution of fraud and non-fraud cases shows that the missing value is not necessarily an indicator of fraud.") 
            st.markdown("#### - For example, when looking at the crosstab between `authorities_contacted` and `incident_type`, it becomes clear that missing values only appear in cases of `Parked Car` and `Vehicle Theft`.")
            st.markdown("#### - Since `Police` is also the most frequently occurring category, we decided to use it as a replacement value during encoding.")
            
    with tab2:
        st.subheader("❔ - police_report_available, property_damage")
        col_img2, col_text2 = st.columns([2, 1])  
        with col_img2:
            try:
                st.image("meine_figur6.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur6.png' not found")
        with col_text2:
            st.header("Observation")
            
            st.markdown("#### - In `police_report_available` and `property_damage`, these ambiguous values may indicate potential fraud, as the information could have been intentionally withheld.")
            st.markdown("#### - To avoid any misleading interpretations or data leakage, these entries were not imputed with common values from the distribution.") 
            st.markdown("#### - But instead were explicitly replaced with a new category called 'Unknown'.") 
            
        
    with tab3:
        st.subheader("❔ - collision_type")
        col_img3, col_text3 = st.columns([2, 1])  
        with col_img3:
            try:
                st.image("meine_figur7.png")
            except FileNotFoundError:
                st.warning("Image file 'meine_figur7.png' not found")
        with col_text3:
            st.header("Observation")
            
            st.markdown("#### - The `?`-values in the `collision_type` column were observed only in relation to the incident types `Vehicle Theft` and `Parked Car`.")
            st.markdown("#### - Given the context, this can be considered as a separate category.")
            st.markdown("#### - Therefore, the `?` entries were replaced with `No Collision` to accurately reflect the nature of these incidents.")
    
    st.divider()
    st.header("Statistical Methods") 
    st.markdown("##### Now we want to check how statistically relevant our features are to the target variable. We will use the tests we learned for numerical and categorical features.")
    
    tab1, tab2 = st.tabs(["Categorical Features", "Numerical Features"])
    
    with tab1:    
        try:
            chi2test_results = []
            for col in df.columns:
                if df[col].dtype == 'object' and col != "fraud_reported":
                    contingency_table = pd.crosstab(df[col], df['fraud_reported'])
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    
                    n = contingency_table.sum().sum()
                    k = min(contingency_table.shape)
                    
                    # Cramér's V
                    cramer_v = np.sqrt(chi2 / (n * (k - 1))) if k > 1 else np.nan
                    
                    chi2test_results.append({
                        'feature': col,
                        'chi2': chi2,
                        'p': p,
                        'dof': dof,
                        "cramer_v": cramer_v
                    })

            chi2test_results = pd.DataFrame(chi2test_results)
            chi2test_results = chi2test_results.sort_values('p')
            st.dataframe(chi2test_results)
        except Exception as e:
            st.error(f"Error calculating chi2 statistics: {e}")
    
    with tab2:
        try:
            results = []
            df["fraud"] = df['fraud_reported'].apply(lambda x: 1 if str(x) == "Y" else 0)
            col_quan = df.select_dtypes(['int64', 'float64']).columns
            for col in col_quan:
                if col != 'fraud':
                    group0 = df[df['fraud'] == 0][col].dropna()
                    group1 = df[df['fraud'] == 1][col].dropna()

                    # t-test 
                    t_stat, p_ttest = ttest_ind(group0, group1, equal_var=False)

                    # Kolmogorov-Smirnov-test
                    ks_stat, p_ks = ks_2samp(group0, group1)

                    # Mann-Whitney-U-test
                    mw_stat, p_mw = mannwhitneyu(group0, group1, alternative='two-sided')

                
                    results.append({
                        'feature': col,
                        'p_ttest': p_ttest,
                        'p_ks_2samp': p_ks,
                        'p_mannwhitneyu': p_mw
                    })

            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by='p_ks_2samp')
            
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error calculating statistical tests: {e}")
            
    st.markdown("##### Based on the statistical tests, we decided to keep the features that are statistically relevant:")
    st.markdown("##### incident_severity, collision_type, incident_type, incident_state, property_damage, authorities_contacted, vehicle_claim")
    st.divider()
####################################################################################################################################################################
####################################################################################################################################################################
elif page == "Feature Engineering":
    st.header("Feature Engineering")
    st.write("In the previous slides we have gained a solid understanding of our data. Based on statistical methods we have reduced the dimensionality of our dataset from 40 explanatory features to 8  **(7 categorical and 1 quantitative)**.")
    try:
        st.image("df_info.png")
    except FileNotFoundError:
        st.warning("Image file 'df_info.png' not found")
        
    # Create the data for the table
    data = {
        "Feature": ["incident_severity", "insured_hobbies", "collision_type", "incident_type", "incident_state", "property_damage", "authorities_contacted"],
        "Nominal / Ordinal": ["nominal / ordinal", "nominal", "nominal / ordinal", "nominal / ordinal", "nominal", "nominal", "nominal"],
        "Encoder": ["OneHotEncoder / OrdinalEncoder", "OneHotEncoder", "OneHotEncoder / OrdinalEncoder", "OneHotEncoder / OrdinalEncoder", "OneHotEncoder", "OneHotEncoder / Ordinal Encoder", "OneHotEncoder"]
    }
    st.subheader("Categorical Features")
    st.write("Consequently, we reviewed these features and decided if and which transformations are needed. We begin with the **categorical features:**")
    df2 = pd.DataFrame(data)
    st.dataframe(df2)

    # Load the X_train.csv file here
    try:
        X_train = pd.read_csv("X_train.csv")
        st.write("The shape of the transformed TRAIN Set is:", X_train.shape)
        st.dataframe(X_train.head(3))
    except FileNotFoundError:
        st.error("Error: X_train.csv not found. Please ensure it is in the same directory as your script.")

    st.subheader("Quantitative Features")
    st.write("The only quantitative feature is **'vehicle_claim'**.  We observe a wide range among the min and max values.  We decided to scale this feature using the **StandardScaler()**.")
    st.write("**Dataframe before scaling:**")
    vehicle_claim_desc = df['vehicle_claim'].describe()[['count', 'mean', 'min', 'max']]
    vehicle_claim_desc_df = vehicle_claim_desc.to_frame().T
    st.dataframe(vehicle_claim_desc_df)
    st.write("**Train set after scaling:**")
    try:
        if 'X_train' in locals(): # Check if X_train was loaded successfully
            vehicle_claim_desc_X_train = X_train['vehicle_claim'].describe()[['count', 'mean', 'min', 'max']].to_frame().T
            st.dataframe(vehicle_claim_desc_X_train)
    except KeyError:
        st.warning("Warning: 'vehicle_claim' column not found in X_train.")
#####################################################################################
#####################################################################################
elif page == "Modelling":
    st.header("Modelling")
    st.divider()
    st.markdown("### In this section, we will present the different steps we achieved to identify the best suitable ML models for the fraud prediction task.")
    st.markdown("### We begin by outlining the preliminary steps, proceed to detail the strategy employed for model selection, and conclude with a concise summary of the modelling process.")
    st.divider()
    st.header("Preliminary steps:")
    
    with st.expander("Show Preliminary steps"):
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["✂️ Cardinality Reduction","🎯 Target Variable Encoding", "➗ Data Set Train/Test Splitting", " ⚖️ Feature Normalisation/Encoding", " ❓ Missing Value Handling", "📏 Evaluation Metric Definition"])
        
        with tab1:
            st.markdown("### The statistical tests conducted identified the following features, presented in the table below, as having the strongest influence on the target variable: ")
            df_model = df[['insured_hobbies', 'incident_type', 'collision_type',
                'incident_severity', 'authorities_contacted', 'incident_state',
                'property_damage', 'vehicle_claim'
            ]]
        
            st.dataframe(df_model.head())

        with tab2:
            st.markdown("### ➡️ Label Encoding was used to mark the fraud cases as '1' and the non-fraud cases as '0'" )   
            st.markdown("### ➡️ Feature encoding transforms raw data into numerical values suitable for input to most machine learning algorithms")
            
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
                    textprops={'fontsize': 4})
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
                try:
                    st.image("Categorical_variables.png")
                except FileNotFoundError:
                    st.warning("Image file 'Categorical_variables.png' not found")
            with col_text4:
                st.markdown("### ➡️ Handling 'incident_severity' as ordinal variable worsen the f1-score. ")
                st.markdown("### ➡️ 'incident_severity' and all other categorical features were handled as nominal variables and therefore encoded with the OneHotEncoder  ")
        
            st.header("Numerical features normalization with the MinMaxScaler")
            col_table5, col_text5 = st.columns([2, 1])  
            with col_table5:
                try:
                    st.image("Numerical_variable.png")
                except FileNotFoundError:
                    st.warning("Image file 'Numerical_variable.png' not found")
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
                try:
                    st.image("F1_score_definition.png", use_container_width=True)
                except FileNotFoundError:
                    st.warning("Image file 'F1_score_definition.png' not found")

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

        tab7, tab8 = st.tabs(["🏆 The Best Scorer Models ","🔝 The LDA Most Influent Features"])
        
        with tab7:
            col_img7, col_text7 = st.columns([2, 1])
            with col_img7:
                try:
                    st.image("Best_scorer_models.png")
                except FileNotFoundError:
                    st.warning("Image file 'Best_scorer_models.png' not found")
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
    st.header("🔍 Our Model Evaluator:")
    with st.expander("Show"):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        # Set Streamlit page configuration
        #st.set_page_config(page_title="Model Evaluation App", layout="centered")

        #st.title("🔍 Fraud Detection Model Evaluator")

        # Sidebar
        st.header("Model Selection")
        model_name = st.selectbox("Choose a model:", ("Linear Discriminant Analysis", "Logistic Regression", "Ridge Classifier"))

        # Map selection to filenames
        model_map = {
                        "Linear Discriminant Analysis": "lda_model.pkl",
                        "Logistic Regression": "logReg_model.pkl",
                        "Ridge Classifier": "RidgeCl_model.pkl"
                    }

        # Prediction Button
        if st.button("🔎 Predict"):
            with st.spinner("Loading model and making predictions..."):
                try:
                    # Load model
                    model_path = os.path.join("models", model_map[model_name])
                    model = joblib.load(model_path)
                    logging.info(f"{model_name} loaded successfully.")

                    # Load test data
                    X_test = pd.read_csv("X_test.csv")
                    y_test = pd.read_csv("y_test.csv").squeeze()

                    # Predict
                    y_pred = model.predict(X_test)

                    # Compute F1 score
                    score = f1_score(y_test, y_pred, average="weighted")
                    st.success(f"✅ Weighted F1-score: **{score:.4f}**")

                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    st.markdown("### 📊 Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix - {model_name}")
                    st.pyplot(fig)

                except FileNotFoundError as e:
                    st.error(f"❌ File not found: {e}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")






#####################################################################################################################################################
#####################################################################################################################################################
elif page == "Sample Application":
    st.header("Sample Application")
    st.subheader("Real-life Sample Application: Insurance Fraud Detection")
    st.markdown("<p style='color:#0077b6; font-size:18px;'>This application demonstrates how the machine learning model could be used in a real-world scenario to assess the risk of fraud.</p>", unsafe_allow_html=True)

    def file_claim_survey():
        st.write("Please provide details about the incident to process your claim.")
        
        # We need to include ALL possible hobby values that were in the training data
        # Add any missing hobbies that caused errors
        all_possible_hobbies = ['sleeping', 'reading', 'board-games', 'bungie-jumping', 'base-jumping', 
                          'skydiving', 'golf', 'hiking', 'camping', 'dancing', 'kayaking', 'yachting', 
                          'paintball', 'movies', 'video-games', 'кроссфит', 'караоке',
                          'chess', 'cross-fit', 'exercise', 'polo', 'basketball']
        
        # Create form with two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            incident_severity = st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"])
            incident_state = st.selectbox("Incident State", ['SC', 'VA', 'NC', 'WV']) 
            insured_hobbies = st.selectbox("Insured Hobby", all_possible_hobbies)
            collision_type = st.selectbox("Collision Type", ["No Collision", "Side Collision", "Rear Collision", "Front Collision", "Other"])
        
        with col2:
            property_damage = st.radio("Property Damage", ["NO", "YES", "Unknown"])
            authorities_contacted = st.selectbox("Authorities Contacted", ['Fire', 'Police', 'None', 'Other', 'Ambulance'])
            vehicle_claim = st.number_input("Vehicle Claim Amount", min_value=0.0, format="%.2f")
            incident_type = st.selectbox("Incident Type", ['Parked Car', 'Single Vehicle Collision', 'Vehicle Theft', 'Multi Vehicle Collision'])
        
        # Store user inputs for prediction
        user_input = {
            "incident_severity": incident_severity,
            "incident_state": incident_state,
            "insured_hobbies": insured_hobbies,
            "collision_type": collision_type,
            "property_damage": property_damage,
            "authorities_contacted": authorities_contacted,
            "vehicle_claim": vehicle_claim,
            "incident_type": incident_type,
        }
        
        # Create DataFrame from user input
        user_input_df = pd.DataFrame([user_input])
        
        # Add a predict button with prominent styling
        predict_button = st.button("Predict Fraud Risk", type="primary")
        
        if predict_button:
            try:
                # Get all feature names from the one-hot encoder
                feature_names = []
                
                # First, add the numerical feature
                feature_names.append('vehicle_claim')
                
                # Add one-hot encoded feature names
                cat_feature_names = []
                for i, feature in enumerate(categorical_cols_trained):
                    cat_values = onehot_encoder.categories_[i]
                    for val in cat_values:
                        cat_feature_names.append(f"{feature}_{val}")
                
                # Create a DataFrame with all expected features, initialized to 0
                processed_input = pd.DataFrame(0, index=[0], columns=feature_names + cat_feature_names)
                
                # Set the scaled vehicle_claim value
                processed_input['vehicle_claim'] = scaler.transform([[user_input_df['vehicle_claim'].iloc[0]]])[0][0]
                
                # Set the one-hot encoded categorical features
                for feature in categorical_cols_trained:
                    value = user_input_df[feature].iloc[0]
                    col_name = f"{feature}_{value}"
                    if col_name in processed_input.columns:
                        processed_input.loc[0, col_name] = 1
                
                # Make prediction
                prediction = lda_model.predict(processed_input)
                
                # Show results with nice formatting
                st.write("---")
                
                if prediction[0] == 1:
                    st.error("⚠️ ALERT: Potential Fraudulent Claim Detected")
                    st.warning("""
                    Our system has flagged this claim for further review. This does not necessarily mean fraud has occurred, 
                    but additional verification steps will be required.
                    
                    **Next steps:** An investigator will contact you within 24-48 hours to gather more information.
                    """)
                    
                    # Create explanation for fraud prediction
                    st.subheader("Why was this claim flagged?")
                    
                    # Check for high-risk factors
                    risk_factors = []
                    
                    # Check hobby factor
                    if insured_hobbies in ['chess', 'cross-fit', 'кроссфит']:
                        risk_factors.append(f"**Hobby Profile:** The hobby '{insured_hobbies}' has a statistically significant correlation with fraudulent claims in our historical data.")
                    
                    # Check severity vs claim amount pattern
                    if incident_severity in ["Minor Damage", "Trivial Damage"] and vehicle_claim > 15000:
                        risk_factors.append(f"**Claim Amount Discrepancy:** The claim amount (${vehicle_claim:,.2f}) appears disproportionately high for {incident_severity}.")
                    
                    # Check incident type patterns
                    if incident_type == "Single Vehicle Collision" and collision_type == "Side Collision":
                        risk_factors.append("**Incident Inconsistency:** The combination of single-vehicle collision with side collision is unusual and may require verification.")
                    
                    # Check incident state factor
                    if incident_state == "SC" and incident_severity == "Major Damage":
                        risk_factors.append("**Geographic Pattern:** Claims with major damage in this state have shown higher fraud rates historically.")
                    
                    # Check for property damage with no collision
                    if collision_type == "No Collision" and property_damage == "YES":
                        risk_factors.append("**Logical Inconsistency:** Property damage reported with no collision event requires verification.")
                    
                    # Display the risk factors
                    if risk_factors:
                        for factor in risk_factors:
                            st.markdown(factor)
                    else:
                        st.markdown("**Complex Pattern Detection:** The model has identified patterns in the combination of features that match previously identified fraudulent claims. The specific pattern is complex and involves multiple feature interactions.")
                    
                    # Disclaimer
                    st.caption("Note: These are potential risk factors and do not constitute proof of fraud. Our model uses statistical patterns to identify claims requiring further review.")
                    
                else:
                    st.success("✅ Claim Assessment: Standard Risk Level")
                    st.info("""
                    Based on our analysis, this claim has been approved for standard processing.
                    
                    **Next steps:** Your claim has been entered into our system and will be processed within our standard timeframe.
                    """)
                
                # Display claim details in a clean format
                st.subheader("Claim Details Summary")
                formatted_claim = {
                    "Incident Type": user_input["incident_type"],
                    "Severity": user_input["incident_severity"],
                    "Location (State)": user_input["incident_state"],
                    "Collision Type": user_input["collision_type"],
                    "Property Damage": user_input["property_damage"],
                    "Authorities Contacted": user_input["authorities_contacted"],
                    "Claimed Amount": f"${user_input['vehicle_claim']:,.2f}"
                }
                
                # Display as a cleaner DataFrame
                st.dataframe(pd.DataFrame([formatted_claim]), use_container_width=True)
                
                # Add a submit button that appears after prediction
                if st.button("Submit Claim"):
                    st.balloons()
                    st.success("Your claim has been submitted successfully!")
                    st.write("Reference #: CLAIM-" + str(np.random.randint(10000, 99999)))
            
            except Exception as e:
                st.error(f"Error processing claim: {str(e)}")
                st.info("Our system encountered an issue processing your claim. Please try again or contact customer support.")
                
                # Debug information in an expander
                with st.expander("Technical Details (for support)"):
                    st.write("Error details:", str(e))
                    st.write("User input:", user_input)
    
    # Call the file_claim_survey function
    file_claim_survey()
############################################################################################
###########################################################################################

elif page == "Conclusion":
    st.header("Conclusion")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Key Results Projects")
        st.subheader("Models")
        st.markdown(
            "- Top models (LDA with SMOTETomek, Ridge, AdaBoost) got weighted F1 around 0.84–0.85\n"
            "- We use F1-score because data imbalanced\n"
            "- Models show good balance of precision and recall"
        )

        st.subheader("Top Features")
        st.markdown(
            "- incident_severity_major_damage\n"
            "- insured_hobbies_chess\n"
            "- insured_hobbies_crossfit"
        )

        st.subheader("Benchmark")
        st.markdown(
            "- Similar projects on Kaggle/GitHub had F1 ≈ 0.67\n"
            "- They often used accuracy, not balancing classes\n"
            "- Our strict feature selection & one-hot encoding gave better results "
        )

    with col2:
        st.subheader("Project Work")
        st.subheader("Main Challenges")
        st.markdown(
            "- Small dataset (1000 claims) limited improvement\n"
            "- Learning curve for model tuning and preprocessing order\n"
            "- Code merge issues in team slowed progress"
        )

        st.subheader("Contributions")
        st.markdown(
            "- Complete data cleaning and preprocessing\n"
            "- Tested many classifiers plus resampling techniques\n"
            "- Hyperparameter tuning and cross-validation"
        )

        st.subheader("Lessons Learned")
        st.markdown(
            "- Plan whole workflow early to avoid rework\n"
            "- Keep test set safe for final unbiased evaluation\n"
            "- Explore more creative feature engineering next time"
        )

elif page == "Appendix - Extras":
    st.header("Appendix - Extras")
    st.subheader("Feature Engineering Ideas to Explore")
    st.subheader("Basic Feature Enhancements")
    st.markdown("""
    * **Date/Time Features:** Extract day of the week, time since policy inception, and time between incident and claim.
    * **Handling of 'Unknown' or Missing Values:** Create a separate category for missing data or a binary flag indicating whether a value was provided (Categories for hobbies).
    """)
    st.subheader("Interaction Features")
    st.markdown("""
    * **Severity x Claim Amount:** Create a ratio of claim amount to incident severity.
    * **Vehicle Age/Type x Claim Amount:** Combine vehicle information with claim amount.
    * **Hobbies x Incident Type/Severity**: Interaction between risky hobbies and incident.
    """)
    st.subheader("Domain-Specific Features")
    st.markdown("""
    * **Claim History Features:** Include number of prior claims and time since the last claim.
    * **Policyholder Behavior Features:** Consider policyholder tenure and changes in coverage.
    * **Geographic Risk Factors:** Incorporate external data about the incident location (e.g., crime rates).
    * **Text Analysis:** Use NLP to extract features from incident descriptions (e.g., keywords, sentiment).
    """)
    st.subheader("Advanced Techniques")
    st.markdown("""
    * **Clustering-Based Features:** Use clustering to group similar claims and use the cluster assignment as a feature.
    """)