import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu
from scipy.stats import chi2_contingency, chi2

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
st.sidebar.write("\n\nCreated by:")
st.sidebar.write("Bertrand Tcheuffa  \n Nathalie Mugrauer  \n Quy-Manh Jurca-Tsan \n")
st.sidebar.write("\n\n\n")
# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page=st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label
  
if page == pages[0] : 
    st.header("Business Problem Introduction")
    st.divider()
    col1, col2 = st.columns([1,8])
    with col1:
        img = Image.open("shakespeare.jpg")
        img_klein = img.resize((200, 200))
        st.image(img_klein)  
    with col2:
        st.markdown("## To fraud, or not to fraud: that is the question")
        st.markdown("##### 📉 Insurance fraud undermines underwriting profitability, so prompt claim assessment is critical.")
        st.markdown("##### ⚙️ Any suspicion of fraud must be underpinned by robust data analysis and pattern detection before opening an investigation.")
        st.markdown("##### 💵 Efficient and accurate identification of fraudulent cases minimizes investigation costs and prevents unwarranted payouts.")
    
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
    st.header("Objective")
    st.divider()
    st.markdown("### ➡️Our target variable, 'fraud_reported', indicates if a claim is fraudulent or not.")
    st.markdown("### ➡️Type of Machine Learning Problem: Supervised learning." )
    st.markdown("### ➡️Binary Classification Problem ")
    st.divider()
    st.markdown("### 🧠🔁📉Build accurate different machine learning models that spots fraudulent insurance claims.")  
    st.markdown("### 🚨🕵️‍♂️🚨Ensure model stability and reliability by achieving consistently prediction scores across different subsets and unseen data.")  
    st.markdown("### ⚡🕵️‍♀️📈Apply the full data science workflow - from understanding and preprocessing to feature engineering and model evaluation.") 
    #st.write(" Considering the relative large dimensionality of our dataset, after carefully exploring the data , we will review each feature and its relevance to the target variable. Based on this, we will select the most relevant features for our model.")
#############################################################################################################################################
#############################################################################################################################################

if page == pages[1] : 
  st.header("Data Visualization")
  st.markdown("### In this section, we will visualize the data to better understand the relationships between features and the target variable.")
  
  tab1, tab2, tab3, tab4 =st.tabs(["📊 Overview Chart","⚙️ Distribution Categorical", "🧮 Distribution Numerical", "📚 Heatmap"])
  
  with tab1:
    col_img, col_text = st.columns([2, 1])
    with col_img:
        st.header("📊 Overview Chart")
        uploaded_overview =   st.image("meine_figur.png")
    with col_text:
        st.header("Observation")
     
        st.markdown("#### - We observe a distribution of gender that is fairly balanced.")
        st.markdown("#### - The 'authorities_contacted' feature shows that most claims are reported to the police.") 
        st.markdown("#### - While the 'incident_severity' feature indicates that most incidents are minor.") 
        st.markdown("#### - We observe our target variable, 'fraud_reported', is imbalanced, with a higher number of non-fraudulent claims.")
        st.markdown("#### - For the hobbies distribution we observe  that the hobbies chess and crossfit are the most common among fraudulent claims.")
        
  with tab2:
    col_img2, col_text2 = st.columns([2, 1])  
    with col_img2:
        st.header("⚙️ Distribution Categorical")
        st.image("meine_figur3.png")
    with col_text2:
        st.header("Observation")
        
        st.markdown("#### - Approximately 9.1% of the NaN-values in the column ”authorities_contacted” needed to be handled.")
        st.markdown("#### - In addition to missing values, we also encountered placeholder values represented as ❔ in three columns.") 
        st.markdown("#### - ”police_report_available”, ”property_damage”, ”collision_type”") 
        
    
  with tab3:
    col_img3, col_text3 = st.columns([2, 1])  
    with col_img3:
        st.header("🧮 Distribution Numerical")
        st.image("meine_figur2.png")
    with col_text3:
        st.header("Observation")
        
        st.markdown("#### - We found that only the feature ”policy_annual_premium” showed approximate normal distributions.")
        st.markdown("#### - All other numerical features did not follow a normal distribution.")

  with tab4:
    col_img4, col_text4 = st.columns([2, 1]) 
    with col_img4:
        st.header("📚 Heatmap")
        st.image("meine_figur4.png")
    with col_text4:
        st.header("Observation")
        
        st.markdown("#### - In this visualization, dark colors indicate strong correlations, while lighter shades represent weaker or no correlation.")
        st.markdown("#### - Two distinct clusters of high correlation are clearly visible.") 
        st.markdown("#### - We observe strong correlations between different types of claims—particularly between ”vehicle_claim” and ”total_claim_amount”.") 
        st.markdown("#### - The second cluster, located in the opposite corner of the heatmap, reveals a similarly strong correlation between ”months_as_customer” and ”age”.")

if page == pages[2]:
    st.header("Data Exploration:")
    st.markdown("##### Before proceeding with statistical analyses, we should establish a strategy for handling missing (NaN) values. \
                In addition to missing values, we also encountered placeholder values represented as ”?” in three columns. \
                This character may either be treated as a missing value and replaced accordingly, or considered as a separate category.\
                In addition, it is important to examine whether the occurrence of ”?” shows any systematic relationship with fraudulent activity.")
    
    tab1, tab2, tab3 = st.tabs(["NaN - authorities_contacted","❔ - police_report_available, property_damage", "❔ - collision_type"])
    
    with tab1:
        st.subheader("NaN - authorities_contacted")
        col_img, col_text = st.columns([2, 1])
    with col_img:
        st.image("meine_figur5.png")
    with col_text:
        st.header("Observation")
     
        st.markdown("#### - We do not want to remove these values, as they represent almost 10% of the data and our overall dataset is relatively small.")
        st.markdown("#### - Deleting them could lead to a loss of important information.") 
        st.markdown("#### - The distribution of fraud and non-fraud cases shows that the missing value is not necessarily an indicator of fraud.") 
        st.markdown("#### - For example, when looking at the crosstab between ”authorities_contacted” and ”incident_type”, it becomes clear that missing values only appear in cases of ”Parked Car” and ”Vehicle Theft”.")
        st.markdown("#### - Since ”Police” is also the most frequently occurring category, we decided to use it as a replacement value during encoding.")
        
    with tab2:
        st.subheader("❔ - police_report_available, property_damage")
        col_img2, col_text2 = st.columns([2, 1])  
        with col_img2:
            st.image("meine_figur6.png")
        with col_text2:
            st.header("Observation")
            
            st.markdown("#### - In ”police_report_available” and ”property_damage”, these ambiguous values may indicate potential fraud, as the information could have been intentionally withheld.")
            st.markdown("#### - To avoid any misleading interpretations or data leakage, these entries were not imputed with common values from the distribution.") 
            st.markdown("#### - ”But instead were explicitly replaced with a new category called 'Unknown'.") 
            
        
    with tab3:
        st.subheader("❔ - collision_type")
        col_img3, col_text3 = st.columns([2, 1])  
        with col_img3:
            st.image("meine_figur7.png")
        with col_text3:
            st.header("Observation")
            
            st.markdown("#### - The ”?”-values in the ”collision_type” column were observed only in relation to the incident types ”Vehicle Theft” and ”Parked Car”.")
            st.markdown("#### - Given the context, this can be considered as a separate category.")
            st.markdown("#### - Therefore, the ”?” entries were replaced with ”No Collision” to accurately reflect the nature of these incidents.")
    
    st.divider()
    st.header("Statistical Methods") 
    st.markdown("##### Now we want to check how statistically relevant our features are to the target variable. We will use the tests we learned for numerical and categorical features.")
    
    tab1, tab2 = st.tabs(["Categorical Features","Numerical Features"])
    
    with tab1:    
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
    
    with tab2:
        results = []
        df["fraud"]=df['fraud_reported'].apply(lambda x: 1 if str(x)=="Y" else 0)
        col_quan=df.select_dtypes(['int64', 'float64']).columns
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
        
        st.dataframe(results_df )
    st.markdown("##### Based on the statistical tests, we decided to keep the features that are statistically relevant:")
    st.markdown("##### incident_severity, collision_type, incident_type, incident_state, property_damage, authorities_contacted, vehicle_claim")
    st.divider()

if page == pages[3]:
    st.header("Conclusion") 
    
    col1,col2 =st.columns(2)
    
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
    
   