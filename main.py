import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk


st.set_page_config(
    page_title="Claim Prediction Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Claim Prediction")

st.sidebar.title("Navigation")
st.divider()
st.title("Objective")
st.divider()
st.markdown("### 🧠🔁📉 Develop different machine learning models")  
st.markdown("### 🚨🕵️‍♂️🚨that spots fraudulent insurance claims")  
st.markdown("### ⚡🕵️‍♀️📈based on structured data and historical claim behavior") 



df=pd.read_csv("insurance_claims_geodata.csv", parse_dates=["policy_bind_date", "incident_date"], index_col="policy_number")
df.drop(['months_as_customer',  'umbrella_limit',  '_c39'],axis=1,inplace= True)

total_fraud_amount = df['total_claim_amount'][df['fraud_reported'] == 'Y'].sum()
total_amount=df['total_claim_amount'].sum()
st.divider()
st.title("Economic KPI Dashboard")
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
        <h1 style="font-size:48px; color:'#e74c3c';">${total_fraud_amount:,.0f}</h1>
    </div>
""", unsafe_allow_html=True)
with col3:
    total_legit_amount=total_amount-total_fraud_amount
    st.markdown(f"""
    <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Total Not Fraud Claim Amount</h3>
        <h1 style="font-size:48px; color:#2e86de;">${total_legit_amount:,.0f}</h1>
    </div>
""", unsafe_allow_html=True)

df.rename(columns={'lng':'lon'}, inplace=True)

colA, colB = st.columns(2)
with colA:
    show_fraud = st.checkbox("Only fraud", value=True)
with colB:
    show_nonfraud = st.checkbox("Only no-fraud incidents", value=True)

# 🔍 Daten filtern je nach Auswahl
if not show_fraud and not show_nonfraud:
    st.warning("Choose at least one category")
    filtered_df = df[0:0]  # leeres DataFrame
else:
    conditions = []
    if show_fraud:
        conditions.append("Y")
    if show_nonfraud:
        conditions.append("N")
    filtered_df = df[df['fraud_reported'].isin(conditions)]

# 🔢 Daten für Bar Chart vorbereiten
bar_data = (
    filtered_df
    .groupby([filtered_df['incident_date'].dt.date, 'fraud_reported'])
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
        color_discrete_map={"Y": '#e74c3c', "N":  '#2e86de'},
    )

    fig_pie.update_layout(showlegend=False)

    st.plotly_chart(fig_pie)

with col2:
    st.subheader("Distribution of reported incidents")

    # Gruppieren nach Datum & Fraud
    bar_data = filtered_df.groupby([filtered_df['incident_date'].dt.date, 'fraud_reported']).size().reset_index(name='count')

    fig_bar = px.bar(
        bar_data,
        x='incident_date',
        y='count',
        color='fraud_reported',
        barmode='stack',
        labels={'incident_date': 'Date', 'count': 'Counts'},
        color_discrete_sequence=px.colors.qualitative.D3,
        height=500,
        color_discrete_map={"Y": '#e74c3c', "N": "#2e86de"},
    )
    fig_bar.update_layout(
    height=500,
    showlegend=False 
    )   
    st.plotly_chart(fig_bar, use_container_width=True, showlegend=False)
    
st.divider()
    
with col3:
    st.subheader("Incident Locations")

    fig_map = px.scatter_map(
        filtered_df,
        lat="lat",
        lon="lon",
        color="fraud_reported",
        hover_name="fraud_reported",
        hover_data={"incident_date": True, "lat": False, "lon": False},
        color_discrete_sequence=px.colors.qualitative.D3,
        zoom=4,
        height=400,
        color_discrete_map={"Y": '#e74c3c', "N": "#2e86de"}
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig_map, use_container_width=True)

st.title("Technical KPI Dashboard")
st.divider()
col1,col2, col3, col4, col5, col6= st.columns(6)
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
        <h1 style="font-size:48px; color:#000000;">{df.shape[0]}</h1>
    </div>
""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Features</h3>
        <h1 style="font-size:48px; color:#000000;">{df.shape[1]-1}</h1>
    </div>
""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">NaN-Values</h3>
        <h1 style="font-size:48px; color:#000000;">{df.isna().sum().sum()}</h1>
    </div>
""", unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Quant. Features</h3>
        <h1 style="font-size:48px; color:#000000;">{len(df.select_dtypes(['float64', 'int64']).columns)}</h1>
    </div>
""", unsafe_allow_html=True)
with col6:
    st.markdown(f"""
    <div style="background-color:#FFFFFF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Cat. Features</h3>
        <h1 style="font-size:48px; color:#000000;">{len(df.select_dtypes(['object']).columns)}</h1>
    </div>
""", unsafe_allow_html=True)



