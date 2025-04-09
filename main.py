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

st.title("Overview")
st.sidebar.title("Navigation")


df=pd.read_csv("insurance_claims_geodata.csv", parse_dates=["policy_bind_date", "incident_date"], index_col="policy_number")
df.drop(['months_as_customer',  'umbrella_limit',  '_c39'],axis=1,inplace= True)

# Beispielwerte
total_fraud_amount = df['total_claim_amount'][df['fraud_reported'] == 'Y'].sum()
total_amount=df['total_claim_amount'].sum()

#change_percent = "+5.3%"
st.subheader("KPI Dashboard")
col1,col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div style="background-color:#F8F8FF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Total Claim Amount</h3>
        <h1 style="font-size:48px; color:#000000;">${total_amount:,.0f}</h1>
    </div>
""", unsafe_allow_html=True)
with col2:
     st.markdown(f"""
    <div style="background-color:#F8F8FF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Total Fraud Claim Amount</h3>
        <h1 style="font-size:48px; color:#FF0000;">${total_fraud_amount:,.0f}</h1>
    </div>
""", unsafe_allow_html=True)
with col3:
    total_legit_amount=total_amount-total_fraud_amount
    st.markdown(f"""
    <div style="background-color:#F8F8FF; padding:20px; border-radius:10px; text-align:center">
        <h3 style="margin-bottom:5px;">Total Not Fraud Claim Amount</h3>
        <h1 style="font-size:48px; color:#0000FF;">${total_legit_amount:,.0f}</h1>
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
col1, col2 = st.columns([2, 1]) 

# 🟦 Linke Spalte: Bar Chart
with col1:
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
        color_discrete_map={"Y": "red", "N": "blue"},
    )
    fig_bar.update_layout(
    height=500,
    showlegend=False 
    )   
    st.plotly_chart(fig_bar, use_container_width=True, showlegend=False)
    
with col2:
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
        color_discrete_map={"Y": "red", "N": "blue"}
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig_map, use_container_width=True)





