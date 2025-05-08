import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the insurance claims dataset (assuming you have it locally)
df = pd.read_csv("insurance_claims.csv")  # Replace with the actual path if needed

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

# df=pd.read_csv("insurance_claims.csv")  # Moved to the top
X_train=pd.read_csv("X_train.csv")

st.title("Insurance Claim Prediction ")
st.sidebar.title("Table of contents")
pages=["Business problem", "Data Visualization","Statistical Methods", "Feature Engineering", "Future Feature Engineering", "Modelling", "Real-life Sample Application"] # Added "Future Feature Engineering"
st.sidebar.write("\n\nCreated by:")
st.sidebar.write("Bertrand Tcheuffa  \n Nathalie Mugrauer  \n Quy-Manh Jurca-Tsan \n")
st.sidebar.write("\n\n\n")
# Add a more visible label before the radio buttons
st.sidebar.markdown("<p style='font-size:20px; font-weight:bold; color:white;'>Go to:</p>", unsafe_allow_html=True)
page=st.sidebar.radio("", pages, label_visibility="collapsed")  # Hide the default label
  
if page == pages[0] : 
  st.write("### Introduction")
  ###########################################################################################################################################

if page == pages[1] : 
  st.write("### Data Visualization")
  
#####################################################################################################################

## Feature Engineering
if page == pages[3] :
  st.write("### Feature Engineering")
  st.write("In the previous slides we have gained a solid understanding of our data. " )
  st.dataframe(X_train.info())
  st.write("Missing values in 'authorities_contacted' were handled using the **SimpleImputer()** with the **strategy 'most_frequent'**.")
  st.code("""categorical_columns = ['incident_severity', 'insured_hobbies', 'collision_type', 'incident_type', 'incident_state', 'property_damage', 'authorities_contacted']
 categorical_imputer=SimpleImputer(strategy='most_frequent')

 X_train[categorical_columns]=categorical_imputer.fit_transform(X_train[categorical_columns])
 X_test[categorical_columns]=categorical_imputer.transform(X_test[categorical_columns])
 """)
  st.dataframe(df.head(6))
 #categorical features
  st.write("Consequently, we reviewed these features and decided if and which transformations are needed.")
  st.write("We begin with the categorical features: \n- incident_severity \n - insured_hobbies \n - collision_type \n- incident_type \n- incident_state \n- property_damage \n - authorities_contacted")
  st.write("Most machine learning models work best with numerical input well. Based on this, an  encoding approach is necessary. A internal topic of discussion was the choice of the encoder. ")
  st.image("cat_var.png", caption="Categorical variables encoding")
  st.write(" All categorical variables were encoded with the One-Hot Encoder.")

 #quantitative features
  st.write("The only quantitative feature is **'vehicle_claim'**. ")
  vehicle_claim_desc_df = X_train['vehicle_claim'].describe().to_frame().T
  st.dataframe(vehicle_claim_desc_df)

# --- 4. Future Feature Engineering ---
if page == pages[4]: # Changed from 5 to 4
    st.header("Future Feature Engineering")
    st.write("Here are some potential feature engineering ideas to explore in the future to potentially improve model performance:")

    st.subheader("Basic Feature Enhancements")
    st.markdown("""
    * **More Granular Categorical Encoding:** Explore target encoding or weight of evidence (WOE) for high-cardinality categorical features.
    * **Date/Time Features:** Extract day of the week, time since policy inception, and time between incident and claim.
    * **Handling of 'Unknown' or Missing Values:** Create a separate category for missing data or a binary flag indicating whether a value was provided.
    """)

    st.subheader("Interaction Features")
    st.markdown("""
    * **Severity x Claim Amount:** Create a ratio of claim amount to incident severity.
    * **Incident Location/State Combinations:** Combine location data for more specific geographic information.
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
    * **Anomaly Detection Scores:** Use an anomaly detection model to score claims and use the anomaly score as a feature.
    * **Graph-Based Features:** If applicable, use graph analysis to extract features from relationships between entities.
    """)



# --- 5. Real-life Sample Application --- # Changed from 6 to 5
if page == pages[5]: # Changed from 5 to 6
    st.markdown("<h2 style='color:red;'>Real-life Sample Application: Insurance Fraud Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:red;'>This application demonstrates how the machine learning model could be used in a real-world scenario to assess the risk of fraud.</p>", unsafe_allow_html=True)

    # --- 5.1.  Application Scenario: New Insurance Policy ---
    def new_policy_survey():
        st.subheader("New Insurance Policy Application")
        st.write("Please provide the following information to process your insurance application.")

        # Use a dictionary to structure the questions and their input types
        questions = {
            "Incident Severity": st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]),
            "Incident State": st.selectbox("Incident State", df['incident_state'].unique()),
            "Insured Hobbies": st.multiselect("Insured Hobbies", df['insured_hobbies'].unique()),
            "Collision Type": st.selectbox("Collision Type", ["No Collision", "Rear Collision", "Front Collision", "Side Collision", "Other"]),
            "Property Damage": st.radio("Property Damage", ["YES", "NO", "Unknown"]),
            "Authorities Contacted": st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"]),
            "Vehicle Claim Amount": st.number_input("Vehicle Claim Amount", min_value=0.0, format="%.2f"),
            "Incident Type": st.selectbox("Incident Type", df['incident_type'].unique()),
        }

        # Store the answers in a dictionary
        answers = {question: answer for question, answer in questions.items()}

        # Convert the answers to a DataFrame (for consistency)
        answers_df = pd.DataFrame([answers])

        # Display the answers
        st.subheader("Your Answers:")
        st.dataframe(answers_df)

        # "Submit" button (no actual model prediction here, just for show)
        if st.button("Submit Application"):
            st.success("Your application has been submitted.  A representative will contact you.")
            # In a real app, you'd send 'answers_df' to your fraud prediction model here


    # --- 5.2. Claim Scenario:  Filing an Insurance Claim ---
    def file_claim_survey():
        st.subheader("File an Insurance Claim")
        st.write("Please provide details about the incident to process your claim.")

        questions = {
            "Incident Severity": st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"]),
            "Incident State": st.selectbox("Incident State", df['incident_state'].unique()),
             "Insured Hobbies": st.multiselect("Insured Hobbies", df['insured_hobbies'].unique()),
            "Collision Type": st.selectbox("Collision Type", ["No Collision", "Rear Collision", "Front Collision", "Side Collision", "Other"]),
            "Property Damage": st.radio("Property Damage", ["YES", "NO", "Unknown"]),
            "Authorities Contacted": st.selectbox("Authorities Contacted", ["Police", "Fire", "Ambulance", "Other", "None"]),
            "Vehicle Claim Amount": st.number_input("Vehicle Claim Amount", min_value=0.0, format="%.2f"),
            "Incident Type": st.selectbox("Incident Type", df['incident_type'].unique()),
            "Description of Incident": st.text_area("Description of Incident", height=100),  # Added description
            "Date of Incident": st.date_input("Date of Incident"),  # Added date
            "Location of Incident": st.text_input("Location of Incident"), #Added Location
        }
        answers = {question: answer for question, answer in questions.items()}
        answers_df = pd.DataFrame([answers])

        st.subheader("Your Claim Details:")
        st.dataframe(answers_df)

        if st.button("Submit Claim"):
            st.success("Your claim has been submitted. We will contact you with an update.")
            #  send 'answers_df' to your fraud prediction model here


    # --- 5.3.  Main App  ---
    def main():
        # st.title("Insurance Fraud Detection") #moved to main page
        st.sidebar.title("Choose a Scenario")
        scenario = st.sidebar.radio("Select a Scenario", ["New Policy Application", "File a Claim"])

        if scenario == "New Policy Application":
            new_policy_survey()
        elif scenario == "File a Claim":
            file_claim_survey()

    if __name__ == "__main__":
        main()
