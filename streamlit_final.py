import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

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

df = pd.read_csv("insurance_claims.csv")

st.title("Insurance Claim Prediction ")
st.sidebar.title("Insurance Claim Prediction")
pages = ["Business problem", "Data Visualization", "Statistical Methods", "Feature Engineering", "Modelling", "Feature Engineering - Ideas", "Sample Application", "Conclusion"]

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
    st.write("Quy-Manh Jurca-Than")


#####################################################################################################################
#####################################################################################################################

# Main content area based on the selected page
if page == "Business problem":
    st.header("Business Problem")
    st.write("Content for the Business Problem page...")
elif page == "Data Visualization":
    st.header("Data Visualization")
    st.write("Content for the Data Visualization page...")
elif page == "Statistical Methods":
    st.header("Statistical Methods")
    st.write("Content for the Statistical Methods page...")
elif page == "Feature Engineering":
    st.header("Feature Engineering")
    st.write("In the previous slides we have gained a solid understanding of our data. Based on statistical methods we have reduced the dimensionality of our dataset from 40 explanatory features to 8  **(7 categorical and 1 quantitative)**.")
    st.image("df.info().png")
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
    st.write("**Dataframe  before scaling:**")
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
elif page == "Modelling":
    st.header("Modelling")
    st.write("Content for the Modelling page...")
elif page == "Feature Engineering - Ideas":
    st.header("Feature Engineering Ideas to Explore")
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

elif page == "Conclusion":
    st.header("Conclusion")
    st.write("Content for the Conclusion page...")