import streamlit as st
import pandas as pd
import pickle
# Browser titel  and set up
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.write("Fill out the applicant details below to evaluate loan eligibility instantly.")
st.divider()

# 2. Load the Trained Model Safely
@st.cache_resource
def load_model():
    try:
        with open ('classifier.pkl','rb') as f:
            return pickle.load(f)

    except FileNotFoundError:
        return None
model = load_model()

if model is None:
    st.error(" **Model file not found!** Please ensure `classifier.pkl` is in the same folder .")
else:
    # Createing a Simple Two-Column Input Layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Applicant Profile")
        cibil_score = st.number_input("CIBIL Credit Score", min_value=300, max_value=900, value=700)
        income_annum = st.number_input("Annual Income ($)", min_value=0, value=500000)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

    with col2:
        st.subheader("Loan & Asset Details")
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=1000000)
        loan_term = st.selectbox("Loan Term (Years)", [2, 4, 6, 8, 10, 12, 14], index=4)
        
        st.caption("Asset Values (Used for collateral calculation):")
        residential = st.number_input("Residential Asset Value ($)", min_value=0, value=200000)
        commercial = st.number_input("Commercial Asset Value ($)", min_value=0, value=0)
        luxury = st.number_input("Luxury Asset Value ($)", min_value=0, value=100000)
        bank_asset = st.number_input("Bank Asset Value ($)", min_value=0, value=50000)
    st.divider()

    #  Calculate Total Assets Automatically
    total_asset_value = residential + commercial + luxury + bank_asset
    # st.info(f" **Total Calculated Assets:** ${total_asset_value:,}")

    #  Prediction Logic Trigger
    if st.button("Check Loan Eligibility", type="primary", use_container_width=True):
        
        # Create input DataFrame structured exactly like our training data
        input_df = pd.DataFrame([{
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential,
            'commercial_assets_value': commercial,
            'luxury_assets_value': luxury,
            'bank_asset_value': bank_asset,
            'total_assets_value': total_asset_value  
        }])
        # Run prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)

        if prediction == 0:
            confidence = probabilities[0][0] * 100
            st.success(f"🎉 **Loan Approved!** (Confidence: {confidence:.1f}%)")
            st.balloons()
        else:
            confidence = probabilities[0][1] * 100
            st.error(f"❌ **Loan Rejected.** (Risk Assessment Confidence: {confidence:.1f}%)")


