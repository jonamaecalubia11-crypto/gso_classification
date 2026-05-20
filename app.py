import streamlit as st
import pandas as pd
import joblib

# =====================
# LOAD MODEL FILES
# =====================

model = joblib.load("gso_model.pkl")

le_title = joblib.load("label_title.pkl")
le_department = joblib.load("label_department.pkl")
le_urgency = joblib.load("label_urgency.pkl")
le_target = joblib.load("label_target.pkl")

# =====================
# PAGE TITLE
# =====================

st.title("🏛️ GSO Document Classification System")

st.write("Predict the type of government document.")

# =====================
# USER INPUTS
# =====================

document_title = st.selectbox(
    "Document Title",
    le_title.classes_
)

department = st.selectbox(
    "Department",
    le_department.classes_
)

urgency_level = st.selectbox(
    "Urgency Level",
    le_urgency.classes_
)

pages = st.number_input(
    "Number of Pages",
    min_value=1,
    max_value=20,
    value=1
)

# =====================
# PREDICTION
# =====================

if st.button("Predict Document Type"):

    # ENCODE INPUTS
    title_encoded = le_title.transform([document_title])[0]
    department_encoded = le_department.transform([department])[0]
    urgency_encoded = le_urgency.transform([urgency_level])[0]

    # CREATE INPUT DATAFRAME
    input_data = pd.DataFrame([[
        title_encoded,
        department_encoded,
        urgency_encoded,
        pages
    ]], columns=[
        "document_title",
        "department",
        "urgency_level",
        "pages"
    ])

    # PREDICT
    prediction = model.predict(input_data)[0]

    # DECODE RESULT
    predicted_label = le_target.inverse_transform([prediction])[0]

    # DISPLAY RESULT
    st.success(f"Predicted Document Type: {predicted_label}")
