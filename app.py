import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("💼 Employee Salary Classification (Random Forest)")

st.write("Predict whether salary is High or Low")

# Inputs
age = st.number_input("Age", min_value=18, max_value=65)
experience = st.number_input("Years of Experience", min_value=0)
education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])
hours_per_week = st.number_input("Working Hours per Week", min_value=0)
projects = st.number_input("Number of Projects", min_value=0)
certifications = st.number_input("Certifications Count", min_value=0)

# Convert categorical to numeric
edu_map = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
edu = edu_map[education]

# Dummy dataset
X = np.array([
    [25,2,1,40,2,1],
    [40,10,2,50,5,3],
    [30,5,1,45,3,2],
    [50,20,3,60,6,5],
    [28,3,1,42,2,1],
    [45,15,2,55,5,4]
])

y = np.array([0,1,0,1,0,1])  # 0 = Low, 1 = High

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Prediction
if st.button("Predict Salary Category"):
    input_data = np.array([[age, experience, edu, hours_per_week, projects, certifications]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("High Salary 💰")
    else:
        st.error("Low Salary 💵")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Random Forest classification model for salary prediction.")
