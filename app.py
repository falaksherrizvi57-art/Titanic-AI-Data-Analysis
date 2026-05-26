import streamlit as pd
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# --- 1. APP TITLE & DESCRIPTION ---
st.title("🚢 Titanic Survival Predictor AI")
st.write("This app uses a Machine Learning model to predict if you would survive the Titanic disaster based on your profile!")

# --- 2. LOAD & TRAIN MODEL (Behind the Scenes) ---
@st.cache_data
def train_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    X = df[['Pclass', 'Sex', 'Age']]
    y = df['Survived']
    
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    return model

model = train_model()

# --- 3. USER INTERFACE (Inputs on the Web Page) ---
st.header("Enter Your Passenger Profile:")

# Dropdown for Ticket Class
pclass = st.selectbox("Which Ticket Class would you buy?", options=[1, 2, 3], 
                     format_func=lambda x: f"{x}st Class" if x==1 else f"{x}nd Class" if x==2 else f"{x}rd Class")

# Dropdown for Gender
gender = st.selectbox("Select Gender:", options=["Male", "Female"])
sex_encoded = 1 if gender == "Female" else 0

# Slider for Age
age = st.slider("Select Age:", min_value=1, max_value=100, value=25)

# --- 4. PREDICTION LOGIC ---
if st.button("Predict My Survival Chance"):
    # Make prediction using the trained model
    user_data = pd.DataFrame([[pclass, sex_encoded, age]], columns=['Pclass', 'Sex', 'Age'])
    prediction = model.predict(user_data)[0]
    
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **You would likely have SURVIVED!** The AI predicts a high survival probability for a {age}-year-old {gender.lower()} in {pclass} class.")
    else:
        st.error(f"💀 **You would likely NOT have survived.** The AI predicts low survival odds for a {age}-year-old {gender.lower()} in {pclass} class.")