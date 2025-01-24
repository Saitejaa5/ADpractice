import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models
diabetes_model = pickle.load(open('C:/Users/Gaurav/OneDrive/Desktop/Vitapredict/Saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/Gaurav/OneDrive/Desktop/Vitapredict/Saved models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/Gaurav/OneDrive/Desktop/Vitapredict/Saved models/parkinsons_model.sav', 'rb'))

# App Styling
st.set_page_config(page_title="Vitapredict - ML Models", page_icon="🔬", layout="wide")

# Header Section
st.title("Vitapredict - Machine Learning Models")
st.markdown("""
    **Welcome to Vitapredict!**  
    This is a tool to predict various health conditions using Machine Learning.  
    Choose a model from the sidebar to get started.
""")
st.markdown("---")

# Sidebar for navigation with enhanced design
st.sidebar.title("Vitapredict ML Models")
st.sidebar.write("Select a prediction model:")

# Updated navigation using option_menu
selected = option_menu(
    None,
    ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
    icons=['activity', 'heart', 'person'],
    default_index=0,
    orientation='vertical',
    styles={
        "container": {
            "padding": "0px", 
            "background-color": "#f8f9fa", 
            "border-radius": "10px",  # Rounded corners for the sidebar
        },
        "nav-link": {
            "font-size": "18px", 
            "text-align": "left", 
            "margin": "5px", 
            "--hover-color": "#e0e0e0", 
            "padding": "12px",  # Increased padding for better usability
            "border-radius": "5px",  # Rounded borders for navigation items
        },
        "nav-link-selected": {
            "background-color": "#1f77b4", 
            "color": "white", 
            "font-weight": "bold",  # Highlight the selected item
        },
        "icon": {
            "font-size": "22px",  # Increased icon size for better visibility
            "color": "#007bff",  # Blue color for icons
        },
    }
)

# Content Pages

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction using ML')
    st.markdown("""
    Provide the necessary information below to predict whether the person is diabetic.
    """)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
            Glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=100)
            BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=200, value=80)
            SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100, value=20)
        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0, max_value=300, value=80)
            BMI = st.number_input('BMI value', min_value=0, max_value=60, value=25)
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.0, value=0.5)
            Age = st.number_input('Age of the Person', min_value=18, max_value=120, value=30)

        if st.button('Diabetes Test Result', key='diabetes_test'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            diab_diagnosis = 'The person is Diabetic' if diab_prediction[0] == 1 else 'The person is Not Diabetic'
            st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.header('Heart Disease Prediction using ML')
    st.markdown("""
    Provide the details below to predict the likelihood of heart disease.
    """)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age of the Person', min_value=20, max_value=100, value=50)
            sex = st.selectbox('Sex of the Person', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.number_input('Chest pain types', min_value=0, max_value=3, value=0)
            trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=250)
            fbs = st.selectbox('Fasting blood sugar > 120 mg/dl', options=[0, 1])
        with col2:
            restecg = st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2, value=0)
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=50, max_value=220, value=150)
            exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
            oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=6.0, value=1.0)
            slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2, value=1)
            ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=3, value=1)
            thal = st.selectbox('Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', options=[0, 1, 2])

        if st.button('Heart Test Result', key='heart_test'):
            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            heart_diagnosis = 'The person is suffering from Heart disease' if heart_prediction[0] == 1 else 'The person is Not suffering from Heart disease'
            st.success(heart_diagnosis)

# Parkinsons Prediction Page
if selected == 'Parkinsons Prediction':
    st.header('Parkinsons Prediction using ML')
    st.markdown("""
    Enter the necessary features to predict if the person has Parkinson's disease.
    """)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=50.0, value=150.0)
            fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=1000.0, value=250.0)
            flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=50.0, value=80.0)
            Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=100.0, value=1.0)
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.02)
            RAP = st.number_input('MDVP:RAP', min_value=0.0, max_value=10.0, value=0.5)
            PPQ = st.number_input('MDVP:PPQ', min_value=0.0, max_value=10.0, value=0.5)
        with col2:
            Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=10.0, value=0.5)
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=10.0, value=0.5)
            APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=10.0, value=0.5)
            APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=10.0, value=0.5)
            APQ = st.number_input('MDVP:APQ', min_value=0.0, max_value=10.0, value=0.5)
            DDA = st.number_input('Shimmer:DDA', min_value=0.0, max_value=10.0, value=0.5)
            NHR = st.number_input('NHR', min_value=0.0, max_value=10.0, value=0.5)
            HNR = st.number_input('HNR', min_value=0.0, max_value=10.0, value=0.5)

        if st.button('Parkinsons Test Result', key='parkinsons_test'):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR]])
            parkinsons_diagnosis = 'The person is suffering from Parkinsons disease' if parkinsons_prediction[0] == 1 else 'The person is Not suffering from Parkinsons disease'
            st.success(parkinsons_diagnosis)
