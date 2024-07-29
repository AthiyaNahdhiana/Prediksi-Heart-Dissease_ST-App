import streamlit as st
import pandas as pd
import pickle

#model 1 KNN
model1_filename = 'C:/Users/athiy/Documents/1-KULIAH UDINUS/8 GENAP 23-24/Bengkel Koding/UAS/Heart-Disease-main/Heart-Disease-main/model/modelKNN.pkl'

with open(model1_filename, 'rb') as file:
    model1 = pickle.load(file)

#model 2 Random Forest
model2_filename = 'C:/Users/athiy/Documents/1-KULIAH UDINUS/8 GENAP 23-24/Bengkel Koding/UAS/Heart-Disease-main/Heart-Disease-main/model/modelRandForest.pkl'

with open(model2_filename, 'rb') as file:
    model2 = pickle.load(file)

def main():
    st.title('Heart Disease Prediction')
    age = st.number_input('Age', 28 , 66)
    sex_options = ['Male', 'Female']
    sex = st.selectbox('Sex', sex_options)
    sex_num = 1 if sex == 'Male' else 0 
    cp_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic']
    cp = st.selectbox('Chest Pain Type', cp_options)
    cp_num = cp_options.index(cp)
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120)
    chol = st.slider('Cholesterol', 100, 600, 250)
    fbs_options = ['False', 'True']
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', fbs_options)
    fbs_num = fbs_options.index(fbs)
    restecg_options = ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy']
    restecg = st.selectbox('Resting Electrocardiographic Results', restecg_options)
    restecg_num = restecg_options.index(restecg)
    thalach = st.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang_options = ['No', 'Yes']
    exang = st.selectbox('Exercise Induced Angina', exang_options)
    exang_num = exang_options.index(exang)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope_options = ['Upsloping', 'Flat', 'Downsloping']
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', slope_options)
    slope_num = slope_options.index(slope)
    ca = st.number_input('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 1)
    thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
    thal = st.selectbox('Thalassemia', thal_options)
    thal_num = thal_options.index(thal)

    with open('C:/Users/athiy/Documents/1-KULIAH UDINUS/8 GENAP 23-24/Bengkel Koding/UAS/Heart-Disease-main/Heart-Disease-main/model/mean_std_values.pkl', 'rb') as f:
        mean_std_values = pickle.load(f)


    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'age': [age],
            'sex': [sex_num],  
            'cp': [cp_num],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_num],
            'restecg': [restecg_num],
            'thalach': [thalach],
            'exang': [exang_num],
            'oldpeak': [oldpeak],
            'slope': [slope_num],
            'ca': [ca],
            'thal': [thal_num]
        })
        # Apply saved transformation to new data
        user_input = (user_input - mean_std_values['mean']) / mean_std_values['std']
        
        #model1 KNN
        prediction1 = model1.predict(user_input)
        prediction_proba1 = model1.predict_proba(user_input)

        if prediction1[0] == 1:
            bg_color = 'red'
            prediction_result1 = 'Positive'
        else:
            bg_color = 'green'
            prediction_result1 = 'Negative'
        
        confidence = prediction_proba1[0][1] if prediction1[0] == 1 else prediction_proba1[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction by KNN Model: {prediction_result1}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

        #model2 Random Forest
        prediction2 = model2.predict(user_input)
        prediction_proba2 = model2.predict_proba(user_input)

        if prediction2[0] == 1:
            bg_color = 'red'
            prediction_result2 = 'Positive'
        else:
            bg_color = 'green'
            prediction_result2 = 'Negative'
        
        confidence = prediction_proba2[0][1] if prediction2[0] == 1 else prediction_proba2[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction by Random Forest Model: {prediction_result2}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
