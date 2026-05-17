
import streamlit as st
import pickle
import numpy as np 
model =  pickle.load(open("iris_model.pkl", "rb"))

st.title("Iris flower prediction")

sepal_lenght = st.slider("sepal lenght ",4.0,8.0,5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("predict"):
    features= np.array([[sepal_lenght, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: {species[prediction[0]]}")
    