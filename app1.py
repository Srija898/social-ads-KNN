import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Title
st.title("🧠 Social Network Ads KNN Classifier")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Social_Network_Ads.csv")
    return df

df = load_data()
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# Features & Target
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Standardize
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# KNN Classifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(x_train, y_train)

# Sidebar for prediction
st.sidebar.header("🧪 Predict Purchase")
age = st.sidebar.slider("Select Age", 18, 60, 30)
salary = st.sidebar.slider("Select Estimated Salary", 15000, 150000, 87000)

user_data = np.array([[age, salary]])
scaled_data = sc.transform(user_data)
prediction = classifier.predict(scaled_data)

st.sidebar.write("### Prediction:")
st.sidebar.success("Purchased ✅" if prediction[0] == 1 else "Not Purchased ❌")

# Show prediction results on test data
st.subheader("🔍 Prediction vs Actual (on Test Set)")
y_pred = classifier.predict(x_test)
result_df = pd.DataFrame(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1),
                         columns=['Predicted', 'Actual'])
st.dataframe(result_df)

# Optional: Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
st.write(f"✅ Model Accuracy on Test Set: {accuracy:.2f}")