import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st  # Added Streamlit import

# Load the dataset
df = pd.read_csv('HappinessAlcoholConsumption.csv')

# Display the DataFrame in Streamlit
st.title('Happiness and Alcohol Consumption Data')  # Added title
st.write(df)  # Display the DataFrame