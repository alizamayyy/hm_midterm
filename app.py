import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Kanpai!!", page_icon=":beer:", layout="wide", initial_sidebar_state="auto")

def show_csv():
    df = pd.read_csv('HappinessAlcoholConsumption.csv')
    st.dataframe(df, use_container_width=True)
    st.write("\n")
    return df
    
def data_cleaning(df):
    st.subheader("Cleaning the Dataset")
    st.write("_Insert the process/es of cleaning the dataset._")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("__Handling Missing Values__")
        st.write("_Insert the process/es of handling missing values._")
        
        # insert data cleaning process
        cleaned_df = show_csv() 
        
        st.write("_Insert the analysis after the process._")
        
    with col2:
        st.write("__Handling Duplicate Data__")
        st.write("_Insert the process/es of handling duplicate data._")
        
        # insert data cleaning process
        cleaned_df = show_csv()
        
        st.write("_Insert the analysis after the process._")
        
    st.write("\n")
    return cleaned_df
    
def exploring_stats(df):
    st.subheader("Exploring Statistics")
    st.write("_Insert description of the statistics to get and how._")
    
    stats_list = []
    
    columns = ['HappinessScore', 'HDI', 'GDP_PerCapita', 'Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']
    for column in columns:
        data = df[column]
        
        # Calculate statistics
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
        std_val = data.std()
        var_val = data.var()
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val
        percentiles = np.percentile(data.dropna(), [25, 50, 75])
        
        # Append the statistics to the list
        stats_list.append({
            "Statistic": column,
            "Mean": mean_val,
            "Median": median_val,
            "Mode": mode_val,
            "Standard Deviation": std_val,
            "Variance": var_val,
            "Min": min_val,
            "Max": max_val,
            "Range": range_val,
            "25th Percentile": percentiles[0],
            "50th Percentile": percentiles[1],
            "75th Percentile": percentiles[2]
        })
        
    # create dataframe from stats
    stats_df = pd.DataFrame(stats_list)
    st.dataframe(stats_df, use_container_width=True) 
    
    st.write("_Insert the analysis after the process._")
    st.write("\n")
    

st.header("Happiness and Alcohol Consumption Analysis") 

# Sidebar for navigation
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Preparation", 
                                 "Data Visualization", "Conclusion"])


if options == "Introduction":
    st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True) 
    st.write("_Insert description of the project here._")

elif options == "Data Exploration and Preparation":
    st.subheader("Dataset Overview")
    st.write("_Insert description of the dataset here._")
    
    df = show_csv()
    df = data_cleaning(df)
    exploring_stats(df)
    
    
    
    
        
    
