import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Kanpai!!", page_icon=":beer:", layout="wide", initial_sidebar_state="auto")

# Load dataset
df = pd.read_csv('HappinessAlcoholConsumption.csv')
columns = ['HappinessScore', 'HDI', 'GDP_PerCapita', 'Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']

# Functions
def show_csv(df):
    st.dataframe(df, use_container_width=True)
    st.write("\n")
    return df
    
def clean_csv(df):
    st.subheader("Cleaning the Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("__Handling Missing Values__")
        df.dropna(inplace=True) 
        
        code = '''
        df.dropna(inplace=True)'''
        st.code(code, language="python")
 
        st.write("_Insert the analysis after the process._")
        
    with col2:
        st.write("__Handling Duplicate Data__")
        df.drop_duplicates(inplace=True) 
    
        code = '''
        df.drop_duplicates(inplace=True) '''
        st.code(code, language="python")
        
        st.write("_Insert the analysis after the process._")
        
    # with st.expander("Show Cleaned Dataset", expanded=True):
    #         st.dataframe(df, use_container_width=True)
    st.write("\n")
    return df
    
def explore_stats(df, columns):
    st.subheader("Exploring Statistics")
    
    stats_list = []
    
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
    
def heatmap(df, columns):
    
    st.subheader("Correlation Heatmap")
    
    selected_col = df[columns]
    correlation_matrix = selected_col.corr()
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Happiness and Alcohol Consumption')
    st.pyplot(plt)

    st.write("The heatmap indicates that improving HDI and GDP may play significant roles in enhancing happiness, while alcohol consumption, particularly beer, contributes positively but is less central to the overall happiness score.")

# Sidebar
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Preparation", 
                                 "Data Visualization", "Conclusion"])

# Application
st.header("Happiness and Alcohol Consumption Analysis") 

if options == "Introduction":
    st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True) 
    st.write("_Insert description of the project here._")

elif options == "Data Exploration and Preparation":
    st.subheader("Dataset Overview")
    st.write("_Insert short description of the exploration and preparation._")
    
    show_csv(df)
    df = clean_csv(df)
    explore_stats(df, columns)
    
elif options == "Data Visualization":
    st.subheader("Data Visualization")
    st.write("_Insert short description of the visualization techniques here._")
    
    df = clean_csv(df)
    heatmap(df, columns)
    
    
    
        
    
