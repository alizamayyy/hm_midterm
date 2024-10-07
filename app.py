import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    df.dropna(inplace=True) 
    df.drop_duplicates(inplace=True) 
    return df
    
def show_clean(df):
    st.subheader("Cleaning the Dataset")
    df = clean_csv(df)
    col1, col2 = st.columns(2)
    with col1:
        st.write("__Handling Missing Values__")
        code = '''
        df.dropna(inplace=True)'''
        st.code(code, language="python")
 
        st.write("_Insert the analysis after the process._")     
    with col2:
        st.write("__Handling Duplicate Data__")
        code = '''
        df.drop_duplicates(inplace=True) '''
        st.code(code, language="python")
        
        st.write("_Insert the analysis after the process._")
        
    st.write("\n")
    
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
    st.write("__Correlation Heatmap__")

    modified_columns = [col.replace('_PerCapita', '') for col in columns]
    
    selected_col = df[columns]
    correlation_matrix = selected_col.corr()
    
    correlation_matrix_df = pd.DataFrame(correlation_matrix.values, 
                                         index=modified_columns, 
                                         columns=modified_columns)
    
    fig = px.imshow(correlation_matrix_df,
                    labels=dict(x="", y="", color="Correlation"),
                    x=correlation_matrix_df.columns,
                    y=correlation_matrix_df.index,
                    color_continuous_scale='RdBu',
                    text_auto=True,
                   )
    
    fig.update_layout(width=700, height=600)
    fig.update_xaxes(side="top")
    fig.update_traces(texttemplate="%{z:.2f}", textfont_size=12) 
    fig.update_coloraxes(colorbar=dict(
        orientation="h",  
        yanchor="bottom",
        y=-0.2,  
        xanchor="center",
        x=0.5,  
        title=dict(text='', side='top') 
    ))   
    st.plotly_chart(fig, use_container_width=True)

    st.write("_Insert heatmap description here_")

def box_plot(df, columns):  
    df_melted = df[columns].melt(var_name='Metric', value_name='Value')

    fig = px.box(df_melted, x="Value", y="Metric", color="Metric",
                title="Box Plot of Various Metrics",
                hover_data=["Metric"]  
                )
    
    fig.update_layout(yaxis_title='',
                    xaxis_title='',
                    title='Box Plot of Various Metrics',
                    legend_title_text='',
                    legend=dict(
                        orientation="h",  
                        yanchor="bottom",
                        y=-0.3, 
                        xanchor="center", 
                        x=0.5         
                    )
                )
    
    st.plotly_chart(fig)

    st.write("Potential correlations exist between beer consumption and higher happiness scores, suggesting social aspects of beer drinking may positively impact well-being.")

def world_map(df):
    df['Avg_Alcohol_PerCapita'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)
    
    fig = px.choropleth(df, 
                        locations='Country', 
                        locationmode='country names', 
                        color='Avg_Alcohol_PerCapita',
                        hover_name='Country', 
                        color_continuous_scale=px.colors.sequential.Plasma,
                        labels={'Avg_Alcohol_PerCapita': 'Average Alcohol Consumption (L)'},
                        title='Average Alcohol Consumption Per Capita by Country')
    
    fig.update_coloraxes(colorbar=dict(
        orientation="h",  
        yanchor="bottom",
        y=-0.2,  
        xanchor="center",
        x=0.5,  
        title=dict(text='Average Alcohol Consumption', side='top') 
    ))
        
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig)
        
    st.write("This map visualizes the average alcohol consumption per capita by country, where each country is colored according to the level of alcohol consumption.")
    st.write("\n")

def grouped_by_chart(df):
    region_alcohol_consumption = df.groupby('Region')[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean().reset_index()
    region_alcohol_consumption_melted = region_alcohol_consumption.melt(id_vars='Region', var_name='Alcohol_Type', value_name='PerCapita')
    
    fig = px.bar(region_alcohol_consumption_melted, 
                  x='Region', 
                  y='PerCapita', 
                  color='Alcohol_Type', 
                  barmode='group',
                  title='Average Alcohol Consumption by Region',
                  labels={'PerCapita': 'Average Per Capita Consumption', 'Region': 'Region'},
                )
    
    fig.update_layout(xaxis_title='Region',
                      yaxis_title='Average Per Capita Consumption',
                      xaxis_tickangle=-45,
                      legend_title_text='',
                      legend=dict(
                          orientation="h",  # Horizontal orientation
                          yanchor="bottom",
                          y=-0.5,  # Adjust this value to move the legend further down
                          xanchor="center",
                          x=0.5  # Center the legend
                      )
                 )
    
    fig.update_layout(
        height=700,  
        width=None,  
    )
    
    st.plotly_chart(fig, use_container_width=True)    
    st.write("Consumption patterns highlight distinct regional preferences, with Western Europe favoring wine, while North America and Central & Eastern Europe prefer beer and spirits, respectively.")
    
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
    show_clean(df)
    explore_stats(df, columns)
    
elif options == "Data Visualization":
    st.subheader("Data Visualization")
    st.write("_Insert short description of the visualization techniques here._")
    
    st.write("\n")
    df = clean_csv(df)

    heatmap(df, columns)
    box_plot(df, columns)
    world_map(df)
    grouped_by_chart(df)
    
    
        
    
