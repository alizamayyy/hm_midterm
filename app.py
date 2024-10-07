import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

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
    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.write("__Handling Missing Values__")
        code = '''
        df.dropna(inplace=True)'''
        st.code(code, language="python")
 
        st.write(
        "The `df.dropna()` method eliminates any rows with missing values in the DataFrame, ensuring the data remains reliable. "
        "This step is crucial for conducting accurate analyses, and the use of `inplace=True` ensures that the original DataFrame is updated."
        )
    with col2:
        st.write("__Handling Duplicate Data__")
        code = '''
        df.drop_duplicates(inplace=True) '''
        st.code(code, language="python")
        
        st.write(
        "The `df.drop_duplicates()` function removes any duplicate rows from the DataFrame, ensuring that each record is unique. "
        "This process is essential for maintaining data integrity and avoiding skewed analysis results, and using `inplace=True` updates the DataFrame directly."
    )
        
    st.write("\n")
    
def explore_stats(df, columns):
    st.subheader("Exploring Statistics")
    st.write("After completing the data cleaning process, the table displays a detailed summary of essential statistics for each variable in the dataset. This overview includes crucial measures of central tendency, such as the mean and median, alongside indicators of variability like standard deviation and range. Analyzing these statistics provides important insights into the distribution and nature of the data, which will be vital for informing the next steps in our analysis.")
    
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

    st.write("_Insert short heatmap description here. Mention that this is the heatmap between all numerical colunmns of the dataset. Specifically mention what relationship (x,y) has a value closest to one (direct relationstip) and what relationship has a value closest to -1 (inverse relationship). Mention two of each._")
    st.write("\n")

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
    st.write("_Mention how the happiness score in the box plot is relatively very small among others (Something about the value is only to 10, see CSV). Mention how the GDP per capita has many outliers and why do you think so many outliers are present._")
    st.write("\n")

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
    st.write("_Mention how the area around the blue side in the center is on the lower part of the scare (Mention the countries) and how the Russian Federation is seemingly bright yellow, indicating the country being on the higher point of the scale. Enumerate the possibilities as to why they are at the bottom or why the russians seem to consume more alcohol that anywhere. Special mention the philippines and explain the possibiliteis why._")
    st.write("\n")

def grouped_by_chart(df):
    region_alcohol_consumption = df.groupby('Region')[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean().reset_index()
    region_alcohol_consumption_melted = region_alcohol_consumption.melt(id_vars='Region', var_name='Alcohol_Type', value_name='PerCapita')
    
    fig = px.bar(region_alcohol_consumption_melted, 
                  x='Region', 
                  y='PerCapita', 
                  color='Alcohol_Type', 
                  barmode='group',
                  title='Alcohol Consumption by Region',
                  labels={'PerCapita': 'Average Per Capita Consumption', 'Region': 'Region'},
                )
    
    fig.update_layout(xaxis_title='Region',
                      yaxis_title='Average Per Capita Consumption',
                      xaxis_tickangle=-10,
                      legend_title_text='',
                      legend=dict(
                          orientation="h",  # Horizontal orientation
                          yanchor="bottom",
                          y=-0.25,  # Adjust this value to move the legend further down
                          xanchor="center",
                          x=0.5  # Center the legend
                      )
                 )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig, use_container_width=True)    
    st.write("Consumption patterns highlight distinct regional preferences, with Western Europe favoring wine, while North America and Central & Eastern Europe prefer beer and spirits, respectively.")
    st.write("_Add the highest and lowest consumption of beer, spirit, and wine. Why do you think that region is the highest of beer, spirit, and wine?_")
    
def ave_alcohol_consumption(df):
    mean_values = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean().reset_index()
    mean_values.columns = ['Alcohol Type', 'Average Consumption']
    
    # Remove '_PerCapita' from the 'Alcohol Type' labels
    mean_values['Alcohol Type'] = mean_values['Alcohol Type'].str.replace('_PerCapita', '', regex=False)
    
    fig = px.bar(mean_values, 
                  x='Alcohol Type', 
                  y='Average Consumption', 
                  title='Average Alcohol Consumption by Type',
                  labels={'Average Consumption': 'Average Consumption'},
                  color='Alcohol Type',
                )
    
    fig.update_layout(xaxis_title='Type of Alcohol',
                      yaxis_title='Average Consumption',
                      xaxis_tickangle=0,
                      legend_title_text='',
                      legend=dict(
                          orientation="h", 
                          yanchor="bottom",
                          y=-0.25, 
                          xanchor="center",
                          x=0.5  
                      )
                 )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("These trends underscore varying cultural preferences for alcohol types, which can inform public health initiatives and market strategies tailored to consumer behavior.")   
    st.write("_Highlight that among the three, it is the beer that is the highest and wine as the lowest. Why do you think beer is the highest and wine is the lowest?_")

def horizontal_bar_chart(df):    
    avg_happiness_by_region = df.groupby('Region')['HappinessScore'].mean().reset_index()
    
    top_happiness_regions = avg_happiness_by_region.sort_values(by='HappinessScore', ascending=False).head(10)
    
    fig = px.bar(top_happiness_regions, 
                  x='HappinessScore', 
                  y='Region', 
                  orientation='h', 
                  title='Top 10 Regions by Happiness Score',
                  color='HappinessScore',  
                  color_continuous_scale=px.colors.sequential.YlOrRd,
                )
    
    fig.update_layout(
        xaxis_title='Average Happiness Score',
        yaxis_title='Region',
        legend_title_text='',
        xaxis_tickangle=0,
        coloraxis_colorbar=dict(
            title="",
            thicknessmode="pixels",
            thickness=15,  
            lenmode="pixels",
            len=330,  
            yanchor="top",
            y=1, 
            xanchor="left",
            x=1  
        )
    )
    
    fig.update_layout(
        height=500,  
        width=None,  
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("The chart highlights the top 10 regions with the highest average happiness scores.")
    st.write("_Mention that out of the scale, what is the region with highest happines score? Why do you think people are happy in that region?_")
    st.write("\n")

def custom_pairplots(df, columns):
    st.subheader("Pair Plot of Various Metrics with Trendlines")
    g = sns.pairplot(df[columns], kind='reg', plot_kws={
        'scatter_kws': {'s': 50, 'alpha': 0.5},
        'line_kws': {'color': 'red', 'linewidth': 1}
    })
    plt.suptitle('Pair Plot of Various Metrics with Trendlines', y=1.02)
    st.pyplot(plt)
        
    st.write("The graph indicates that improving HDI and economic conditions can significantly enhance happiness levels.")

def dynamic_regression_plot(df, columns):
    st.write("__Dynamic Regression Plot__")
    
    col1, col2 = st.columns(2)
    with col1:
        x_column = st.selectbox('Select X-axis value', columns, index=0)
    with col2:
        y_column = st.selectbox('Select Y-axis value', columns, index=1)

    X = df[[x_column]].values 
    y = df[y_column].values

    model = LinearRegression()
    model.fit(X, y)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1,1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X.squeeze(), 
        y=y, 
        mode='markers', 
        name=f'Data Points', 
        marker=dict(color='blue')  # Color for data points
    ))
    fig.add_trace(go.Scatter(
        x=x_range.squeeze(), 
        y=y_range, 
        mode='lines', 
        name='Regression Line', 
        line=dict(color='red')  # Color for regression line
    ))
    fig.update_layout(
        title=f'Regression Plot of {y_column} vs {x_column}',
        xaxis_title=x_column,
        yaxis_title=y_column,
        showlegend=False,
        legend=dict(
            orientation="h",  
            yanchor="bottom",  
            y=-0.25,  
            xanchor="center",  
            x=0.5 
        ),
        height=600,  
        width=None,  
    )

    st.plotly_chart(fig)
    
    st.write("_Compare in the heatmap (closest to 1 and -1 in the heatmap). Say how the closest to 1 has clearly a positive or direct relationship as shown with the trendline and how the closest to -1 has a clear negative or inverse relationship. Mention that the relationship between variables with closest to 0 value in the heatmap has a no definite relationship._")
    st.write("\n")

def dynamic_histogram(df, columns):
    st.write("__Dynamic Histogram Plot__")

    selected_column = st.selectbox('Select Column for Histogram', columns, index=0)

    fig = go.Figure()
    
    hist_data = np.histogram(df[selected_column], bins=20)  # Using 20 bins to match your Seaborn example
    bin_edges = hist_data[1]
    bin_counts = hist_data[0]

    fig.add_trace(go.Histogram(
        x=df[selected_column],
        name='Histogram',
        marker=dict(color='blue'),  # Color for histogram bars
        opacity=0.7,
        xbins=dict(
            start=bin_edges[0],  
            end=bin_edges[-1],  
            size=(bin_edges[1] - bin_edges[0]) 
        )
    ))

    max_y_value = np.max(bin_counts)

    fig.update_layout(
        title=f'Histogram of {selected_column}',
        xaxis_title=selected_column,
        yaxis_title='',
        showlegend=False,
        legend=dict(
            orientation="h",  
            yanchor="bottom",  
            y=-0.25,  
            xanchor="center",  
            x=0.5 
        ),
        height=600,
        width=None,
        xaxis=dict(showgrid=True, zeroline=False),  
        yaxis=dict(
            showgrid=True, 
            zeroline=False,
            range=[0, max_y_value * 1.1]  
        )
    )
    
    st.plotly_chart(fig)
    st.write("_Interpret this to maybe 2-3 lines, idk how to interpret this._")
    st.write("\n")
    
def scatter_HDI_HS(df):
    fig = px.scatter(
    df,
    x='HDI',  
    y='HappinessScore',  
    color='Region', 
    title='Happiness Score vs. HDI by Region',
    labels={'HDI': 'Human Development Index', 'Happiness Score': 'Happiness Score'},
    trendline='ols',
    )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig)
    st.write("\n")

def scatter_ALConsumption_GDP(df):
    df['Average_Alcohol_Consumption'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)
    
    fig = px.scatter(
    df,
    x='Average_Alcohol_Consumption',  
    y='GDP_PerCapita',  
    color='Region', 
    title='GDP vs. Average Alcohol Consumption by Region',
    labels={'Average_Alcohol_Consumption': 'Average Alcohol Consumption', 'GDP': 'Gross Domestic Product'},
    trendline='ols',
    )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig)
    
    st.write("_Explain what the chart is about. Explain the relationship between alcohol consumption and GDP is. highlight the subsharan africa. # Does a higher GDP correlate with higher alcohol consumption? How does income inequality within a country affect alcohol consumption levels? Is there a difference in consumption between different socioeconomic groups? (Try to see the countries of the region as to what -world they are in (first world, third world))_")

def scatter_ALConsumption_HDI(df):
    df['Average_Alcohol_Consumption'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)
    
    fig = px.scatter(
    df,
    x='Average_Alcohol_Consumption',  
    y='HDI',  
    color='Region', 
    title='HDI vs. Average Alcohol Consumption by Region',
    labels={'Average_Alcohol_Consumption': 'Average Alcohol Consumption', 'HDI': 'Human Development Index'},
    trendline='ols',
    )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig)
    
    st.write("_Explain what the chart is about. Explain the relationship between alcohol consumption and HDI is. highlight the subsharan africa. What is the correlation between HDI and alcohol consumption per capita in different countries? Do social factors affect alcohol consumption? How does the level of education, as part of HDI, influence alcohol consumption patterns? How do income levels, as part of the HDI calculation, affect alcohol consumption?_")
    
def scatter_ALConsumption_HS(df):
    df['Average_Alcohol_Consumption'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)
    
    fig = px.scatter(
    df,
    x='Average_Alcohol_Consumption',  
    y='HappinessScore',  
    color='Region', 
    title='Happiness Score vs. Average Alcohol Consumption by Region',
    labels={'Average_Alcohol_Consumption': 'Average Alcohol Consumption', 'HDI': 'Human Development Index'},
    trendline='ols',
    )
    
    fig.update_layout(
        height=600,  
        width=None,  
    )
    
    st.plotly_chart(fig)
    st.write("_Explain what the chart is. What are these variables. highlight the middle east and northern africa as to why they  are the only region who has a clear positive relationship. Does a higher happiness score correlate with increased or decreased alcohol consumption? Are individuals who consume alcohol in moderation happier than those who abstain or overconsume? Do alcohol consumption really affect the happiness of a person?_")
    st.write("\n")

# Sidebar
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Preparation", 
                                 "Data Visualization", "Conclusion"])

# Application
st.header("Happiness and Alcohol Consumption Analysis") 
st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True) 

if options == "Introduction":
    st.write("_Insert introduction/abstract-ish._")

elif options == "Data Exploration and Preparation":
    st.subheader("Dataset Overview")
    
    st.write("_What is the dataset_")
    show_csv(df)
    st.write("_Dataset source and purpose, Number of rows and columns, Column names and data types and definition of EACH, Initial observations and insights_")
    st.write("\n")
    
    show_clean(df)
    explore_stats(df, columns)
    
elif options == "Data Visualization":
    st.subheader("Data Visualization")
    df = clean_csv(df)

    tab_labels = [
    "Data Overview", 
    "Correlation and Distribution Data", 
    "Geographical Insights", 
    "Dynamic Data", 
    "Alcohol Consumption Relationships"
    ]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels)
    
    with tab1:
        st.subheader("Data Overview")
        
        col1, col2 = st.columns(2, gap='large')
        with col1:
            grouped_by_chart(df)
        with col2:
            ave_alcohol_consumption(df)
        horizontal_bar_chart(df)
    
    with tab2:
        st.subheader("Correlation and Distribution Data")
        heatmap(df, columns)
        box_plot(df, columns)
        dynamic_histogram(df, columns)
        
    with tab3:
        st.subheader("Geographical Insights")
        world_map(df)
    
    with tab4:
        st.subheader("Dynamic Data")
        dynamic_regression_plot(df, columns)
    
    with tab5:
        st.subheader("Alcohol Consumption Relationships")
        # scatter_HDI_HS(df)
        scatter_ALConsumption_GDP(df)
        scatter_ALConsumption_HDI(df)
        scatter_ALConsumption_HS(df)
        
elif options == "Conclusion":
    st.write("### Conclusion")
    st.write("This analysis provides insights into the relationship between happiness and alcohol consumption.")
    
    st.write("**Key findings include:**")
    st.write("- **Correlations:** A moderate positive correlation exists between happiness scores and GDP per capita and HDI, emphasizing the importance of economic factors in influencing happiness. Moderate correlations between happiness and beer, spirits, and wine suggest a nuanced relationship between alcohol consumption and happiness.")
    st.write("- **Regional Patterns:** The box plot and bar chart showcase the variability in metrics and highlight significant differences in alcohol consumption across regions, suggesting that cultural and economic factors heavily influence drinking habits.")
    st.write("- **Cultural Influence:** The cultural context appears to shape not just consumption patterns but also overall happiness levels, reflecting the complex interplay between economic indicators, societal norms, and personal well-being.")
    
    st.write("Overall, the insights gleaned from this dataset could serve as a basis for further studies and policy discussions aimed at improving happiness and well-being through both economic development and cultural understanding.")
