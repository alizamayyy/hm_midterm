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
    col1, col2, col3 = st.columns(3, gap='medium')
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
        st.write("__Dropping Unused Columns__")
        code = '''
        df.drop(columns=['Hemisphere'], inplace=True)'''
        st.code(code, language="python")

        st.write(
            "The `df.drop(columns=['Hemisphere'], inplace=True)` method removes the 'Hemisphere' column from the DataFrame. "
            "Since this column is not used in the analysis, dropping it helps streamline the DataFrame, making it easier to work with and improving performance."
        )
        
    with col3:
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
    
    st.write("The heatmap visualizes the correlation between alcohol consumption and various social indicators, highlighting patterns and relationships across countries.")

    st.write("- **Happiness Score and HDI (0.82)**: This strong positive correlation suggests that countries with higher Human Development Index scores tend to report higher happiness levels. This may be due to better access to healthcare, education, and overall living conditions, contributing to enhanced life satisfaction.")
    st.write("- **GDP and HDI (-0.57)**: The moderate negative correlation indicates that higher GDP does not always align with a higher Human Development Index. This could imply that economic growth in some countries may not equitably translate to improvements in health, education, and well-being, potentially due to income inequality or social disparities.")

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

    st.write("- The box plot shows a Happiness Score median of about 5.5, indicating moderate variability among countries, with most above the midpoint. The HDI median is around 740, suggesting high human development but notable disparities. The GDP per capita median of approximately 20,000 highlights significant economic inequalities, with outliers indicating extreme wealth.")
    st.write("- The alcohol consumption data reveals a higher median for beer, indicating a widespread cultural preference, while spirits and wine show lower median consumption with fewer outliers, suggesting limited cultural integration.")
    st.write("- Significant outliers in GDP and alcohol consumption highlight extreme cases, particularly among wealthier nations.")
    st.write("- Potential correlations exist between beer consumption and higher happiness scores, suggesting social aspects of beer drinking may positively impact well-being.")
    
    st.write("\n")
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
    st.write("- **High Alcohol Consumption**: The Russian Federation has clearly the highest alcohol consumption, indicated by a bright yellow hue on the map. Australia also displays relatively high alcohol consumption, represented in orange, likely due to cultural factors and social drinking norms that promote higher intake.")
    st.write("- **Moderate Alcohol Consumption**: Countries such as Mexico, China, and Mongolia exhibit moderate levels of alcohol consumption. This may result from a mix of traditional drinking practices and the influence of globalization, leading to varied consumption patterns across different regions.")
    st.write("- **Low Alcohol Consumption**: The regions surrounding Mali, Niger, Chad, Egypt, and the Democratic Republic of Congo show low alcohol consumption, as depicted on the map. Socioeconomic factors, cultural beliefs, and local regulations often contribute to lower alcohol intake in these areas.")

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
    st.write("\n")  
    st.write("\n")  
    st.write("The graph illustrates significant regional disparities in alcohol consumption, reflecting cultural, economic, and social factors.");
    st.write("Western Europe shows the highest wine consumption, while North America and Central & Eastern Europe prefer beer and spirits, respectively. Additionally, regions like Africa and Asia exhibit lower averages across all types, highlighting distinct regional preferences.")
    
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
    
    st.write("🍺 **Beer**: Shows the highest average consumption, indicating its popularity among consumers.")   
    st.write("🍷 **Spirits**: Reflects moderate consumption levels, suggesting a notable but lesser preference compared to beer.")
    st.write("🍾 **Wine**: Exhibits the lowest average consumption, highlighting its relatively niche appeal.")
    st.write("These trends underscore varying cultural preferences for alcohol types, which can inform public health initiatives and market strategies tailored to consumer behavior.")

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
    st.write("The horizontal bar chart depicting the top 10 regions with the highest average happiness scores reveals that Australia and New Zealand stand out with the highest score, indicated by a deep red hue on the color scale. This elevated level of happiness may be attributed to factors such as robust economic performance, high standards of living, and access to quality healthcare and education. Additionally, a strong sense of community, cultural identity, and abundant recreational opportunities contribute to overall life satisfaction, fostering a positive outlook among residents in this region.")
    
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
    st.write("- There is a potential relationship between Happiness and HDI. Both distributions show a similar pattern of skewness, suggesting a potential positive correlation between happiness and human development.")
    st.write("- A potential relationship between GDP per capita and alcohol consumption can also be seen. All three alcohol consumption metrics (beer, spirit, and wine) show similar skewed distributions, suggesting a potential relationship between economic prosperity and alcohol consumption. However, further analysis would be needed to confirm any causal relationship.")
    
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
    
    st.write("The scatter plot illustrates the relationship between average alcohol consumption (x-axis) and GDP (y-axis) across various regions, with Sub-Saharan Africa as a notable outlier due to its higher GDP values of 300-900, compared to most regions that plateau around 100. The trendline for Sub-Saharan Africa shows a negative correlation between GDP and alcohol consumption, indicating that higher economic output does not necessarily lead to increased alcohol consumption in this region. In contrast, other regions, excluding Eastern Asia and Southeastern Asia, generally exhibit a slight positive correlation, suggesting that higher GDP may be associated with increased alcohol consumption.")
    st.write("Income inequality significantly influences consumption patterns among different socioeconomic groups. Wealthier segments tend to have higher alcohol consumption due to greater disposable income, while lower-income groups often face limited access to alcoholic beverages.")

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
    
    st.write("The scatter plot illustrates the relationship between average alcohol consumption (x-axis) and HDI (y-axis) across regions. Sub-Saharan Africa displays values ranging from 300 to 500, while most regions cluster between 600 and 900. The majority of regions exhibit a positive correlation, indicating that higher HDI tends to be associated with increased alcohol consumption per capita. However, Eastern Asia and Southeastern Asia show a negative relationship, suggesting that in these areas, higher HDI does not correlate with higher alcohol consumption. Western Europe and North America have nearly horizontal trendlines, reflecting little to no correlation between HDI and alcohol consumption.")
    st.write("The correlation between HDI and alcohol consumption varies across countries, as higher HDI generally signifies better living standards, education, and healthcare, which can influence drinking patterns. Social factors, including cultural attitudes and behaviors regarding alcohol, are significant in shaping consumption levels. Additionally, education and income levels, integral to HDI, play critical roles; higher education may lead to healthier lifestyle choices, while increased disposable income typically allows for greater access to alcoholic beverages.")
 
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
    
    st.write("The scatter plot reveals the relationship between happiness scores (y-axis) and average alcohol consumption (x-axis) across various regions. While Sub-Saharan Africa generally shows lower happiness levels, it does not stand out as an outlier in this chart. The Middle East and Northern Africa exhibit a clear positive relationship, indicating that increased alcohol consumption may correlate with higher happiness scores in this region. Conversely, Southeastern Asia, Latin America and the Caribbean, Central and Eastern Europe, and Sub-Saharan Africa show only a slight positive relationship, with trendlines lacking significant correlation. Western Europe, Australia, and New Zealand trendlines are nearly straight, implying minimal relationship between alcohol consumption and happiness, while North America displays a negative correlation, suggesting higher alcohol consumption may relate to lower happiness.")
    st.write("The distinct positive relationship in the Middle East and Northern Africa could be attributed to cultural factors, social norms, or specific economic conditions that promote both alcohol consumption and overall happiness. Generally, the data suggests that while higher happiness scores do not consistently correlate with increased alcohol consumption, individuals who drink alcohol in moderation may experience higher happiness than those who either abstain entirely or overconsume. This relationship highlights the complex interplay between alcohol consumption and happiness, suggesting that factors beyond mere consumption levels, such as social interactions, cultural attitudes, and individual lifestyle choices, significantly influence overall well-being.")

# Sidebar
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Preparation", 
                                 "Data Visualization", "Conclusion"])

# Application
st.header("Happiness and Alcohol Consumption Analysis") 
st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True) 

if options == "Introduction":
   
    st.write("Happiness and well-being are increasingly central themes in global discussions about public health, societal development, and policy making. One aspect of human behavior that has been closely studied in relation to mental health and happiness is alcohol consumption.")
    st.write("Does higher alcohol consumption lead to lower happiness levels? Or are happier countries more likely to have higher alcohol consumption due to social factors?")
    st.write("The aim of this analysis is to explore how alcohol consumption correlates with happiness levels.")
    st.write("The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/marcospessotto/happiness-and-alcohol-consumption).")
    col1, col2, col3= st.columns(3)
    with col1:
        st.image("giphy.gif", )
    with col2:
        st.image("beer-happy.gif", )
    with col3:
        st.image("cheers.gif", )
    
elif options == "Data Exploration and Preparation":
    st.subheader("Dataset Overview")
    
    st.write("The **Happiness and Alcohol Consumption** dataset investigates the relationship between alcohol consumption and social indicators such as happiness, Human Development Index (HDI), and Gross Domestic Product (GDP) per capita by country. Sourced from 2016, it consists of 122 rows and 9 columns, offering insights into how happiness, economic prosperity, and personal development correlate with alcohol consumption patterns across various types, including beer, spirits, and wine.")
    show_csv(df)
    
    columns_info = {
    "Country": "Contains the name of each country in the dataset.",
    "Region": "Indicates the geographical region to which each country belongs.",
    "Hemisphere": "Specifies the hemisphere (north or south) in which each country is located.",
    "HappinessScore": "Represents the happiness level of the population, rated on a scale from 0 to 10.",
    "HDI": "Provides the Human Development Index, a measure of socio-economic development.",
    "GDP_PerCapita": "Reflects the Gross Domestic Product per capita, indicating the average economic output per person.",
    "Beer_PerCapita": "Shows the average liters of beer consumed per person annually.",
    "Spirit_PerCapita": "Represents the average liters of spirits consumed per person annually.",
    "Wine_PerCapita": "Indicates the average liters of wine consumed per person annually."
    }

    with st.expander("_Show Column Descriptions_", expanded=True):
        for column, description in columns_info.items():
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ∘ &nbsp;&nbsp;**{column}**: {description}")
    
    st.write("\n")
    st.write("\n")
    
    show_clean(df)
    
    st.write("\n")
    st.write("\n")
    
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
    st.write("The analysis of alcohol consumption relative to various socioeconomic indicators, including GDP, HDI, and happiness scores, reveals intricate relationships shaped by regional differences and cultural factors. Key findings from the study include:")
    
    st.write("- **GDP and Alcohol Consumption:** While Sub-Saharan Africa presents a notable outlier with a negative correlation between GDP and alcohol consumption, most regions exhibit a slight positive correlation, _suggesting that higher GDP may be associated with increased alcohol consumption_, albeit with exceptions in Eastern and Southeastern Asia.")
    st.write("- **HDI and Alcohol Consumption:** A majority of regions demonstrate a positive correlation between HDI and alcohol consumption, _indicating that higher living standards and education levels are linked to greater alcohol consumption_. However, Eastern and Southeastern Asia stand out with a negative correlation, reflecting unique cultural attitudes toward alcohol.")
    st.write("- **Happiness Scores and Alcohol Consumption:** The Middle East and Northern Africa display a clear positive relationship between alcohol consumption and happiness scores, _indicating that moderate consumption may be associated with increased happiness_. Conversely, regions like North America show a negative correlation, highlighting that higher consumption might not necessarily translate to higher happiness.")
    st.write("- **Regional Patterns:** The box plot and bar chart showcase the variability in metrics and highlight significant differences in alcohol consumption across regions, suggesting that cultural and economic factors heavily influence drinking habits.")
    st.write("- **Cultural Influence:** The cultural context appears to shape not just consumption patterns but also overall happiness levels, reflecting the complex interplay between economic indicators, societal norms, and personal well-being.")
    
    st.write("Overall, these insights emphasize the importance of considering both economic and social dimensions when evaluating the impacts of alcohol consumption on well-being. They could serve as a basis for further studies and policy discussions aimed at improving happiness and well-being through both economic development and cultural understanding. However, it is crucial to recognize that many other factors also affect these metrics, adding layers of complexity to the relationship between alcohol consumption and various socioeconomic indicators.")
