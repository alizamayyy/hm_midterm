import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import base64

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

    st.write("- **Happiness Score and GDP Per Capita**: The correlation is positive but weaker than with HDI, suggesting that while wealth contributes to happiness, the quality of life, as reflected in HDI, is also important.")
    st.write("- **Happiness Score and Alcohol Consumption**: Moderate positive correlations are observed between happiness and all types of alcohol consumption, with beer showing the strongest correlation, possibly linked to social interactions.")
    st.write("- **Interrelations among Alcohol Types**: Moderate correlations among beer, spirits, and wine indicate that high consumption of one type often correlates with high consumption of others, suggesting shared cultural factors.")
    st.write("Overall, the heatmap indicates that improving HDI and GDP may play significant roles in enhancing happiness, while alcohol consumption, particularly beer, contributes positively but is less central to the overall happiness score.")

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
    st.write("🔴**High Alcohol Consumption**")
    st.markdown("""
    <ul>
        <li>Countries in Eastern Europe and parts of Northern Europe (e.g., Russia, Belarus) are shaded in dark blue, indicating high average alcohol consumption.</li>
        <li>Some areas in Oceania, such as Australia, also show relatively high levels of alcohol consumption.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.write("🟡**Moderate Alcohol Consumption**")
    st.markdown("""
    <ul>
        <li>Countries in Western Europe (such as Germany, France, and the UK) display moderate shades of blue, suggesting a moderate level of alcohol consumption per capita (between 100 and 150 liters).</li>
    </ul>
    """, unsafe_allow_html=True)

    st.write("🟢**Low Alcohol Consumption**")
    st.markdown("""
    <ul>
        <li>Many countries in Africa, parts of Asia, and South America are shaded in very light blue or white, indicating low or minimal alcohol consumption per capita.</li>
    </ul>
    """, unsafe_allow_html=True)

    

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
    st.write("The graph illustrates significant regional disparities in alcohol consumption, reflecting cultural, economic, and social factors. Western Europe shows the highest wine consumption, while North America and Central & Eastern Europe prefer beer and spirits, respectively. Regions like Africa and Asia exhibit lower averages across all types.")
    st.write("Consumption patterns highlight distinct regional preferences, with Western Europe favoring wine, while North America and Central & Eastern Europe prefer beer and spirits, respectively.")
    
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
    st.write("This chart highlights the top 10 regions with the highest average happiness scores.")
    st.write("🇨🇿 **Czech Republic**: Leads in total alcohol consumption per capita, highlighting a strong cultural acceptance and integration of alcohol in daily life.")   
    st.write("🇷🇺 **Russia**: Following closely, indicating a significant cultural inclination towards alcohol, particularly spirits like vodka, which are embedded in social traditions.")
    st.write("🇫🇷 **France**: Ranks third, reflecting its renowned wine culture, where alcohol consumption is often associated with meals and social gatherings.")
    st.write("🇩🇪 **Germany**: High consumption levels, influenced by its beer culture and festivals like Oktoberfest, which celebrate communal drinking.")
    st.write("🇱🇹 **Lithuania**, 🇱🇺 **Luxembourg**, 🇭🇺 **Hungary**, 🇸🇰 **Slovakia**, 🇵🇱 **Poland**, and 🇵🇹 **Portugal**: These countries show moderate to high alcohol consumption, suggesting a robust cultural presence of alcohol that may impact public health and social dynamics.")

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
    
    st.markdown("""
    <ul>
        <li>This graph suggests that while the general trend shows a negative relationship between GDP and alcohol consumption, it varies significantly across regions. Cultural factors (e.g., religion, societal norms) likely play a major role in shaping alcohol consumption patterns, sometimes overriding economic factors.</li>
        <li>While the trendline suggests a slight negative correlation between GDP per capita and alcohol consumption, this relationship is weak and varies across regions.</li>
        <li>The Western Europe and North America regions show relatively high alcohol consumption paired with moderate to high GDP, standing out from the global trend.</li>
        <li>Countries in regions like the Middle East and Northern Africa show low alcohol consumption despite high GDP, likely due to cultural and religious factors.</li>
        <li>The Sub-Saharan Africa and Eastern Asia regions display lower GDP and alcohol consumption, with most countries clustered at the lower end of both metrics.</li>
    </ul>
    """, unsafe_allow_html=True)

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
    
    st.markdown("""
    <ul>
        <li>This graph shows the relationship between the Human Development Index and the Average Alcohol Consumption per capita.</li>
        <li>The plot likely reveals a positive correlation between Average Alcohol Consumption Per Capita and HDI. This suggests that, on average, countries with higher alcohol consumption also tend to have higher HDI scores, indicating better overall development and quality of life.</li>
        <li>Western European countries tend to cluster in the high HDI and high average alcohol consumption area, reflecting cultural acceptance and regulation of alcohol linked to higher socio-economic status. In contrast, Sub-Saharan Africa shows a cluster of lower HDI and alcohol consumption, indicating socio-economic challenges that affect both development and drinking patterns.</li>
        <li>There is a positive correlation between average alcohol consumption per capita and Human Development Index (HDI) across most regions, indicating that higher alcohol consumption often coincides with higher HDI.</li>
    </ul>
    """, unsafe_allow_html=True)
    
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
    st.markdown("""
    <ul>
        <li>This graph shows the relationship between the Happiness Score and Average Alcohol Consumption per capita, grouped by region.</li>
        <li>There appears to be a positive correlation between average alcohol consumption and happiness scores. As average alcohol consumption increases, happiness scores tend to rise, suggesting that higher alcohol consumption may be associated with greater levels of reported happiness in certain contexts.</li>
        <li>The relationship between alcohol consumption and happiness scores varies significantly across different regions. While some regions show a positive correlation, others show little to no correlation or even a negative correlation.</li>
    </ul>
    """, unsafe_allow_html=True)

# Sidebar
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Preparation", 
                                 "Data Visualization", "Conclusion"])

# Application
st.header("Happiness and Alcohol Consumption Analysis 🍺")
st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True) 

if options == "Introduction":
   
    st.write("Happiness and well-being are increasingly central themes in global discussions about public health, societal development, and policy making. One aspect of human behavior that has been closely studied in relation to mental health and happiness is alcohol consumption.")
    st.write(" **_Does higher alcohol consumption lead to lower happiness levels?_** Or are **_happier countries more likely to have higher alcohol consumption due to social factors?_**")
    st.write("The aim of this analysis is to explore how alcohol consumption correlates with happiness levels.")
    st.write("The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/marcospessotto/happiness-and-alcohol-consumption).")
    
    st.write("")
    st.write("")
    st.write("")
    st.write("### The Team ✨")
    
            # Function to encode the image
    def img_to_base64(img_path):
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    # Encode the image
    bataluna = img_to_base64("images/bataluna.jpg")
    mier = img_to_base64("images/mier.png")
    alegam = img_to_base64("images/alegam.png")
    madaya = img_to_base64("images/madaya.jpg")
    cabo = img_to_base64("images/cabo.jpg")

    # Create a 3x2 grid of divs with rounded corners, drop shadows, and hover effects
    grid_html = """
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 20px;
            margin-top: 10px;
        }
        .grid-item {
            background-color: #0e1117;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            width: 250px;  /* Set a fixed width */
            height: 250px;
            font-size: 24px;
            font-weight: bold;
            transition: background-color 0.3s ease;
            cursor: pointer;
            flex-direction: column;
            padding-top: 30px;
            
        }
        .grid-item:hover {
            background-color: #8b0026;
        }
        .grid-item img {
            width: 150px;  /* Set a fixed width */
            height: 150px; /* Set a fixed height */
            object-fit: cover; 
            padding-bottom: 10px;
            
            
            border-radius: 100px;  // {{ edit_1 }} Added border-radius for rounded corners
        }
    </style>
    <div class="grid-container">
    """

    # Add items to the grid (5 items)
    grid_items = [
        (bataluna, "Aliza May Bataluna"),
        (mier, "France Gieb Mier"),
        (alegam, "Cielo Alegam"),
        (madaya, "Angela Madaya"),
        (cabo, "Kerch Cabo"),
    ]

    for img, label in grid_items:
        grid_html += f'<div class="grid-item"><img src="data:image/png;base64,{img}" alt="{label}"><p>{label}</p></div>'

    grid_html += "</div>"

    st.markdown(grid_html, unsafe_allow_html=True)


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
