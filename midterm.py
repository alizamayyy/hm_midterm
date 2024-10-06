# midterm.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st

df = pd.read_csv('HappinessAlcoholConsumption.csv')

# Calculate the average alcohol consumption per capita
df['Avg_Alcohol_PerCapita'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)

# Define the columns to be used in various analyses
columns = ['HappinessScore', 'HDI', 'GDP_PerCapita', 'Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']

st.title('Happiness and Alcohol Consumption Analysis')

# Sidebar for navigation
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Cleaning", 
                                 "Data Visualization", "Conclusion"])

if options == "Introduction":
    st.markdown("<small>by Halimaw Magbeg</small>", unsafe_allow_html=True)  # Add the author's name in small font
    st.write("Happiness and well-being are increasingly central themes in global discussions about public health, societal development, and policy making. One aspect of human behavior that has been closely studied in relation to mental health and happiness is alcohol consumption.")
    st.write("Does higher alcohol consumption lead to lower happiness levels? Or are happier countries more likely to have higher alcohol consumption due to social factors?")
    st.write("The aim of this analysis is to explore how alcohol consumption correlates with happiness levels.")
    st.write("The dataset is taken from [Kaggle](https://www.kaggle.com/datasets/marcospessotto/happiness-and-alcohol-consumption).")

elif options == "Data Exploration and Cleaning":
    st.subheader("Data Overview")
    st.write(df.head())
    st.write("### Data Cleaning")
    st.write(df.isnull().sum()) 

    st.subheader("Descriptive Statistics")

    # Create a list to hold the statistics for each column
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
            "50th Percentile (Median)": percentiles[1],
            "75th Percentile": percentiles[2]
        })

    # Create a DataFrame from the statistics list
    stats_df = pd.DataFrame(stats_list)

    # Display the DataFrame as a table in Streamlit
    st.dataframe(stats_df) 

elif options == "Data Visualization":
    visualization_option = st.sidebar.selectbox("Select a visualization type:", 
                                                 ["Heatmap", "Box Plot", "Histogram", 
                                                  "Grouped Bar Chart", "Bar Chart", 
                                                  "Horizontal Bar Chart", "Choropleth Map", 
                                                  "Pair Plot", "Scatter Plots"])

    if visualization_option == "Heatmap":
        st.subheader("Correlation Heatmap")
    
        df_selected = df[columns]
        correlation_matrix = df_selected.corr()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Happiness and Alcohol Consumption')
        st.pyplot(plt)

        st.write("The heatmap indicates that improving HDI and GDP may play significant roles in enhancing happiness, while alcohol consumption, particularly beer, contributes positively but is less central to the overall happiness score.")

    elif visualization_option == "Box Plot":
        st.subheader("Box Plot of Various Metrics")
        df_melted = df[columns].melt(var_name='Metric', value_name='Value')
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Value', y='Metric', data=df_melted, hue='Metric', palette='viridis', dodge=False)
        plt.title('Box Plot of Various Metrics')
        plt.xlabel('Value')
        plt.ylabel('Metric')
        plt.grid(True)
        st.pyplot(plt)
        
        st.write("Potential correlations exist between beer consumption and higher happiness scores, suggesting social aspects of beer drinking may positively impact well-being.")


    elif visualization_option == "Histogram":
        st.subheader("Histograms of Various Metrics")
    
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(columns):
            plt.subplot(2, 3, i + 1) 
            sns.histplot(df[column], bins=20, kde=True, color='blue', alpha=0.7)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

        plt.tight_layout()  
        plt.suptitle('Histograms of Various Metrics', y=1.02)  
        st.pyplot(plt)
        
        st.write("The histograms provide a visual representation of the distribution of various metrics related to happiness, human development, economic prosperity, and alcohol consumption.")


    elif visualization_option == "Grouped Bar Chart":
        st.subheader("Grouped Bar Chart of Average Alcohol Consumption by Region")
        region_alcohol_consumption = df.groupby('Region')[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean().reset_index()
        region_alcohol_consumption_melted = region_alcohol_consumption.melt(id_vars='Region', var_name='Alcohol_Type', value_name='PerCapita')
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Region', y='PerCapita', hue='Alcohol_Type', data=region_alcohol_consumption_melted, palette='viridis')
        plt.xticks(rotation=45)
        plt.title('Average Alcohol Consumption Per Capita by Region')
        plt.xlabel('Region')
        plt.ylabel('Average Per Capita Consumption')
        plt.legend(title='Alcohol Type')
        st.pyplot(plt)
        
        st.write("Consumption patterns highlight distinct regional preferences, with Western Europe favoring wine, while North America and Central & Eastern Europe prefer beer and spirits, respectively.")


    elif visualization_option == "Bar Chart":
        st.subheader("Bar Chart of Average Alcohol Consumption")
        mean_values = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean().reset_index()
        mean_values.columns = ['Alcohol Type', 'Average Consumption']
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Alcohol Type', y='Average Consumption', data=mean_values, palette='viridis')
        plt.title('Average Alcohol Consumption by Type')
        plt.xlabel('Type of Alcohol')
        plt.ylabel('Average Consumption')
        st.pyplot(plt)
        
        st.write("These trends underscore varying cultural preferences for alcohol types, which can inform public health initiatives and market strategies tailored to consumer behavior.")

    elif visualization_option == "Horizontal Bar Chart":
        st.subheader("Horizontal Bar Chart of Average Happiness Scores by Region")
        avg_happiness_by_region = df.groupby('Region')['HappinessScore'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='HappinessScore', y='Region', data=avg_happiness_by_region, palette='viridis')
        plt.title('Average Happiness Scores by Region')
        plt.xlabel('Average Happiness Score')
        plt.ylabel('Region')
        st.pyplot(plt)
        
        st.write("The chart highlights the top 10 countries with the highest total alcohol consumption per capita, measured in liters.")


    elif visualization_option == "Choropleth Map":
        st.subheader("Choropleth Map of Average Alcohol Consumption")
        
        fig = px.choropleth(df, 
                            locations='Country', 
                            locationmode='country names', 
                            color='Avg_Alcohol_PerCapita',
                            hover_name='Country', 
                            color_continuous_scale='Blues',
                            labels={'Avg_Alcohol_PerCapita': 'Average Alcohol Consumption (L)'},
                            title='Average Alcohol Consumption Per Capita by Country')
        
        st.plotly_chart(fig)
        
        st.write("This map visualizes the average alcohol consumption per capita by country, where each country is colored according to the level of alcohol consumption.")

    elif visualization_option == "Pair Plot":
        st.subheader("Pair Plot of Various Metrics with Trendlines")
        g = sns.pairplot(df[columns], kind='reg', plot_kws={
            'scatter_kws': {'s': 50, 'alpha': 0.5},
            'line_kws': {'color': 'red', 'linewidth': 1}
        })
        plt.suptitle('Pair Plot of Various Metrics with Trendlines', y=1.02)
        st.pyplot(plt)
        
        st.write("The graph indicates that improving HDI and economic conditions can significantly enhance happiness levels.")

    elif visualization_option == "Scatter Plots":
        st.subheader("Scatter Plots")
        
        # HDI vs Happiness Score
        plt.figure(figsize=(10, 6))
        sns.lmplot(x='HDI', y='HappinessScore', hue='Region', data=df, 
                   palette='Set2', height=6, aspect=1.5, scatter_kws={'s': 50}, ci=None)
        plt.title('HDI vs Happiness Score with Trendlines')
        plt.xlabel('Human Development Index (HDI)')
        plt.ylabel('Happiness Score')
        st.pyplot(plt)
        
        st.write("The graph effectively illustrates that while there is a general trend of increased happiness with higher HDI, regional differences suggest a complex interplay of various factors influencing happiness.")

        # HDI vs Average Alcohol Consumption
        plt.figure(figsize=(10, 6))
        sns.lmplot(x='Avg_Alcohol_PerCapita', y='HDI', hue='Region', data=df, 
                   palette='Set2', height=6, aspect=1.5, scatter_kws={'s': 50}, ci=None)
        plt.title('HDI vs Average Alcohol Consumption Per Capita with Trendlines')
        plt.xlabel('Average Alcohol Consumption Per Capita (L)')
        plt.ylabel('Human Development Index (HDI)')
        st.pyplot(plt)
        
        st.write("There is a positive correlation between average alcohol consumption per capita and Human Development Index (HDI) across most regions, indicating that higher alcohol consumption often coincides with higher HDI.")

        # Happiness Score vs Average Alcohol Consumption
        plt.figure(figsize=(10, 6))
        sns.lmplot(x='Avg_Alcohol_PerCapita', y='HappinessScore', hue='Region', data=df, 
                   palette='Set2', height=6, aspect=1.5, scatter_kws={'s': 50}, ci=None)
        plt.title('Happiness Score vs Average Alcohol Consumption Per Capita with Trendlines')
        plt.xlabel('Average Alcohol Consumption Per Capita (L)')
        plt.ylabel('Happiness Score')
        st.pyplot(plt)
        
        st.write("The relationship between alcohol consumption and happiness scores varies significantly across different regions. While some regions show a positive correlation, others show little to no correlation or even a negative correlation.")

elif options == "Conclusion":
    st.write("### Conclusion")
    st.write("This analysis provides insights into the relationship between happiness and alcohol consumption.")
    
    st.write("**Key findings include:**")
    st.write("- **Correlations:** A moderate positive correlation exists between happiness scores and GDP per capita and HDI, emphasizing the importance of economic factors in influencing happiness. Moderate correlations between happiness and beer, spirits, and wine suggest a nuanced relationship between alcohol consumption and happiness.")
    st.write("- **Regional Patterns:** The box plot and bar chart showcase the variability in metrics and highlight significant differences in alcohol consumption across regions, suggesting that cultural and economic factors heavily influence drinking habits.")
    st.write("- **Cultural Influence:** The cultural context appears to shape not just consumption patterns but also overall happiness levels, reflecting the complex interplay between economic indicators, societal norms, and personal well-being.")
    
    st.write("Overall, the insights gleaned from this dataset could serve as a basis for further studies and policy discussions aimed at improving happiness and well-being through both economic development and cultural understanding.")