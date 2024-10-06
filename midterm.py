# midterm.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st  # Added Streamlit import

# Load the dataset
df = pd.read_csv('HappinessAlcoholConsumption.csv')

# Calculate the average alcohol consumption per capita
df['Avg_Alcohol_PerCapita'] = df[['Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']].mean(axis=1)

# Define the columns to be used in various analyses
columns = ['HappinessScore', 'HDI', 'GDP_PerCapita', 'Beer_PerCapita', 'Spirit_PerCapita', 'Wine_PerCapita']

# Streamlit app setup
st.title('Happiness and Alcohol Consumption Analysis')

# Sidebar for navigation
options = st.sidebar.selectbox("Select a section:", 
                                ["Introduction", "Data Exploration and Cleaning", 
                                 "Data Visualization", "Conclusion"])

if options == "Introduction":
    st.write("This application analyzes the relationship between happiness and alcohol consumption.")
    st.write("The dataset used contains information on happiness scores and alcohol consumption per capita across various countries.")
    st.write("The aim of this analysis is to explore how alcohol consumption correlates with happiness levels.")

elif options == "Data Exploration and Cleaning":
    st.subheader("Data Overview")
    st.write(df.head())  # Display the first few rows of the DataFrame
    st.write("### Checking for Missing Values")
    st.write(df.isnull().sum())  # Check for missing values
    st.write("### Data Information")
    st.write(df.info())  # Display DataFrame info

    st.subheader("Descriptive Statistics")
    for column in columns:
        st.write(f"Statistics for '{column}':")
        data = df[column]
        
        # Central Tendency
        mean_val = data.mean()
        median_val = data.median()
        mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
        
        st.write(f"  Mean: {mean_val}")
        st.write(f"  Median: {median_val}")
        st.write(f"  Mode: {mode_val}")

        # Spread of Data
        std_val = data.std()
        var_val = data.var()
        
        st.write(f"  Standard Deviation: {std_val}")
        st.write(f"  Variance: {var_val}")

        # Min, Max, and Range
        min_val = data.min()
        max_val = data.max()
        range_val = max_val - min_val
        
        st.write(f"  Min: {min_val}")
        st.write(f"  Max: {max_val}")
        st.write(f"  Range: {range_val}")

        # Percentiles
        percentiles = np.percentile(data.dropna(), [25, 50, 75])
        st.write(f"  25th Percentile: {percentiles[0]}")
        st.write(f"  50th Percentile (Median): {percentiles[1]}")
        st.write(f"  75th Percentile: {percentiles[2]}")

        # Summary Statistics
        summary = stats.describe(data.dropna())
        st.write("\n  Summary Statistics from scipy.stats.describe:")
        st.write(f"  Count: {summary.nobs}")
        st.write(f"  Min: {summary.minmax[0]}")
        st.write(f"  Max: {summary.minmax[1]}")
        st.write(f"  Mean: {summary.mean}")
        st.write(f"  Variance: {summary.variance}")
        st.write(f"  Skewness: {summary.skewness}")
        st.write(f"  Kurtosis: {summary.kurtosis}")
        st.write("\n" + "-"*50)

elif options == "Data Visualization":
    st.subheader("Data Visualization Options")
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

    elif visualization_option == "Histogram":
        st.subheader("Histogram of Happiness Scores")
        plt.figure(figsize=(10, 6))
        sns.histplot(df['HappinessScore'], bins=20, kde=True)
        plt.title('Histogram of Happiness Scores')
        plt.xlabel('Happiness Score')
        plt.ylabel('Frequency')
        st.pyplot(plt)

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

    elif visualization_option == "Horizontal Bar Chart":
        st.subheader("Horizontal Bar Chart of Average Happiness Scores by Region")
        avg_happiness_by_region = df.groupby('Region')['HappinessScore'].mean().reset_index()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='HappinessScore', y='Region', data=avg_happiness_by_region, palette='viridis')
        plt.title('Average Happiness Scores by Region')
        plt.xlabel('Average Happiness Score')
        plt.ylabel('Region')
        st.pyplot(plt)

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

    elif visualization_option == "Pair Plot":
        st.subheader("Pair Plot of Various Metrics with Trendlines")
        g = sns.pairplot(df[columns], kind='reg', plot_kws={
            'scatter_kws': {'s': 50, 'alpha': 0.5},
            'line_kws': {'color': 'red', 'linewidth': 1}
        })
        plt.suptitle('Pair Plot of Various Metrics with Trendlines', y=1.02)
        st.pyplot(plt)

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

        # HDI vs Average Alcohol Consumption
        plt.figure(figsize=(10, 6))
        sns.lmplot(x='Avg_Alcohol_PerCapita', y='HDI', hue='Region', data=df, 
                   palette='Set2', height=6, aspect=1.5, scatter_kws={'s': 50}, ci=None)
        plt.title('HDI vs Average Alcohol Consumption Per Capita with Trendlines')
        plt.xlabel('Average Alcohol Consumption Per Capita (L)')
        plt.ylabel('Human Development Index (HDI)')
        st.pyplot(plt)

        # Happiness Score vs Average Alcohol Consumption
        plt.figure(figsize=(10, 6))
        sns.lmplot(x='Avg_Alcohol_PerCapita', y='HappinessScore', hue='Region', data=df, 
                   palette='Set2', height=6, aspect=1.5, scatter_kws={'s': 50}, ci=None)
        plt.title('Happiness Score vs Average Alcohol Consumption Per Capita with Trendlines')
        plt.xlabel('Average Alcohol Consumption Per Capita (L)')
        plt.ylabel('Happiness Score')
        st.pyplot(plt)

elif options == "Conclusion":
    st.write("### Conclusion")
    st.write("This analysis provides insights into the relationship between happiness and alcohol consumption.")
    st.write("The visualizations and statistics reveal patterns that can help understand how these factors interact.")
    st.write("Further research could explore causal relationships and the impact of other variables.")