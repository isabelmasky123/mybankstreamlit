import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Load the dataset
try:
    data = pd.read_csv('bank_mascarenhas_revised.csv')
except FileNotFoundError:
    st.error("The data file was not found. Please check the file path.")
except Exception as e:
    st.error(f"An error occurred: {e}")

# App Title and Description
st.title("Interactive Data Pattern Exploration App")
st.title("Problem Statement & Objective:")
st.write("Using the Bank 01 dataset, I will explore patterns in the customer sales data and also determine relationships between different sales related variables and demographic features. How does customer demographic information like gender, income, homeownership, or age correlate with sales metrics? And what are the trends in customer sales over time?")

# Sidebar Filters
st.sidebar.header("Filter Options")

# Sidebar Filters for Numerical Variables
st.sidebar.subheader("Numerical Filters")
sales_range = st.sidebar.slider("New Sales (int_tgt)", min_value=int(data['int_tgt'].min()), max_value=int(data['int_tgt'].max()), value=(0, 500000))
sales_past_3_range = st.sidebar.slider("Average Sales Past Three Years (rfm1)", min_value=int(data['rfm1'].min()), max_value=int(data['rfm1'].max()), value=(0, 500))
sales_lifetime_range = st.sidebar.slider("Average Sales Lifetime (rfm2)", min_value=int(data['rfm2'].min()), max_value=int(data['rfm2'].max()), value=(0, 309))

# Sidebar Filters for Categorical Variables
st.sidebar.subheader("Categorical Filters")
gender = st.sidebar.multiselect("Gender", options=["female", "male"], default=["female", "male"])
homeowner = st.sidebar.multiselect("Homeownership", options=["yes", "no"], default=["yes", "no"])

# Filter data based on gender
if "female" in gender:
    filtered_data = data[data['demog_genf'] == 'yes']
if "male" in gender:
    filtered_data = data[data['demog_genm'] == 'yes']

# Apply other filters (numerical and categorical)
filtered_data = filtered_data[
    (filtered_data['int_tgt'].between(*sales_range)) &
    (filtered_data['rfm1'].between(*sales_past_3_range)) &
    (filtered_data['rfm2'].between(*sales_lifetime_range)) &
    (filtered_data['demog_ho'].isin(homeowner))
]

# Handle missing data
filtered_data = filtered_data.dropna(subset=['int_tgt', 'rfm1', 'rfm2', 'demog_age', 'demog_inc', 'demog_homeval'])

# Show filtered data if user selects the option
if st.sidebar.checkbox("Show Filtered Data"):
    st.write(filtered_data)

# Check if filtered data is empty
st.write("Filtered Data Shape: ", filtered_data.shape)
if filtered_data.empty:
    st.warning("No data available for the selected filters.")
else:
    st.title("Univariate Analysis:")
    # Histogram
    st.header("Distribution of New Sales (int_tgt)")
    fig, ax = plt.subplots()
    sns.histplot(filtered_data['int_tgt'], bins=20, color='skyblue', kde=False, ax=ax)
    ax.set_title("Histogram of New Sales (int_tgt)")
    ax.set_xlabel("New Sales")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.title("Bivariate Analysis:")
    # Scatter Plot
    st.header("Scatter Plot: Average Sales Past Three Years vs. Average Sales Lifetime")
    show_trendline = st.checkbox("Show Trendline", value=False)
    fig = px.scatter(filtered_data, x='rfm1', y='rfm2', title="Average Sales Past Three Years vs. Average Sales Lifetime", trendline="ols" if show_trendline else None)
    st.plotly_chart(fig)

    # Correlation Matrix

    st.header("Correlation Matrix")
    st.write("Check the box to view the correlation matrix for numeric variables.")
    continuous_vars = ['int_tgt', 'rfm1', 'rfm2', 'rfm3', 'rfm4', 'rfm5', 'rfm6', 'rfm7', 'rfm8', 'rfm9', 
                   'rfm10', 'rfm11', 'rfm12', 'demog_age', 'demog_homeval', 'demog_inc', 'demog_pr']
    if st.checkbox("Show Correlation Matrix"):
        corr_matrix = filtered_data[continuous_vars].corr()
        st.write("NaNs in correlation matrix:", corr_matrix.isna().sum().sum())
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            ax=ax,
            annot_kws={"size": 8}  # Bigger or smaller depending on space
    )
        plt.xticks(rotation=45, ha='right')  # Improve label readability
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)


    # Customer Age vs. Income
    st.header("Customer Age vs. Income")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_data, x='demog_age', y='demog_inc', color='green', ax=ax)
    ax.set_title("Customer Age vs. Income")
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    st.pyplot(fig)


st.title("Insights:")
st.write("The distribution of INT_TGT is quite right-skewed. This means most customers have relatively low new sales, while a much smaller number of customers have very high new sales.The majority of the data seems to fall between 0 and 25,000.")
st.write("There is a positive linear relationship between RFM1 vs. RFM2 in that customers who purchased more over the last three years also tended to have higher total purchases over their lifetime.")
st.write("RFM1, RFM2, and RFM4 are all moderately positively correlated with INT_TGT. RFM1 has a value of 0.48, RFM2 has a value of 0.37, and RFM4 has a value of 0.44. This may mean that the customer's past buying behavior is a good predictor of future sales.")
st.write("For all people (males, females, homeowners, and non-homeowners) customer age does not seem to impact income. People can be of all ages and yet still have low incomes, or be young and have very high incomes. Many customers have an income of 0 all the way from ages 15-100. Therefore, there is not a strong correlation between the two.")

st.title("Recommendations:")
st.write("As RFM1, RFM2, and RFM4 are moderately correlated with future sales (INT_TGT), these variables should be prioritized in predictive models and marketing campaigns. Customers who have historically made frequent purchases (RFM1), have a high lifetime spend (RFM2), or show good retention behaviors (RFM4) are likely to continue purchasing, thus driving future sales. Maybe we could separate customers based on RFM metrics and target those with high RFM1, RFM2, and RFM4 values as they would be the ones who will maximize the possibility of new sales. We also want to understand who the customers are above the 25000 INT_TGT mark. We need to understand why those people are spending so much more and if there are more targeted campaigns that can be made to retain those customers who spend a lot.")