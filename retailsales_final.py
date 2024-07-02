import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
import base64  
import numpy as np
import pickle
from datetime import date
from streamlit_extras.add_vertical_space import add_vertical_space

def style_submit_button():
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #0F52BA;
            color: white;
            width: 70%}
        </style>
    """, unsafe_allow_html=True)

# Streamlit part
st.set_page_config(layout="wide")

st.title("Retail Sales Prediction 💵  ")
st.write(" 🧑‍💻 Tech Used: ML/DL deployment, AWS sagemaker, Time Series Analysis, Predictive Modeling, Data Preprocessing, Feature Engineering, Exploratory Data Analysis (EDA), Model Evaluation and Validation, Impact Analysis of Promotions and Holidays, Data Visualization.")

def datafr():
    df = pd.read_csv(r"K:\DS\retail_sales_prediction_fp\data_final.csv")
    return df

df = datafr()
    
select = option_menu(
    menu_title=None,
    options=["About RSP", "Insights", "Prediction"],
    icons=["graph-up", "clipboard2-data", "bounding-box"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

# HOME PAGE
if select == "About RSP":
    st.header("Project Overview :signal_strength:")
    st.write("")
    st.write('''***Retail Sales Forecast employs advanced machine learning techniques, 
             prioritizing careful data preprocessing, feature enhancement, and comprehensive 
             algorithm assessment and selection. The streamlined Streamlit application integrates 
             Exploratory Data Analysis (EDA) to find trends, patterns, and data insights. 
             It offers users interactive tools to explore top-performing stores and departments, 
             conduct insightful feature comparisons, and obtain personalized sales forecasts. 
             With a commitment to delivering actionable insights, the project aims to optimize 
             decision-making processes within the dynamic retail landscape.***''')
    st.header("Technologies used ⚛️")
    st.write("")
    st.write("***Python, Pandas, Plotly, Streamlit, Scikit-Learn, Numpy, Seaborn***")

# OVERVIEW PAGE
if select == "Insights":
    # Convert Month and Year to a datetime object
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

    # Group by Date and sum Weekly_Sales
    sales_over_time = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
    fig = px.line(sales_over_time,
                  title='Weekly Sales Over Time',
                  x='Date',
                  y='Weekly_Sales')
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='Total Weekly Sales',
                      xaxis=dict(tickangle=45),
                      template='plotly_white')
    st.plotly_chart(fig, use_container_width=True) 

    # Distribution of Weekly Sales by Holiday
    fig = px.box(df,
                 title='Distribution of Weekly Sales During Holidays',
                 x='IsHoliday',
                 y='Weekly_Sales')
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate the Weekly_Sales by Store and Department
    store_dept_sales = df.groupby(['Store'])['Weekly_Sales'].sum().reset_index()

    # Sort the aggregated data to identify top-performing and underperforming stores/departments
    top_performing = store_dept_sales.sort_values(by='Weekly_Sales', ascending=False).head(10)

    fig = px.bar(data_frame=top_performing,
                 x='Store',
                 y='Weekly_Sales',
                 title='Top-Performing Stores'
                )
    fig.update_layout(xaxis_title='Store',
                      yaxis_title='Weekly Sales',
                      width=800,
                      height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Compute the correlation matrix
    correlation_matrix = df[['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']].corr()
    fig = px.imshow(correlation_matrix,
                    title='Correlation Heatmap of Weekly Sales and Numerical Features',
                    text_auto=True,
                    width=800, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate the Weekly_Sales by Department
    dept_sales = df.groupby(['Dept'])['Weekly_Sales'].sum().reset_index()

    # Sort the aggregated data to identify top-performing and underperforming departments
    top_dept = dept_sales.sort_values(by='Weekly_Sales', ascending=False).head(10)

    fig = px.bar(top_dept,
                 x='Dept',
                 y='Weekly_Sales',
                 title='Top-Performing Department across Stores'
                )
    fig.update_layout(xaxis_title='Department',
                      yaxis_title='Total Weekly Sales',
                      xaxis={'categoryorder': 'total descending'}, 
                      bargap=0.1, 
                      width=1700,
                      height=600)
    st.plotly_chart(fig, use_container_width=True)
            
    # Aggregate the Weekly_Sales by Store and Department
    store_dept_sales = df.groupby(['Store'])['Weekly_Sales'].sum().reset_index()

    # Sort the aggregated data to identify top-performing and underperforming stores/departments
    underperforming = store_dept_sales.sort_values(by='Weekly_Sales').head(10)

    fig = px.bar(underperforming,
                 x='Store',
                 y='Weekly_Sales',
                 title='Under-Performing Stores'
                )
    fig.update_layout(xaxis_title='Store',
                      yaxis_title='Total Weekly Sales',
                      xaxis={'categoryorder': 'total descending'},
                      bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
            
# Prediction PAGE
if select == "Prediction":
    with st.form('prediction'):
        col1, col2, col3 = st.columns([0.5, 0.1, 0.5])

        with col1:
            user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                      max_value=date(2013, 12, 31), value=date(2010, 2, 5))
            store = st.number_input(label='Store', min_value=1, max_value=45,
                                    value=1, step=1)
            dept = st.selectbox(label='Department', options=df.Store.unique())
            holiday = st.selectbox(label='Holiday', options=['Yes', 'No'])
            temperature = st.number_input(label='Temperature(°F)', min_value=-10.0,
                                          max_value=110.0, value=-7.29)
            fuel_price = st.number_input(label='Fuel Price', max_value=10.0,
                                         value=2.47)
            cpi = st.number_input(label='CPI', min_value=100.0,
                                  max_value=250.0, value=126.06)    

        with col3:
            type = st.number_input(label='Type', min_value=1, max_value=3,
                                   value=1, step=1)
            size = st.selectbox(
                "Size",
                options=[151315, 202307,  37392, 205863,  34875, 202505,  70713, 155078,
                         125833, 126512, 207499, 112238, 219622, 200898, 123737,  57197,
                         93188, 120653, 203819, 203742, 140167, 119557, 114533, 128107,
                         152513, 204184, 206302,  93638,  42988, 203750, 203007,  39690,
                         158114, 103681,  39910, 184109, 155083, 196321,  41062, 118221],
            )
            markdown1 = st.number_input(label='MarkDown1', value=-271.45)
            markdown2 = st.number_input(label='MarkDown2', value=-265.76)
            markdown3 = st.number_input(label='MarkDown3', value=-179.26)
            markdown4 = st.number_input(label='MarkDown4', value=0.22)
            markdown5 = st.number_input(label='MarkDown5', value=-185.87)
            unemployment = st.number_input(label='Unemployment', max_value=20.0, value=3.68)

        add_vertical_space(2)

        c1, c2, c3 = st.columns([0.5, 0.1, 0.5])
        with c1:
            button = st.form_submit_button(label='PREDICT WITH MARKDOWN')
            style_submit_button()
        with c3:  
            button2 = st.form_submit_button(label='PREDICT WITHOUT MARKDOWN')
            style_submit_button()

    # User entered all input values and clicked the button
    if button:
        with st.spinner(text='Processing...'):
            # Load the regression pickle model
            with open(r"K:\DS\retail_sales_prediction_fp\Randomforest_Model2.pkl", 'rb') as f:
                model = pickle.load(f)

            holiday_dict = {'Yes': 1, 'No': 0}

            # Make array for all user input values in required order for model prediction
            user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                   store, dept, type, size,
                                   holiday_dict[holiday], temperature,
                                   fuel_price, markdown1, markdown2, markdown3,
                                   markdown4, markdown5, cpi, unemployment]])

            # Model predict the selling price based on user input
            y_pred = model.predict(user_data)[0]

            # Round the value with 2 decimal point (Eg: 1.35678 to 1.36)
            weekly_sales = f"{y_pred:.2f}"

            if weekly_sales:
                # Apply custom CSS style for prediction text
                st.markdown(
                    """
                    <style>
                    .center-text {
                        text-align: center;
                        color: #FF3131
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(f'### <div class="center-text">Predicted Sales with markdown is {weekly_sales}</div>', 
                            unsafe_allow_html=True)   

    if button2:
        with st.spinner(text='Processing...'):
            # Load the regression pickle model
            with open(r"K:\DS\retail_sales_prediction_fp\Randomforest_Model1.pkl", 'rb') as f:
                model = pickle.load(f)

            holiday_dict = {'Yes': 1, 'No': 0}

            # Make array for all user input values in required order for model prediction
            user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                   store, dept, type, size,
                                   holiday_dict[holiday], temperature,
                                   fuel_price, cpi, unemployment]])

            # Model predict the selling price based on user input
            y_pred = model.predict(user_data)[0]

            # Round the value with 2 decimal point (Eg: 1.35678 to 1.36)
            weekly_sales = f"{y_pred:.2f}"

            if weekly_sales:
                # Apply custom CSS style for prediction text
                st.markdown(
                    """
                    <style>
                    .center-text {
                        text-align: center;
                        color: #FF3131    
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(f'### <div class="center-text">Predicted Sales without markdown is {weekly_sales}</div>', 
                            unsafe_allow_html=True) 
