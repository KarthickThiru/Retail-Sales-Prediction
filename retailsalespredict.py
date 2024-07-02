import pandas as pd
import streamlit as st
import plotly.express as px
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

#Function to display the sidebar background

def sidebar_bg(side_bg):

   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
      }}
      </style>
      """,

      
      unsafe_allow_html=True,
      )
   
side_bg = r'K:\DS\retail_sales_prediction_fp\rspfp_img_2.png'
sidebar_bg(side_bg)

with st.sidebar:
    st.title("About the Project")

st.title(":blue[WEEKLY SALES PREDICTION]")
# load store dataset
data = pd.read_csv(r"K:\DS\retail_sales_prediction_fp\data_final.csv")

# get input from users
with st.form('prediction'):

    col1, col2, col3 = st.columns([0.5, 0.1, 0.5])

    with col1:

        user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                    max_value=date(2013, 12, 31), value=date(2010, 2, 5))

        store = st.number_input(label='Store', min_value=1, max_value=45,
                                value=1, step=1)

        dept = st.selectbox(label='Department',
                            options=data.Store.unique())

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
                options = [151315, 202307,  37392, 205863,  34875, 202505,  70713, 155078,
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

        unemployment = st.number_input(label='Unemployment',
                                        max_value=20.0, value=3.68)

    add_vertical_space(2)

    c1, c2, c3 = st.columns([0.5, 0.1, 0.5])
    with c1:
        button = st.form_submit_button(label='PREDICT WITH MARKDOWN')
        style_submit_button()
    with c3:  
        button2 = st.form_submit_button(label='PREDICT WITHOUT MARKDOWN')
        style_submit_button()

# user entered the all input values and click the button
if button:
    with st.spinner(text='Processing...'):

        # load the regression pickle model
        with open(r"K:\DS\retail_sales_prediction_fp\Randomforest_Model2.pkl", 'rb') as f:
            model = pickle.load(f)

        holiday_dict = {'Yes': 1, 'No': 0}
       # type_dict, size_dict = prediction.type_size_dict()

        # make array for all user input values in required order for model prediction
        user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                store, dept,type,size,
                                holiday_dict[holiday], temperature,
                                fuel_price, markdown1, markdown2, markdown3,
                                markdown4, markdown5, cpi, unemployment]])

        # model predict the selling price based on user input
        y_pred = model.predict(user_data)[0]

        # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
        weekly_sales = f"{y_pred:.2f}"

        if weekly_sales:

        # apply custom css style for prediction text
            st.markdown(
            """
                <style>
                .center-text {
                    text-align: center;
                    color: #FF3131
                }
                </style>
                """,
            unsafe_allow_html=True)

            st.markdown(f'### <div class="center-text">Predicted Sales with markdown is {weekly_sales}</div>', 
                        unsafe_allow_html=True)   

if button2:
    with st.spinner(text='Processing...'):

        # load the regression pickle model
        with open(r"K:\DS\retail_sales_prediction_fp\Randomforest_Model1.pkl", 'rb') as f:
            model = pickle.load(f)

        holiday_dict = {'Yes': 1, 'No': 0}
       # type_dict, size_dict = prediction.type_size_dict()

        # make array for all user input values in required order for model prediction
        user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                store, dept,type,size,
                                holiday_dict[holiday], temperature,
                                fuel_price,  cpi, unemployment]])

        # model predict the selling price based on user input
        y_pred = model.predict(user_data)[0]

        # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
        weekly_sales = f"{y_pred:.2f}"

        if weekly_sales:

        # apply custom css style for prediction text
            st.markdown(
            """
                <style>
                .center-text {
                    text-align: center;
                    color: #FF3131	
                }
                </style>
                """,
            unsafe_allow_html=True)

            st.markdown(f'### <div class="center-text">Predicted Sales without markdown is {weekly_sales}</div>', 
                        unsafe_allow_html=True)                 