import datetime
from email.contentmanager import raw_data_manager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from plotly import graph_objs as go
from keras.models import load_model
import streamlit as st
import yfinance as yf  

#start = '2010-01-01'
#end = '2020-12-31'

stocks =("AAPL","GOOG","MSFT","SBIN.NS","KOTAKBANK.NS","AXISBANK.NS","INFY.NS","BTC-INR")
st.set_page_config(layout="wide")
def side_display():
    with st.sidebar:
        st.title("Dashboard😊")
        st.subheader("To compare the stocks of more than one company 📊")
        dropdown= st.multiselect('Pick your assets',stocks)

        start_date = st.date_input("Select Starting Date",datetime.date.today())
        end_date = st.date_input("Select Ending Date",datetime.date.today())
    start_date =None if start_date>=datetime.date.today() else start_date
    end_date =None if end_date>=datetime.date.today() else end_date
    return start_date,end_date,dropdown

@st.cache
def load_data(ticker,start,end):
    #st.download_button('Download',data.to_csv())
    data = yf.download(ticker,start,end)
    data.reset_index(inplace=True) 
    return data

def relativeret(df):
    rel  =df.pct_change()
    cumret =(1+rel).cumprod()-1
    cumret = cumret.fillna(0)
    return cumret

st.title("Stock Price Prediction ")
st.subheader("- using python🚀📈ᵇʸ ˢᵘˢʰᵃⁿᵗ ᴮⁱˢʰᵗ ")


kpi1, kpi2,kpi3= st.columns(3)
kpi1.metric(label="$OPEN",value=200.22,delta=-1.4)
kpi2.metric(label="$CLOSE",value=250.71,delta=+2.1)
kpi3.metric(label="$AAPP",value=195.63,delta=+0.2)

selected_stocks = st.selectbox("Select Dataset for prediction ",stocks)

n_years = st.slider("Years of prediction : ",10,20)
period = n_years *365

start_date,end_date,dropdown=side_display()
data_load_state = st.text("Load data................")
data = load_data(selected_stocks,start_date,end_date)
data_load_state.text("Loading done...!!")
if len(dropdown)>0:
    df = relativeret(yf.download(dropdown,start_date,end_date)['Adj Close'])
    st.line_chart(df)





full_display,details_display = st.columns(2)
full_display.subheader("Raw Data")
full_display.write(data)

details_display.subheader('Data Description')
details_display.write(data.describe())

dataploy_open,dataplot_close = st.columns(2)

def plot_open_dataset():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="Stock Open",line=dict(color='green')))
    fig.layout.update(title_text ="Open Price",xaxis_rangeslider_visible=True)
    dataploy_open.plotly_chart(fig)

def plot_close_dataset():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close",line=dict(color='red')))
    fig.layout.update(title_text ="Close Price",xaxis_rangeslider_visible=True)
    dataplot_close.plotly_chart(fig)

plot_open_dataset()
plot_close_dataset()

raw_data_col,moving_average_col = st.columns(2)

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close"))
    fig.layout.update(title_text ="Time series data",xaxis_rangeslider_visible=True)
    raw_data_col.plotly_chart(fig)

def moving_average(data):
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name="Stock Close"))
    fig.add_trace(go.Scatter(x=data['Date'],y=ma100,name="Moving average for 100 values"))
    fig.add_trace(go.Scatter(x=data['Date'],y=ma200,name="Moving average for 200 values"))
    fig.layout.update(title_text ="Moving Average Data",xaxis_rangeslider_visible=True)
    moving_average_col.plotly_chart(fig)


plot_raw_data(data)  
moving_average(data)
df = data.drop(['Date','Adj Close'], axis=1)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days,data_testing],ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test =[]
y_test =[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test ,y_test = np.array(x_test) , np.array(y_test)  
y_predicted = model.predict(x_test)
scale= scaler.scale_  
scale_factor = 1/scale[0]
y_predicted = y_predicted * scale_factor
y_test = y_test*scale_factor

st.subheader("prediction")
fig2 =plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)   