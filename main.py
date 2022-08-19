from asyncio import futures
import streamlit as st 
from datetime import date 
import yfinance as yf 
from prophet import * 


from plotly import graph_objects as go 


START= "2015-01-01"
TODAY= date.today()

st.title("Stock Prediction App ")

stocks= ("AAPL", "TSLA", "MSFT", "GOOG", "IBM")
select_stocks= st.selectbox("Select Datset", stocks)

n_years = st.slider("Years of Prediction", 1, 4)
perriod = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data 



data_load_state = st.text("Load Data...")
data =load_data(select_stocks)
data_load_state.text("Loading Data...Done")



st.subheader('Raw Data')
st.write(data.tail())


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    
plot_raw_data()



#forecasting 

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=perriod)
forecast = m.predict(future)




st.write("Forecast Components")
fig2=m.plot_components(forecast)
st.write(fig2)