import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
# import modin.pandas as pd

from PIL import Image
import os

import datetime 
from datetime import date

from MCForecastTools import MCSimulation
import yfinance as yf

from datetime import date
import perf_analysis as pa
from portfolio_test import *

# input_dir = "/content/investment_risk/data/"
input_dir = "data/"
new_data = True


st.set_page_config(
    page_title="Estimate Risk and Investment Potential in Advanced Portfolio Construction/Management")

# st.title("Financial Planning")
# st.write("©Our Financial Journey")

header_image = Image.open(f"{input_dir}our_financial_journey.png")
st.image(header_image, use_column_width=True)
st.write("© 2021 JV")

symbols = ["VWCE.DE", "0P0000SCQD.F", "UST.PA", "AAPL", "BTC-USD"]

NUM_YEARS = 66

end_date = date.today()
start_date = end_date-datetime.timedelta(days=NUM_YEARS*365)

st.header("**Analysis on Risk and Investment Potential**")
st.write("Take as an example these 5 Assets")
st.write(" ")
df = pd.read_csv(f"{input_dir}assets.csv")
# df = df.set_index('Ticker')

st.table(df)
# st.write("**0P0000SCQD.F** : Euro Government Bond Index Fund ETF - low risk and return")
# st.write("**VWCE.DE** : FTSE All-World Stock UCITS ETF - medium low risk and return - ETF with High diversification between sectors and geography ")
# st.write("**UST.PA** : Nasdaq-100 UCITS ETF - medium high risk and return - Low diversification between sectors and geography - US TOP 100 Tech stocks")
# st.write("**AAPL** : Apple Inc. - high risk and return - Single company stock")
# st.write("**BTC-USD** : Bitcoin - high risk and return - Unregulated Crypto Asset ")

st.write(" ")

st.write("**Sharpe Ratio and Risk-Adjusted Returns measures of an investment**")
st.write(""" "A risk-adjusted return measures an investment's return after taking into account the degree of risk that was taken to achieve it. There are several methods of risk-adjusting performance, such as the Sharpe ratio." """)
st.markdown("Further reading on [Risk-Adjusted Returns and Sharpe Ratio](https://www.investopedia.com/terms/r/riskadjustedreturn.asp).")

df_1 = pd.read_csv(f"{input_dir}risk-adjusted_data.csv")
df_1['Assets'] = df_1['Unnamed: 0']
df_1 = df_1.drop("Unnamed: 0", axis=1).set_index('Assets')

efficient_image = Image.open(f"{input_dir}fig_efficient.png")
st.image(efficient_image, use_column_width=True)

st.table(df_1.style.format({
     'Annualized return': '{:.2%}',
     'Annualized volatility': '{:.2%}',
     'Annualized Sharpe ratio': '{:.2}',        
     'Max drawdown': '{:.2%}',
}))   


st.write("**Correlation and Risk-Adjusted Returns**")
st.write(""" "Modern portfolio theory (MPT) asserts that an investor can achieve diversification and reduce the risk of losses by reducing the correlation between the returns of the assets selected for the portfolio. The goal is to optimize the expected return against a certain level of risk." """)
st.markdown("Further reading on [Correlation and Modern Portfolio Theory](https://www.investopedia.com/ask/answers/030515/how-correlation-used-modern-portfolio-theory.asp).")

correlation = Image.open(f"{input_dir}correlation.png")
st.image(correlation, use_column_width=True)

st.write(" We choose 3 assets (Euro Government Bond Index Fund ETF, High diversified and low cost All-World Stock UCITS ETF and Crypto as an hedge against inflation) for simulating a Risk-Adjusted Portfolio. the choice was based on the data above. They are the least correlated assets, most diversified and with higher sharpe ratio.")

st.header("**Monte Carlo Simulations based on Historical Data**")
button_MC = st.checkbox('What is a Monte Carlo Simulation?', False)

if button_MC:
    st.write("Monte Carlo methods are a class of algorithms that rely on repeated random sampling to obtain numerical results - by the law of large numbers, the expected value of a random variable can be approximated by taking the sample mean of independent samples.")
    st.write("In plain English: while we don't know what the correct answer or strategy is, we can simulate thousands or millions of attempts and the average will approximate the right answer. Monte Carlo simulations can be applied in anything that has a probabilistic element or has risk, e.g. stock market and portfolio performance in Finance etc.")

type_phase = ["Wealth accumulation Phase", "Wealth distribution Phase"]

st.write("Nowadays during the wealth accumulating phase we take out the bonds from the portfolio since the return is too low. Their portfolio role of offering some stability and protection during certain periods of market stress can also be achived by a consistent dollar cost average. But always take into consideration your risk profile.")

colForecast1, colForecast2 = st.columns(2)
with colForecast1:
    simulation_port = st.number_input("Enter the number of simulations.", min_value=200.00)
    simulation_port = int(simulation_port)
with colForecast2:
    phase = st.selectbox('Select your investment phase', type_phase)

st.subheader("**Asset Investment Simulation**")
st.write(f"This tool will calculate the mean expected returns for the asset below, with a starting investment of 1,000€ and using the timespan in years selected below). The results consider a `95%` confidence interval.")

type_ports = [ "Min Volatility" ,"Max Sharpe Ratio"]

colForecast1, colForecast2 = st.columns(2)
with colForecast1:
    simulation_pred = st.number_input("Enter the number of simulations.", min_value=100.00)
    simulation_pred = int(simulation_pred)
    type_port = st.selectbox('Select type of portfolio', type_ports)

with colForecast2:
    forecast_year = st.number_input("Enter your forecast year (min 1 year): ", min_value= 0.0, value=10.0) #, format='%d')
    invest_value = st.number_input("Initial investment value (€): ", min_value= 0.0, value=1000.0) 
    st.write(' ')
#     submit = st.button("Simulate")    


if True: 
    st.error("Keep in mind that Monte Carlo simulatons are highly computational intensive models. A higher number of simulations will give you more accurate results but will also take more time. Please be patient, your results will appear soon.")
    
#     fig_port, df_allocation, df_metrics = output_portfolio(input_dir)
    
#     df_allocation_test = df_allocation.fillna(0)
#     test_1 = df_allocation_test['Max Sharpe Ratio'].to_list()

#     df_1, _ = pa.perfAnalysis(df_allocation_test.index.to_list(), 
#                     portf_weights=df_allocation_test['Max Sharpe Ratio'].to_list(),
#                     start=date(2000,1,4), end=date.today(),
#                     riskfree_rate=0.01,  init_cap=1000,
#                     chart_size=(22,12))


#     df_1 = df_1[["Annualized return", "Annualized volatility", "Annualized Sharpe ratio", "Max drawdown"]][:-1]

    
    st.subheader("**Risk-Adjusted Portfolio**")


    if phase == "Wealth accumulation Phase":

        fig_port, df_allocation, df_metrics = output_portfolio(input_dir, acc=True, num_sim=simulation_port)

        symbols = df_allocation.index.to_list()

        symbols = df_allocation.index.to_list()
        dfs = {}
        for i, s in enumerate(symbols):
            dfs[s] = yf.download(s,start_date,end_date)#['Adj Close']

            list_allocation = []

            ticker = list(dfs.keys())[i]

            df1 = dfs[ticker][["Open","High", "Low","Close", "Volume"]]

            df1 = df1.rename(columns={"Open": "open", "High": "high", "Low":"low", "Close": "close", "Volume":"volume"})

            tuples = [(ticker,'open'), 
                      (ticker,'high'), 
                      (ticker,'low'), 
                      (ticker,'close'),
                      (ticker,'volume'),
                     ]

            df1.columns = pd.MultiIndex.from_tuples(tuples)
            if i == 0:
                df = df1
            else:
                df = pd.concat([df, df1], axis=1).dropna()
    else:

        fig_port, df_allocation, df_metrics = output_portfolio(input_dir, num_sim=simulation_port)

        symbols = df_allocation.index.to_list()

        symbols = df_allocation.index.to_list()
        dfs = {}
        for i, s in enumerate(symbols):
            dfs[s] = yf.download(s,start_date,end_date)#['Adj Close']

            list_allocation = []

            ticker = list(dfs.keys())[i]

            df1 = dfs[ticker][["Open","High", "Low","Close", "Volume"]]

            df1 = df1.rename(columns={"Open": "open", "High": "high", "Low":"low", "Close": "close", "Volume":"volume"})

            tuples = [(ticker,'open'), 
                      (ticker,'high'), 
                      (ticker,'low'), 
                      (ticker,'close'),
                      (ticker,'volume'),
                     ]

            df1.columns = pd.MultiIndex.from_tuples(tuples)
            if i == 0:
                df = df1
            else:
                df = pd.concat([df, df1], axis=1).dropna()
                
    st.pyplot(fig_port)
    
    st.write("Composition of Risk-Adjusted Portfolio based on simulation results")
    st.table(df_allocation.style.format({
         'Min Volatility': '{:.2%}',
         'Max Sharpe Ratio': '{:.2%}',
    }))
    
    st.write("Performance results of these portfolios")
    st.table(df_metrics.style.format({
         'Expected Return': '{:.2%}',
         'Volatility': '{:.2%}',
    }))
    
    weights = df_allocation[type_port].to_list()
      
    
    MC_thirtyyear = MCSimulation(
        portfolio_data = df,
        weights = weights,
        num_simulation = int(simulation_pred),
        num_trading_days = 252*int(forecast_year)
    )

    portfolio_cumulative_returns = MC_thirtyyear.calc_cumulative_return()
    # portfolio_cumulative_returns = pd.read_csv("data/mt_carlo_results.csv")

    principal = invest_value
    portfolio_cumulative_returns_inv = portfolio_cumulative_returns * invest_value

    fig = go.Figure()
    for column in portfolio_cumulative_returns_inv.columns.to_list(): 
        fig.add_trace(
                go.Scatter(
                    x=portfolio_cumulative_returns_inv.index, 
                    y=portfolio_cumulative_returns_inv[column],
    #                 name="Forecast Salary"
                )
            )

    fig.update_layout(
    #     title='Monte Carlo Simulations',
        xaxis_title='Days',
        yaxis_title='Amount(€)',
        showlegend=False
                     )

    st.subheader("**Monte Carlo simulation to forecast next years cumulative returns**")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary statistics and distribution from the MC results")
    st.subheader(" ")
    colForecast1, colForecast2 = st.columns(2)
    with colForecast1:
        dist_plot = MC_thirtyyear.plot_distribution()
        st.pyplot(dist_plot)
    with colForecast2:
        tbl = MC_thirtyyear.summarize_cumulative_return()
    #     tbl = pd.read_csv("data/metrics_mttc.csv")
        st.text(tbl)

    start_date = date.today()
    end_date = start_date + datetime.timedelta(days=int(forecast_year)*252)

    if invest_value == 0:
        projected_returns = portfolio_cumulative_returns.quantile([ 0.025,0.50, 0.975], axis = 1).T
    else:    
        projected_returns = portfolio_cumulative_returns.quantile([ 0.025,0.50, 0.975], axis = 1).T
        projected_returns = projected_returns * principal 

    fig = go.Figure()

    fig.add_trace(
            go.Scatter(
                x=projected_returns.index, 
                y=projected_returns[0.025],
                name="95% lower confidence intervals"
            )
        )
    fig.add_trace(
            go.Scatter(
                x=projected_returns.index, 
                y=projected_returns[0.50],
                name="Mean forecast"
            )
        )
    fig.add_trace(
            go.Scatter(
                x=projected_returns.index, 
                y=projected_returns[0.975],
                name="95% upper confidence intervals"
            )
        )

    fig.update_layout(
    #     title='Monte Carlo Simulations',
        xaxis_title='Days',
        yaxis_title='Amount(€)',
        showlegend=False
                     )
    st.subheader("Monte Carlo results")
    st.plotly_chart(fig, use_container_width=True)
    
    principal = invest_value

    ci_lower = round(tbl[8]*principal,2)
    ci_upper = round(tbl[9]*principal,2)
    mean_value = round(tbl[1]*principal,2)

    st.write(f"Over the next **{int(forecast_year)}** years the mean forecasted value for an initial investment of {principal}€ will be around **{mean_value}€**, an increase on your initial investment of **{np.round(((mean_value - principal)/principal)*100, 2)}%**.")
    st.write(f"There is a **95%** chance that an initial investment of {principal}€ in the portfolio over the next {int(forecast_year)} years will end within in the range of **{ci_lower}€** and **{ci_upper}€**. This means a variation between **{np.round(((ci_lower - principal)/principal)*100, 2)}%** and **{np.round(((ci_upper- principal)/principal)*100, 2)}%**")



#         df_allocation_test = np.format_float_positional(df_allocation_test*100)
#         st.table(df_allocation_test.style.format("{:.2}"))

