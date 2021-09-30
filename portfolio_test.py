import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf
from finquant.portfolio import build_portfolio
import warnings

import datetime 
from datetime import date
from dateutil.relativedelta import relativedelta
import streamlit as st


warnings.filterwarnings("ignore")
sns.set(rc={'figure.figsize':(16,7)})

input_dir = "data"

NUM_YEARS = 40

end_date = date.today()
start_date = end_date-datetime.timedelta(days=NUM_YEARS*365)


def allocation_data(input_dir):
    
    df_portfolio = pd.read_csv(f"{input_dir}/Portfolio_test.csv") 
    df_portfolio["Value Stock"] = None
    df_portfolio["Name"] = df_portfolio["Ticker"]

    tickers_list = df_portfolio.Ticker.to_list()
    for num, i in enumerate(tickers_list):
        df_portfolio["Value Stock"].loc[num] = yf.Ticker(i).history(period="max")["Close"][-1:].values[0]

    df_portfolio["Value"] = df_portfolio["Nº Stock"] * df_portfolio["Value Stock"]
    df_portfolio["Allocation"] = (df_portfolio["Value"]/df_portfolio["Value"].sum())*100
        
    return df_portfolio


def allocation_by_sectors(df_portfolio):
    
    df_portfolio_stocks = df_portfolio[df_portfolio["Asset"] == "Stock"].reset_index(drop=True)
    if df_portfolio_stocks["Allocation"].any() == 0:
        df_portfolio_stocks["Allocation"] = 1
    df_portfolio_etfs = df_portfolio[df_portfolio["Asset"] == "ETF_Stock"].reset_index(drop=True)
    if df_portfolio_etfs["Allocation"].any() == 0:
        df_portfolio_etfs["Allocation"] = 1
    df_portfolio_crypto = df_portfolio[df_portfolio["Asset"] == "Crypto"].reset_index(drop=True)
    if df_portfolio_crypto["Allocation"].any() == 0:
        df_portfolio_crypto["Allocation"] = 1
    df_portfolio_reit = df_portfolio[df_portfolio["Asset"] == "ETF_REIT"].reset_index(drop=True)
    if df_portfolio_reit["Allocation"].any() == 0:
        df_portfolio_reit["Allocation"] = 1
    df_portfolio_bond = df_portfolio[df_portfolio["Asset"] == "ETF_Bond"].reset_index(drop=True)
    if df_portfolio_bond["Allocation"].any() == 0:
        df_portfolio_bond["Allocation"] = 1
    df_portfolio_gold = df_portfolio[df_portfolio["Asset"] == "ETF_Gold"].reset_index(drop=True)
    if df_portfolio_gold["Allocation"].any() == 0:
        df_portfolio_gold["Allocation"] = 1
    
    return df_portfolio_stocks, df_portfolio_etfs, df_portfolio_crypto, df_portfolio_reit, df_portfolio_bond, df_portfolio_gold 


def allocation_process(df_portfolio):
    allocation = {}
    d = {}

    symbols = df_portfolio.Ticker.to_list()
    names = df_portfolio.Ticker.to_list()

    for num, i in enumerate(symbols):
        allocation[i] = df_portfolio["Allocation"].loc[num]

    for num, values in enumerate(df_portfolio[["Name", "Allocation"]].to_dict(orient='records')):
        d[num] = values
    return symbols, names, allocation, d


def build_my_portfolio(d, names, start_date, end_date, num_sim=5000, weight=1):#, allocation=allocation): 
    
    pf_allocation = pd.DataFrame.from_dict(d, orient="index")

    pf = build_portfolio(
        names=names, 
        pf_allocation=pf_allocation, 
        start_date=start_date, 
        end_date=end_date,
        data_api="yfinance")

    opt_w, opt_res = pf.mc_optimisation(num_trials=num_sim)

    fig_6, ax2 = plt.subplots()
#     ax2 = pf.mc_plot_results()
    ax2 = pf.ef_plot_efrontier()
#     ax2 = pf.ef.plot_optimal_portfolios()
    ax2 = pf.plot_stocks()
    
    df_allocation = opt_w.T*weight
    df_allocation = df_allocation.reset_index()
    df_allocation = df_allocation.rename(columns={'index': 'Ticker'})
#     df_allocation["Our Portfolio"] = (df_allocation['Ticker'].map(allocation))/100
    df_allocation = df_allocation.set_index("Ticker")

    return fig_6, df_allocation

# @st.cache
def output_portfolio(input_dir):
    
    df_portfolio = allocation_data(input_dir)
    
    symbols, names, allocation, d = allocation_process(df_portfolio)
    fig_all_total, df_allocation = build_my_portfolio(d, names, start_date, end_date, num_sim=10000, weight=1)#, allocation=allocation)

    return fig_all_total, df_allocation