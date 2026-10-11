'''
custom tool : get share price of a stock from yahoo finance

'''

import yfinance as yf
from crewai.tools import tool


# @ : decorator to define a custom tool
@tool("Get share price")
def get_share_price(symbol:str) -> str:
    '''
    gets the closing price and 1 month price change of a share symbol
    yfinance code : INFY --> INFY.NS
    '''
    prices = yf.Ticker(symbol).history(period="1mo")["Close"]
    if prices.empty:
        return f"No price data found for {symbol}"
    latest = prices.iloc[-1]
    change = (latest-prices.iloc[0])/prices.iloc[0]*100
    return f" {symbol} : latest closing price : {latest:.2f} , 1 month change : {change:.2f}%"
        
        
# run this file to test the tool
if __name__ == "__main__":
    print(get_share_price.run(symbol= "INFY.NS"))