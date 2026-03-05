import yfinance as yf
import pandas as pd

print("Downloading Gold Price Data...")

gold = yf.download("GC=F", period="10y", interval="1d")

gold.reset_index(inplace=True)

gold.to_csv("data/gold_prices.csv", index=False)

print("Gold data saved successfully!")
print(gold.tail())