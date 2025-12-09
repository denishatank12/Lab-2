#Denisha Tank
import matplotlib.pyplot as plt

from os import system, name

def clearing_screen():
    if name == "nt": 
        _ = system('cls')
    else: 
        _ = system('clear')

def sortingStocks(stock_list):
    stock_list.sort(key=lambda stock: stock.symbol.upper())

def sortingofDailyData(stock_list):
    for stock in stock_list:
        stock.DataList.sort(key=lambda daily: daily.date)

def displaying_stock_chart(stock_list, symbol):
    target = next((s for s in stock_list if s.symbol.upper() == symbol.upper()), None)
    if target is None or not target.DataList:
        print("No data available to chart for symbol:", symbol)
        return

    target.DataList.sort(key=lambda d: d.date)

    dates = [d.date for d in target.DataList]
    closes = [d.close for d in target.DataList]

    plt.figure()
    plt.plot(dates, closes)
    plt.title(f"{target.symbol} Closing Price History")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
