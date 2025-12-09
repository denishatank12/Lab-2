# Denisha Tank



import sqlite3
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import csv
import time
from datetime import datetime
from utilities import clear_screen
from utilities import sortDailyData
from stock_class import Stock, DailyData

# Creating the SQLite database
def create_database():
    stockDB = "stocks.db"
    conn = sqlite3.connect(stockDB)
    cur = conn.cursor()
    createStockTableCmd = """CREATE TABLE IF NOT EXISTS stocks (
                            symbol TEXT NOT NULL PRIMARY KEY,
                            name TEXT,
                            shares REAL
                        );"""
    createDailyDataTableCmd = """CREATE TABLE IF NOT EXISTS dailyData (
                                symbol TEXT NOT NULL,
                                date TEXT NOT NULL,
                                price REAL NOT NULL,
                                volume REAL NOT NULL,
                                PRIMARY KEY (symbol, date)
                        );"""   
    cur.execute(createStockTableCmd)
    cur.execute(createDailyDataTableCmd)

def saving_stock_data(stock_list):
    stockDB = "stocks.db"
    conn = sqlite3.connect(stockDB)
    cur = conn.cursor()
    insertStockCmd = """INSERT INTO stocks
                            (symbol, name, shares)
                            VALUES
                            (?, ?, ?); """
    insertDailyDataCmd = """INSERT INTO dailyData
                                    (symbol, date, price, volume)
                                    VALUES
                                    (?, ?, ?, ?);"""
    for stock in stock_list:
        insertValues = (stock.symbol, stock.name, stock.shares)
        try:
            cur.execute(insertStockCmd, insertValues)
            cur.execute("COMMIT;")
        except:
            pass
        for daily_data in stock.DataList: 
            insertValues = (stock.symbol,daily_data.date.strftime("%m/%d/%y"),daily_data.close,daily_data.volume)
            try:
                cur.execute(insertDailyDataCmd, insertValues)
                cur.execute("COMMIT;")
            except:
                pass
    
def loading_stock_data(stock_list):
    stock_list.clear()
    stockDB = "stocks.db"
    conn = sqlite3.connect(stockDB)
    stockCur = conn.cursor()
    stockSelectCmd = """SELECT symbol, name, shares
                    FROM stocks; """
    stockCur.execute(stockSelectCmd)
    stockRows = stockCur.fetchall()
    for row in stockRows:
        new_stock = Stock(row[0],row[1],row[2])
        dailyDataCur = conn.cursor()
        dailyDataCmd = """SELECT date, price, volume
                        FROM dailyData
                        WHERE symbol=?; """
        selectValue = (new_stock.symbol)
        dailyDataCur.execute(dailyDataCmd,(selectValue,))
        dailyDataRows = dailyDataCur.fetchall()
        for dailyRow in dailyDataRows:
            daily_data = DailyData(datetime.strptime(dailyRow[0],"%m/%d/%y"),float(dailyRow[1]),float(dailyRow[2]))
            new_stock.add_data(daily_data)
        stock_list.append(new_stock)
    sortDailyData(stock_list)



def retrieving_stock_web(dateStart, dateEnd, stock_list):
    dateFrom = str(int(time.mktime(time.strptime(dateStart, "%m/%d/%y"))))
    dateTo = str(int(time.mktime(time.strptime(dateEnd, "%m/%d/%y"))))
    recordCount = 0

    for stock in stock_list:
        stockSymbol = stock.symbol
        url = (
            "https://finance.yahoo.com/quote/"
            + stockSymbol
            + "/history?period1=" + dateFrom
            + "&period2=" + dateTo
            + "&interval=1d&filter=history&frequency=1d"
        )

        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        try:
            driver = webdriver.Chrome(options=options)
            driver.implicitly_wait(60)
            driver.get(url)
        except Exception:
            raise RuntimeWarning("Chrome Driver Not Found")

        soup = BeautifulSoup(driver.page_source, "html.parser")
        driver.quit()

        # Yahoo's historical pricingg table
        table = soup.find("table", attrs={"data-test": "historical-prices"})
        if not table:
            table = soup.find("table")
        if not table:
            continue

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            rowList = [c.text for c in cols]

            if len(rowList) == 7:
                try:
                    date = datetime.strptime(rowList[0], "%b %d, %Y")
                    close = float(rowList[5].replace(",", ""))
                    volume = float(rowList[6].replace(",", ""))
                except ValueError:
                    continue

                daily_data = DailyData(date, close, volume)
                stock.add_data(daily_data)
                recordCount += 1

    return recordCount



def importing_stock_web_csv(stock_list, symbol, filename):
    for stock in stock_list:
        if stock.symbol == symbol:
            with open(filename, newline='') as stockdata:
                datareader = csv.reader(stockdata, delimiter=',')
                next(datareader, None)

                for row in datareader:
                    if not row or len(row) < 3:
                        continue

                    date_str = row[0].strip()
                    close_str = row[1].strip()
                    volume_str = row[2].strip()

                    try:
                        date = datetime.strptime(date_str, "%m/%d/%Y")
                        close = float(close_str.replace("$", "").replace(",", ""))
                        volume = float(volume_str.replace(",", ""))
                    except ValueError:
                        continue

                    daily_data = DailyData(date, close, volume)
                    stock.add_data(daily_data)

            break


def main():
    clear_screen()
    print("This module will handle data storage and retrieval.")

if __name__ == "__main__":

    main()