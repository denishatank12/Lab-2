# Denisha Tank

from datetime import datetime
from os import path
from tkinter import *
from tkinter import ttk
from tkinter import messagebox, simpledialog, filedialog
import csv
import stock_data
from stock_class import Stock, DailyData
from utilities import clear_screen, display_stock_chart, sortStocks, sortDailyData

class StockApp:
    def __init__(self):
        self.stock_list = []

        if path.exists("stocks.db") == False:
            stock_data.create_database()

   

        
        self.root = Tk()
        self.root.title("Stock Manager")  # Replace with a suitable name for your program

       
        self.menubar = Menu(self.root)

        # Adding File Menu
        self.fileMenu = Menu(self.menubar, tearoff=0)
        self.fileMenu.add_command(label="Load Data", command=self.load)
        self.fileMenu.add_command(label="Save Data", command=self.save)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Exit", command=self.root.destroy)
        self.menubar.add_cascade(label="File", menu=self.fileMenu)

        # Add Web Menu 
        self.webMenu = Menu(self.menubar, tearoff=0)
        self.webMenu.add_command(label="Retrieve Data From Web", command=self.scraping_web_data)
        self.webMenu.add_command(label="Import From CSV File", command=self.importingCSV_web_data)
        self.menubar.add_cascade(label="Web", menu=self.webMenu)

        # Add Chart Menu
        self.chartMenu = Menu(self.menubar, tearoff=0)
        self.chartMenu.add_command(label="Display Chart", command=self.display_chart)
        self.menubar.add_cascade(label="Chart", menu=self.chartMenu)

        # Add menus to window       
        self.root.config(menu=self.menubar)


        self.mainFrame = ttk.Frame(self.root, padding="5")
        self.mainFrame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        leftFrame = ttk.Frame(self.mainFrame, padding="5")
        leftFrame.grid(row=0, column=0, sticky="ns")
      
        rightFrame = ttk.Frame(self.mainFrame, padding="5")
        rightFrame.grid(row=0, column=1, sticky="nsew")
        self.mainFrame.columnconfigure(1, weight=1)
        self.mainFrame.rowconfigure(0, weight=1)

        
        self.headingLabel = ttk.Label(rightFrame, text="Select a stock", font=("TkDefaultFont", 12, "bold"))
        self.headingLabel.grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Add stock list
        ttk.Label(leftFrame, text="Stocks").grid(row=0, column=0, sticky="w")
        self.stockList = Listbox(leftFrame, height=10, exportselection=False)
        self.stockList.grid(row=1, column=0, rowspan=4, sticky="nswe")
        stock_scroll = ttk.Scrollbar(leftFrame, orient=VERTICAL, command=self.stockList.yview)
        stock_scroll.grid(row=1, column=1, rowspan=4, sticky="ns")
        self.stockList.config(yscrollcommand=stock_scroll.set)
        self.stockList.bind("<<ListboxSelect>>", self.update_data)
        leftFrame.rowconfigure(1, weight=1)
        leftFrame.columnconfigure(0, weight=1)

        # Add Tabs
        self.tabControl = ttk.Notebook(rightFrame)
        self.mainTab = ttk.Frame(self.tabControl)
        self.historyTab = ttk.Frame(self.tabControl)
        self.reportTab = ttk.Frame(self.tabControl)
        self.tabControl.add(self.mainTab, text="Main")
        self.tabControl.add(self.historyTab, text="History")
        self.tabControl.add(self.reportTab, text="Report")
        self.tabControl.grid(row=1, column=0, sticky="nsew")
        rightFrame.rowconfigure(1, weight=1)
        rightFrame.columnconfigure(0, weight=1)

        main_inner = ttk.Frame(self.mainTab, padding="5")
        main_inner.pack(fill=BOTH, expand=True)

        addFrame = ttk.LabelFrame(main_inner, text="Add Stock", padding="5")
        addFrame.grid(row=0, column=0, sticky="we", pady=(0, 5))
        ttk.Label(addFrame, text="Symbol:").grid(row=0, column=0, sticky="e")
        self.addSymbolEntry = ttk.Entry(addFrame, width=10)
        self.addSymbolEntry.grid(row=0, column=1, sticky="w")
        ttk.Label(addFrame, text="Name:").grid(row=1, column=0, sticky="e")
        self.addNameEntry = ttk.Entry(addFrame, width=25)
        self.addNameEntry.grid(row=1, column=1, sticky="w")
        ttk.Label(addFrame, text="Shares:").grid(row=2, column=0, sticky="e")
        self.addSharesEntry = ttk.Entry(addFrame, width=10)
        self.addSharesEntry.grid(row=2, column=1, sticky="w")
        ttk.Button(addFrame, text="Add Stock", command=self.adding_stock).grid(row=3, column=0, columnspan=2, pady=(5, 0))

        updateFrame = ttk.LabelFrame(main_inner, text="Update Shares", padding="5")
        updateFrame.grid(row=1, column=0, sticky="we")
        ttk.Label(updateFrame, text="Shares:").grid(row=0, column=0, sticky="e")
        self.updateSharesEntry = ttk.Entry(updateFrame, width=10)
        self.updateSharesEntry.grid(row=0, column=1, sticky="w")
        ttk.Button(updateFrame, text="Buy", command=self.buy_shares).grid(row=1, column=0, pady=(5, 0), sticky="we")
        ttk.Button(updateFrame, text="Sell", command=self.sell_shares).grid(row=1, column=1, pady=(5, 0), sticky="we")
        ttk.Button(updateFrame, text="Delete Stock", command=self.delete_stock).grid(row=2, column=0, columnspan=2, pady=(5, 0), sticky="we")

        self.dailyDataList = Text(self.historyTab, width=60, height=20)
        self.dailyDataList.pack(fill=BOTH, expand=True)

        self.stockReport = Text(self.reportTab, width=60, height=20)
        self.stockReport.pack(fill=BOTH, expand=True)

        self.root.mainloop()

       
    def load(self):
        self.stockList.delete(0,END)
        stock_data.load_stock_data(self.stock_list)
        sortStocks(self.stock_list)
        for stock in self.stock_list:
            self.stockList.insert(END,stock.symbol)
        messagebox.showinfo("Load Data","Data Loaded")

    def save(self):
        stock_data.save_stock_data(self.stock_list)
        messagebox.showinfo("Save Data","Data Saved")

    def update_data(self, evt=None):
        self.display_stock_data()

    def display_stock_data(self):
        if not self.stockList.curselection():
            return
        symbol = self.stockList.get(self.stockList.curselection())
        for stock in self.stock_list:
            if stock.symbol == symbol:
                self.headingLabel['text'] = stock.name + " - " + str(stock.shares) + " Shares"
                self.dailyDataList.delete("1.0",END)
                self.stockReport.delete("1.0",END)
                self.dailyDataList.insert(END,"- Date -   - Price -   - Volume -\n")
                self.dailyDataList.insert(END,"=================================\n")

                sortDailyData(self.stock_list)

                for daily_data in stock.DataList:
                    row = daily_data.date.strftime("%m/%d/%y") + "   " + '${:0,.2f}'.format(daily_data.close) + "   " + str(int(daily_data.volume)) + "\n"
                    self.dailyDataList.insert(END,row)

                if stock.DataList:
                    closes = [d.close for d in stock.DataList]
                    latest_price = closes[-1]
                    total_value = latest_price * stock.shares
                    high = max(closes)
                    low = min(closes)
                    avg = sum(closes) / len(closes)

                    self.stockReport.insert(END,f"Symbol: {stock.symbol}\n")
                    self.stockReport.insert(END,f"Name: {stock.name}\n")
                    self.stockReport.insert(END,f"Shares: {stock.shares}\n")
                    self.stockReport.insert(END,f"Current Price: ${latest_price:0,.2f}\n")
                    self.stockReport.insert(END,f"Position Value: ${total_value:0,.2f}\n")
                    self.stockReport.insert(END,f"High Close: ${high:0,.2f}\n")
                    self.stockReport.insert(END,f"Low Close: ${low:0,.2f}\n")
                    self.stockReport.insert(END,f"Average Close: ${avg:0,.2f}\n")
                break

    def adding_stock(self):
        try:
            new_stock = Stock(self.addSymbolEntry.get().upper(),self.addNameEntry.get(),float(str(self.addSharesEntry.get())))
        except ValueError:
            messagebox.showerror("Input Error","Shares must be a number")
            return
        self.stock_list.append(new_stock)
        sortStocks(self.stock_list)
        self.stockList.delete(0,END)
        for stock in self.stock_list:
            self.stockList.insert(END,stock.symbol)
        self.addSymbolEntry.delete(0,END)
        self.addNameEntry.delete(0,END)
        self.addSharesEntry.delete(0,END)

    def buy_shares(self):
        if not self.stockList.curselection():
            messagebox.showwarning("No Selection","Select a stock first.")
            return
        symbol = self.stockList.get(self.stockList.curselection())
        for stock in self.stock_list:
            if stock.symbol == symbol:
                try:
                    shares = float(self.updateSharesEntry.get())
                except ValueError:
                    messagebox.showerror("Input Error","Shares must be a number")
                    return
                stock.buy(shares)
                self.headingLabel['text'] = stock.name + " - " + str(stock.shares) + " Shares"
        messagebox.showinfo("Buy Shares","Shares Purchased")
        self.updateSharesEntry.delete(0,END)
        self.display_stock_data()

    def sell_shares(self):
        if not self.stockList.curselection():
            messagebox.showwarning("No Selection","Select a stock first.")
            return
        symbol = self.stockList.get(self.stockList.curselection())
        for stock in self.stock_list:
            if stock.symbol == symbol:
                try:
                    shares = float(self.updateSharesEntry.get())
                except ValueError:
                    messagebox.showerror("Input Error","Shares must be a number")
                    return
                if shares > stock.shares:
                    messagebox.showerror("Sell Shares","Not enough shares to sell.")
                    return
                stock.sell(shares)
                self.headingLabel['text'] = stock.name + " - " + str(stock.shares) + " Shares"
        messagebox.showinfo("Sell Shares","Shares Sold")
        self.updateSharesEntry.delete(0,END)
        self.display_stock_data()

    def delete_stock(self):
        if not self.stockList.curselection():
            messagebox.showwarning("No Selection","Select a stock first.")
            return
        symbol = self.stockList.get(self.stockList.curselection())
        self.stock_list = [s for s in self.stock_list if s.symbol != symbol]
        self.stockList.delete(0,END)
        sortStocks(self.stock_list)
        for stock in self.stock_list:
            self.stockList.insert(END,stock.symbol)
        self.headingLabel['text'] = "Select a stock"
        self.dailyDataList.delete("1.0",END)
        self.stockReport.delete("1.0",END)
        messagebox.showinfo("Delete Stock",symbol + " Deleted")

  
    def scraping_web_data(self):
        dateFrom = simpledialog.askstring("Starting Date","Enter Starting Date (m/d/yy)")
        dateTo = simpledialog.askstring("Ending Date","Enter Ending Date (m/d/yy)")
        if not dateFrom or not dateTo:
            return
        try:
            records = stock_data.retrieve_stock_web(dateFrom, dateTo, self.stock_list)
        except RuntimeWarning:
            messagebox.showerror("Cannot Get Data from Web","Check Path for Chrome Driver")
            return
        self.display_stock_data()
        messagebox.showinfo("Get Data From Web", f"{records} records retrieved")

        
    def importingCSV_web_data(self):
        if not self.stockList.curselection():
            messagebox.showwarning("No Selection","Select a stock first.")
            return
        symbol = self.stockList.get(self.stockList.curselection())
        filename = filedialog.askopenfilename(
            title="Select " + symbol + " File to Import",
            filetypes=[('Yahoo Finance! CSV','*.csv')]
        )
        if filename != "":
            stock_data.import_stock_web_csv(self.stock_list,symbol,filename)
            self.display_stock_data()
            messagebox.showinfo("Import Complete",symbol + " Import Complete")   
    
    def display_chart(self):
        if not self.stockList.curselection():
            messagebox.showwarning("No Selection","Select a stock first.")
            return
        symbol = self.stockList.get(self.stockList.curselection())
        display_stock_chart(self.stock_list,symbol)


def main():
        app = StockApp()
        

if __name__ == "__main__":

    main()

