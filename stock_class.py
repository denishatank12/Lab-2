# Denisha TANK


from datetime import datetime

class Stock:
    def __init__(self, symbol, name, shares):
        self._symbol = symbol
        self._name = name
        self._shares = shares
        self.DataList = [] 

    @property
    def symbol(self):
        return self._symbol
    @symbol.setter
    def symbol(self, symbol):
        raise RuntimeWarning("Cannot Change Stock Symbol")
    
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self,name):
        self._name = name
    
    @property
    def shares_(self):
        return self._shares
    @shares_.setter
    def shares_(self,shares):
        raise RuntimeWarning("Use buy() or sell() to change shares.")

    def buy(self, shares):
        self._shares = self._shares + shares

    def sell(self, shares):
       self._shares = self._shares - shares
       
    # Add daily stock data
    def add_data(self, stock_data):
        self.DataList.append(stock_data)
    

class DailyData:
    def __init__(self, date, close, volume):
        self._date = date
        self._close = close
        self._volume = volume

    @property
    def date(self):
        return self._date
    @date.setter
    def date(self, date):
        self._date = date

    @property
    def close(self):
        return self._close
    @close.setter
    def close(self, close):
        self._close = close
    
    @property
    def volume(self):
        return self._volume
    @volume.setter
    def volume(self, volume):
        self._volume = volume



def main():
    error_count = 0
    error_list = []
    print("Unit Testing Starting---")
    # Test Adding Stock
    print("Testing Add Stock...",end="")
    try:
        testStock = Stock("TEST","Test Company",100)
        print("Successful!")
    except:
        print("***Adding Stock Failed!")
        error_count = error_count+1
        error_list.append("Stock Constructor Error")
  
    print("Testing Change Symbol...",end="") 
    try:
        testStock.symbol = "NEWTEST"
        print("***ERROR! Changing stock symbol should not be allowed.")
        error_count = error_count+1
        error_list.append("Stock symbol change allowed. Stock symbol changes should not be allowed.")
    except:
        print("Successful! - Stock symbol change blocked")
    # Test Changing Name
    print("Test Change Name...",end="")
    try:
        testStock.name = "New Test Company"
        if testStock.name == "New Test Company":
            print("Successful!")
        else:
            print("***ERROR! Name change unsuccessful.")
            error_count = error_count+1
            error_list.append("Name Change Error")
    except:
        print("***ERROR! Name change failed.")
        error_count = error_count+1
        error_list.append("Name Change Failure")

    print("Test Change Shares...",end="")
    try:
        testStock.shares_ = 200
        print("***ERROR! Changing stock shares directly should not be allowed.")
        error_count = error_count+1
        error_list.append("Stock shares change allowed. Change in shares should be done through buy() or sell().")
    except:
        print("Successful! - Stock shares change blocked")

    print("Test Buy shares...",end="")
    try:
        testStock.buy(50)
        if testStock.shares_ == 150:
            print("Successful!")
        else:
            print("ERROR! Buy shares unsuccessful.")
            error_count = error_count + 1
            error_list.append("Buy Shares Failure!")
    except:
        print("ERROR! Buy shares failed.")
        error_count = error_count + 1
        error_list.append("Buy Shares Failure!")
    print("Test Sell shares...",end="")
    try:
        testStock.sell(25)
        if testStock.shares_ == 125:
            print("Successful!")
        else:
            print("ERROR! Sell shares unsuccessful.")
            error_count = error_count+1
            error_list.append("Sell Shares Failure!")
    except:
        print("ERROR! Sell shares failed.")
        error_count = error_count + 1
        error_list.append("Sell Shares Failure!")

    # Test add daily data
    print("Creating daily stock data...",end="")
    daily_data_error = False
    try:
        dayData = DailyData(datetime.strptime("1/1/20","%m/%d/%y"),float(14.50),float(100000))
        testStock.add_data(dayData)
        if testStock.DataList[0].date != datetime.strptime("1/1/20","%m/%d/%y"):
            error_count = error_count + 1
            daily_data_error = True
            error_list.append("Add Daily Data - Problem with Date")
        if testStock.DataList[0].close != 14.50:
            error_count = error_count + 1
            daily_data_error = True
            error_list.append("Add Daily Data - Problem with Closing Price")
        if testStock.DataList[0].volume != 100000:
            error_count = error_count + 1
            daily_data_error = True
            error_list.append("Add Daily Data - Problem with Volume")  
    except:
        print("ERROR! Add daily data failed.")
        error_count = error_count + 1
        error_list.append("Add daily data Failure!")
        daily_data_error = True
    if daily_data_error == True:
        print("ERROR! Creating daily data failed.")
    else:
        print("Successful!")
    
    if (error_count) == 0:
        print("Congratulations - All Tests Passed")
    else:
        print("-=== Problem List - Please Fix ===-")
        for em in error_list:
            print(em)
    print("Goodbye")


if __name__ == "__main__":

    main()