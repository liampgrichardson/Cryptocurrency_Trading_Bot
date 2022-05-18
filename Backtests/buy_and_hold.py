import datetime
import sqlite3
import pandas as pd
import numpy as np
import backtrader as bt


# plugin for accepting fractions for cryptos
class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        """Returns fractional size for cash operation @price"""
        return self.p.leverage * (cash / price)


class BuyAndHold(bt.Strategy):

    def __init__(self):
        self.order = None
        self.val_start = None
        self.roi = None
        self.sell_create_datetime = None

    def log(self, txt, dt=None):
        """Logging function fot this strategy"""
        dt = dt or datetime.datetime.combine(self.datas[0].datetime.date(0), self.datas[0].datetime.time(0))
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):

        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def start(self):
        # save the starting cash value
        self.val_start = self.broker.get_value()
        # the size of buy is based on the first open price
        size = (self.broker.get_value() / self.data.open[1])*(1-commission)
        self.order = self.order_target_size(target=size)
        # the sell order will be created on the second last bar and executed on the last bar
        self.sell_create_datetime = f"{datetime.datetime.combine(self.datas[0].datetime.date(-1), self.datas[0].datetime.time(-1))}"

    def next(self):
        dtime = f"{self.datas[0].datetime.date(0).strftime('%Y-%m-%d')} {self.datas[0].datetime.time(0).strftime('%H:%M:%S')}"
        if dtime == self.sell_create_datetime:
            self.order = self.close()

    def stop(self):
        # calculate the actual returns
        self.roi = (cerebro.broker.get_value() / self.val_start) - 1.0
        print(f"ROI: {round(100.0 * self.roi, 2)}")


if __name__ == '__main__':

    RATIO = RATIO_TO_PREDICT = "BTCUSDT"
    db_connection = sqlite3.connect(
        "../../Binance_Zips_To_SQLite/SQLite_DBs/HD_from_2021-01_to_2022-03_interval_1min_created_2022-04-19_11-50-56.db")

    dataframe = pd.read_sql_query(f"SELECT * FROM {RATIO} WHERE dateTime <= date('2021-07-01')", db_connection)
    dataframe["close"] = pd.to_numeric(dataframe["close"])
    dataframe["open"] = pd.to_numeric(dataframe["open"])
    dataframe["volume"] = pd.to_numeric(dataframe["volume"])
    dataframe.rename(columns={"dateTime": "datetime"}, inplace=True)
    dataframe["datetime"] = pd.to_datetime(dataframe.datetime, format="%Y-%m-%d %H:%M:%S")
    dataframe.set_index("datetime", inplace=True)  # set time as index, so we can join them on this shared time

    print('----------------DATAFRAME---------------------')
    print(dataframe)
    print('----------------------------------------------')

    # Create a cerebro entity
    cerebro = bt.Cerebro(cheat_on_open=True)

    # for crypto fractions
    cerebro.broker.addcommissioninfo(CommInfoFractional())

    # Add a strategy
    cerebro.addstrategy(BuyAndHold)

    # Pass dataframe to the backtrader datafeed and add it to the cerebro
    data = bt.feeds.PandasData(dataname=dataframe,
                               timeframe=bt.TimeFrame.Minutes,
                               compression=1,
                               fromdate=datetime.datetime.combine(datetime.date(2021, 6, 1), datetime.time(0, 0)),
                               name="BTCUSDT",
                               todate=datetime.date(2021, 6, 29))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set the commission
    commission = 0.001
    cerebro.broker.setcommission(commission=commission)

    # Set the starting cash
    cerebro.broker.setcash(1000)  # self.data.open[1]

    # Run over everything
    cerebro.run()

    # Plot the result
    cerebro.plot(volume=False)
