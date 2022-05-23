import datetime
import sqlite3
import pandas as pd
import numpy as np
import backtrader as bt
import backtrader.analyzers as btanalyzers


# plugin for accepting fractions for cryptos
class CommInfoFractional(bt.CommissionInfo):
    def getsize(self, price, cash):
        """Returns fractional size for cash operation at price"""
        return self.p.leverage * (cash / price)


# Create a Strategy
class ScalpingStrategy(bt.Strategy):
    params = (
        ('s_ma_period', 60),
        ('f_ma_period', 10),
    )

    def log(self, txt, dt=None):
        """
        Logging method for this strategy
        """
        dt = dt or datetime.datetime.combine(self.datas[0].datetime.date(0), self.datas[0].datetime.time(0))
        print(f"{dt}, {txt}")

    def __init__(self):
        """
        Initialises objects for this strategy
        """
        # To keep track of pending orders and buy price/commission
        self.order = None
        # Add a MovingAverageSimple indicator
        self.s_sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.s_ma_period)
        self.f_sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.f_ma_period)

    def notify_order(self, order):
        """
        Method gets called when an order is placed
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        elif order.status in [order.Completed]:
            # Check if an order has been completed
            # Attention: broker could reject order if not enough cash
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))
            else:  # Order is sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price, order.executed.value, order.executed.comm))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        # Reset after order executed
        self.order = None

    def notify_trade(self, trade):
        """
        Method gets called when an open position is closed
        """
        if not trade.isclosed:
            return
        self.log(f"OPERATION PROFIT OR LOSS, GROSS {trade.pnl}, NET {trade.pnlcomm}")

    def start(self):
        """
        Method gets called once at the start of the backtest
        """
        return

    def next(self):
        """
        Method gets called once for each minute in data feed
        """
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # MIGHT BUY
        if self.get_rnn_result() == self.get_indicators_result() == "Buy":
            buy_quantity = self.get_buy_quantity()
            if buy_quantity > self.get_threshold_quantity():
                self.log('BUY CREATE')
                self.order = self.buy(target=buy_quantity)
        # MIGHT SELL
        elif self.get_rnn_result() == self.get_indicators_result() == "Sell":
            sell_quantity = self.get_sell_quantity()
            if sell_quantity > self.get_threshold_quantity():
                self.log('SELL CREATE')
                self.order = self.sell(target=sell_quantity)

    def get_rnn_result(self):
        """
        Method gets called in next()
        Returns a buy or sell decision based on the result of the RNN output
        """
        return

    def get_indicators_result(self):
        """
        Method gets called in next()
        Returns a buy or sell decision based on the results of technical indicators
        """
        return

    def get_sell_quantity(self):
        """
        Method gets called in next()
        Returns the amount to sell based on the results of technical indicators
        """
        return

    def get_buy_quantity(self):
        """
        Method gets called in next()
        Returns the amount to buy based on the results of technical indicators
        """
        return

    def get_threshold_quantity(self):
        """
        Method gets called in next()
        Returns the threshold amount for placing an order
        """
        return

    def stop(self):
        """
        Method gets called once at the end of the backtest
        """
        return


if __name__ == '__main__':

    RATIO = "BTCUSDT"
    db_connection = sqlite3.connect("Historical_Data.db")
    dataframe = pd.read_sql_query(f"SELECT * FROM {RATIO} WHERE dateTime <= date('2021-07-01')", db_connection)
    dataframe["close"] = pd.to_numeric(dataframe["close"])
    dataframe["volume"] = pd.to_numeric(dataframe["volume"])
    dataframe["dateTime"] = pd.to_datetime(dataframe.datetime, format="%Y-%m-%d %H:%M:%S")
    dataframe.set_index("dateTime", inplace=True)

    # Create a cerebro entity
    cerebro = bt.Cerebro(stdstats=False)

    # Add a strategy
    cerebro.addstrategy(ScalpingStrategy)

    # Pass dataframe to the backtrader datafeed
    data = bt.feeds.PandasData(dataname=dataframe,
                               timeframe=bt.TimeFrame.Minutes,
                               compression=1,
                               fromdate=datetime.datetime.combine(datetime.date(2021, 1, 2), datetime.time(0, 0)),
                               name=RATIO,
                               todate=datetime.date(2021, 6, 29))

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(1000.0)

    # Plugin for cryptocurrencies allowing fractional order sizes
    cerebro.broker.addcommissioninfo(CommInfoFractional())

    # Set the commission
    cerebro.broker.setcommission(commission=0.001)

    # Observers
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.BuySell)

    # Analyzers
    cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, compression=1, factor=365, annualize=True)

    # Run the backtest
    stats = cerebro.run()

    # Print the Sharpe ratio
    print('Sharpe Ratio:', stats[0].analyzers.sharpe.get_analysis())

    # Plot results of the backtest
    cerebro.plot(volume=False)
