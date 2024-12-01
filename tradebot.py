import os
import json
import ccxt
import pandas as pd
import numpy as np
from sqlalchemy import func
import asyncio
from web3 import Web3
from web3.exceptions import ContractLogicError
from sqlalchemy import inspect
import subprocess
import time

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

# Create an SQLite database connection (it creates a file 'trades.db')
engine = create_engine('sqlite:///trades.db', echo=False)
Base = declarative_base()


class Trade(Base):
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)               # Unique trade ID
    signal = Column(String)                              # Signal (BUY or SELL)
    price = Column(Float)                                # Price at which the trade was executed
    tp = Column(Float)                                   # Take profit price
    sl = Column(Float)                                   # Stop loss price
    trade_id = Column(Integer)                           # Trade ID from the system
    equity = Column(Float, nullable=False, default=0.0)  # Equity for the trade
    is_open = Column(Boolean, default=True)              # Whether the trade is still open

    def __repr__(self):
        return f"<Trade(id={self.id}, signal='{self.signal}', price={self.price}, is_open={self.is_open})>"

# Create the table if it doesn't exist
Base.metadata.create_all(engine)

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# session.query(Trade).delete()
# session.commit()  # Make sure to commit the changes

# Initialize exchange connection
exchange = ccxt.binance({
    'enableRateLimit': True,
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET')
})

# Define the trading pair and indicators
pair = 'ETH/USDT'
timeframe = '15m'
periods = 100

# Contract parameters
TAKE_PROFIT = 1.05  # e.g., 5% profit
STOP_LOSS = 0.95    # e.g., 5% loss
trade_id = None
entry_price = None

# Technical analysis functions
def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

async def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(window=period, min_periods=1).mean()
    avg_loss = losses.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

async def calculate_cci(df, period=10):
    TP = (df['high'] + df['low'] + df['close']) / 3
    sma = TP.rolling(window=period).mean()
    mean_deviation = TP.rolling(window=period).apply(lambda x: np.fabs(x - x.mean()).mean())
    cci = (TP - sma) / (0.015 * mean_deviation)
    return cci

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# Function to fetch data and calculate signal
async def generate_signal():
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(pair, timeframe, limit=periods)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Calculate indicators
    df['EMA25'] = calculate_ema(df['close'], 25)
    df['RSI4'] = await calculate_rsi(df['close'], 4)
    df['CCI10'] = await calculate_cci(df, 10)
    df['ATR14'] = calculate_atr(df, 14)

    latest = df.iloc[-1]
    signal = None

    # Generate signal based on indicators
    if latest['RSI4'] < 30 and latest['CCI10'] < -100 and latest['close'] < latest['EMA25']:
        signal = 'SELL'
    elif latest['RSI4'] > 70 and latest['CCI10'] > 100 and latest['close'] > latest['EMA25']:
        signal = 'BUY'

    # Return signal and ATR
    return signal, latest['ATR14']

def get_current_price():
    # Replace `pair` with your trading pair (e.g., 'BTC/USDT') and `exchange` with your exchange object
    ticker =  exchange.fetch_ticker(pair)
    return ticker['last']  # Return the last traded price

uniswap_router = Localweb3.eth.contract(address=UNISWAP_V2_ROUTER, abi=UNISWAP_V2_ABI)


# Main routine to periodically check for signals and execute trades
async def main():
    global trade_id, entry_price

    while True:
        # Generate a signal and get the relevant ATR
        signal, atr = await generate_signal()

        # Fetch the current price
        price = get_current_price()

        print(f"Signal generated: {signal} at price {price} with ATR {atr}")

        # Check if there are existing open trades
        open_trades = session.query(Trade).filter_by(is_open=True).all()

        if len(open_trades) < 2:
            if signal is not None:
                # Calculate TP and SL based on the ATR
                if signal == 'BUY':
                    tp = price + (2 * atr)  # Take profit is 2x ATR above the entry price
                    sl = price - (1 * atr)  # Stop loss is 1x ATR below the entry price
                elif signal == 'SELL':
                    tp = price - (2 * atr)  # Take profit is 2x ATR below the entry price
                    sl = price + (1 * atr)  # Stop loss is 1x ATR above the entry price

                # Place Buy and Sell trades
                buy_trade = Trade(
                    signal='BUY',
                    price=price,
                    tp=price + (2 * atr),  # TP for BUY trade
                    sl=price - (atr),  # SL for BUY trade
                    is_open=True
                )

                session.add(buy_trade)
                print(f"Buy Trade Stored: Entry Price={price}, TP={buy_trade.tp}, SL={buy_trade.sl}, ATR={atr}")

                sell_trade = Trade(
                    signal='SELL',
                    price=price,
                    tp=price - (2 * atr),  # TP for SELL trade
                    sl=price + (atr),  # SL for SELL trade
                    is_open=True
                )

                session.add(sell_trade)
                print(f"Sell Trade Stored: Entry Price={price}, TP={sell_trade.tp}, SL={sell_trade.sl}, ATR={atr}")

                session.commit()

                print(f"Placed Buy and Sell trades: {buy_trade}, {sell_trade}")

        else:
            print("Two trades are already active. Monitoring them.")

        # Monitor trades for closure
        for trade in open_trades:
            current_price = get_current_price()

            if trade.signal == 'BUY':
                if current_price >= trade.tp:
                    print(f"Take Profit Hit for Buy trade at {current_price}")
                    trade.equity = current_price - trade.price
                    trade.is_open = False
                elif current_price <= trade.sl:
                    print(f"Stop Loss Hit for Buy trade at {current_price}")
                    trade.equity = current_price - trade.price
                    trade.is_open = False
            elif trade.signal == 'SELL':
                if current_price <= trade.tp:
                    print(f"Take Profit Hit for Sell trade at {current_price}")
                    trade.equity = trade.price - current_price
                    trade.is_open = False
                elif current_price >= trade.sl:
                    print(f"Stop Loss Hit for Sell trade at {current_price}")
                    trade.equity = trade.price - current_price
                    trade.is_open = False

            if not trade.is_open:
                session.add(trade)
                session.commit()
                print(f"Closed trade: {trade}")

        # Calculate total equity
        total_equity = session.query(func.sum(Trade.equity)).scalar() or 0
        print(f"Total Equity: {total_equity}")

        await asyncio.sleep(150)
        
asyncio.run(main())
