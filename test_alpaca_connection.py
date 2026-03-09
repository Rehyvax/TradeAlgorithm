import os
from dotenv import load_dotenv
load_dotenv()
from alpaca.trading.client import TradingClient

trading_client = TradingClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_API_SECRET"],
    paper=True
)

account = trading_client.get_account()

print("Account status:", account.status)
print("Equity:", account.equity)
print("Buying power:", account.buying_power)
print("Trading blocked:", account.trading_blocked)

balance_change = float(account.equity) - float(account.last_equity)
print("Today's PnL:", balance_change)
