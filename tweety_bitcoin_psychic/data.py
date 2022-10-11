"""Loading and processing the bitcoin price data.

This package returns a Datafame with following columns:
  index:
    Datetime, daily based in format: "%Y-%m-%d"
  open: float64
    Opening bitcoin price in USD
  high: float64
    Highest price in USD
  low: float64
    Lowest price in USD
  close: float64
    Closing bitcoin price in USD
  open_pchg: float64
    Percentage change in closing price within the last day
  high_pchg: float64
    Percentage change of the maximum price within the last day
  low_pchg: float64
    Percentage change of the lowest price within the last day
  close_pchg: float64
    Percentage change in closing price within the last day
  volume_pchg: float64
    Percentage change in traded volume within the last day
  open_close_pdiff: float64
    Percentage difference between opening and closing price
  high_close_pdiff: float64
    Percentage difference between highest and closing price
  low_close_pdiff: float64
    Percentage difference between lowest and closing price
  pfc: int64
    Performance variable based on 'open_close_pdiff'
  ma1: float64
    1 day moving average based on closing price
  ma7: float64
    7 day moving average based on closing price
  ma14: float64
    14 day moving average based on closing price
  day_of_week: int64
    Day of week as integer, i.e. 0 for Monday, 1 for Tuesday
  gtrend: float64
    Google Trend index value of the keyword: bitcoin
"""

from numpy import nan
from pytrends.request import TrendReq
from yfinance import Ticker


def _request_yfinance(symbol, period="max", interval="1d"):
    """Get historical market data from Yahoo! Finance.

    This function depends on the yfinance package and
    loads data of the last ~2350 days on an daily basis.
    Yahoo! Finance sources crypto data from CoinMarketCap.
      PyPi: https://pypi.org/project/yfinance/
      GitHub: https://github.com/ranaroussi/yfinance

    Args:
      symbol: string
        Valid Yahoo! Finance symbol, i.e. ^GDAXI, BTC-USD
      period : string
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
      interval : string
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days

    Returns:
      pandas.core.frame.DataFrame
    """
    ticker = Ticker(symbol)
    return ticker.history(period=period, interval=interval)


def _wrangle_yfinance(df):
    """Wrangle the historical market data from Yahoo! Finance.

    Args:
      df: pandas.core.frame.DataFrame
        DataFrame from request_yfinance()

    Returns:
      pandas.core.frame.DataFrame
    """
    # Lower column names, reset index
    # seperate datetime and drop unnecessary columns
    df.columns = df.columns.str.lower()
    df = df.reset_index()
    df = df.drop(["dividends", "stock splits"], axis=1)
    dates = df.pop("Date")  # DF.pop("index")

    # Handle volume outliers with Interquartile Range (IQR)
    # and interpolate empty volume values
    q1 = df["volume"].quantile(0.25)
    q3 = df["volume"].quantile(0.75)
    iqr = q3 - q1
    df.loc[df["volume"] < (q1 - 1.5 * iqr), "volume"] = nan
    df.loc[df["volume"] > (q3 + 1.5 * iqr), "volume"] = nan
    df["volume"].replace(0, nan, inplace=True)
    df["volume"] = df["volume"].interpolate(method="linear", limit_direction="both")

    # Round and calculate the percentage changes over all columns
    for column in df:
        df[column] = [round(i, 1) for i in df[column]]
        df[f"{column}_pchg"] = [
            0
            if idx == df.index[0]
            else round((val[column] * 100) / df[column][idx - 1], 3) - 100
            for idx, val in df.iterrows()
        ]

    # Calculate percentage differences and create a performance variable
    # based on the difference between opening and closing price:
    # “1” if the performance was positive (open_diff > 0) and “0” if else
    for column in ["open", "high", "low"]:
        df[f"{column}_close_pdiff"] = [
            round((val["close"] * 100) / val[column], 3) - 100
            for idx, val in df.iterrows()
        ]
    df["pfc"] = [1.0 if diff > 0 else 0.0 for diff in df["open_close_pdiff"]]

    # Calculate several moving averages
    for ma in [1, 7, 14]:
        df[f"ma{ma}"] = round(df["close"].rolling(24 * ma).mean(), 1)

    #  Create several time based columns
    df["day_of_week"] = [int(i.dayofweek) for i in dates]
    df["date"] = [i.strftime("%Y-%m-%d") for i in dates]

    # Drop volume and reset the original index
    df = df.drop(["volume"], axis=1)
    df.index = dates
    return df


def _request_gtrends(keywords, start_date, end_date):
    """Get historical data from Google Trends.

    This function depends on the pytrends package
    which is an unofficial API for Google Trends.
      PyPi: https://pypi.org/project/pytrends/
      GitHub: https://github.com/GeneralMills/pytrends
    Request category: all, geo: world, property: web searches.

    Args:
      keywords: string
        Search Keyword
      start_date: string
        Date in the format 'YYYY-MM-DD'
      end_date: string
        Date in the format 'YYYY-MM-DD'

    Returns:
      pandas.core.frame.DataFrame
    """
    if isinstance(keywords, str):
        keywords = [keywords]

    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keywords, timeframe=f"{start_date} {end_date}")
    return pytrend.interest_over_time()


def _wrangle_gtrends(df):
    """Wrangle the weekly based data from Google Trends.

    Args:
      df: pandas.core.frame.DataFrame
        DataFrame from request_gtrends()

    Returns:
      pandas.core.frame.DataFrame
    """
    # Lower column names, reset index, drop column, format date, rename columns
    df.columns = df.columns.str.lower()
    df = df.reset_index()
    df = df.drop(["ispartial"], axis=1)
    df["date"] = [i.strftime("%Y-%m-%d") for i in df["date"]]
    df.columns = ["date", "gtrend"]
    return df


def get_bitcoin_data(yfd=None, gtd=None):
    """Combine all data sources.

    Args:
      yfd: pandas.core.frame.DataFrame
        DataFrame from wrangle_yfinance()
      gtd: pandas.core.frame.DataFrame
        DataFrame from wrangle_gtrends()

    Returns:
      pandas.core.frame.DataFrame
    """
    # Request Yahoo! Finance
    if yfd is None:
        yfd = _wrangle_yfinance(_request_yfinance("BTC-USD"))

    # Request Google Trends
    if gtd is None:
        try:
            start_date = yfd.index[0].strftime("%Y-%m-%d")
            end_date = yfd.index[-1].strftime("%Y-%m-%d")
            gtd = _wrangle_gtrends(
                _request_gtrends(
                    keywords="Bitcoin", start_date=start_date, end_date=end_date
                )
            )
        except Exception as e:
            print("[WARNING] Cannot request Google Trends:")
            print(e)

    # Combine both DataFrames
    if yfd is not None and gtd is not None:
        df = yfd.merge(gtd, how="left", on="date").set_axis(yfd.index)

        # Interpolate Google Trend data
        idf = df[["date", "gtrend"]].drop_duplicates()
        idf = idf.interpolate(method="linear", limit_direction="both")
        idf["gtrend"] = [round(i, 1) for i in idf["gtrend"]]
        df = df.drop(["gtrend"], axis=1)
        df = df.merge(idf, how="left", on="date").set_axis(df.index)

    elif yfd is not None and gtd is None:
        df = yfd.copy()

    # Drop date and rows with NaN (because of Moving Average)
    df = df.drop(["date"], axis=1)
    df = df.dropna()
    return df
