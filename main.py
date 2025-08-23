#!/usr/bin/env python3
"""Investment Analysis Agent for A-share Market Data"""

import argparse
import os
import sys
from typing import Optional, Union

import pandas as pd
from pandas import DataFrame
from typing import cast
import akshare as ak
import requests
import numpy as np
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.agents import AgentOutputParser, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import ShellTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.chains import LLMChain
import re
from dotenv import load_dotenv
from loguru import logger

from util.moe import moe_prompt


class StockHistoryInput(BaseModel):
    """Input for fetching historical stock data."""
    symbol: str = Field(..., description="The A-share stock code (e.g., '000001').")
    start_date: str = Field(..., description="Start date in YYYYMMDD format.")
    end_date: str = Field(..., description="End date in YYYYMMDD format.")
    period: str = Field('daily', description="Data frequency: daily, weekly, or monthly.")
    adjust: Optional[str] = Field(None, description="Adjustment method: qfq, hfq, or None.")
    timeout: Optional[float] = Field(None, description="Request timeout in seconds.")


class StockNoticeInput(BaseModel):
    """Input for fetching stock notifications/announcements."""
    notice_type: str = Field(..., description="Announcement type: 全部, 重大事项, 财务报告, 融资公告, 风险提示, 资产重组, 信息变更, 持股变动")
    date: str = Field(..., description="Specific date in YYYYMMDD format (e.g., '20220511')")
    stock_code: Optional[str] = Field(None, description="Optional A-share stock code to filter results (e.g., '000001')")


class RealTimeQuoteInput(BaseModel):
    """Input for fetching real-time stock quotes."""
    symbol: str = Field(..., description="The A-share stock code (e.g., '000001').")


class FinancialStatementInput(BaseModel):
    """Input for fetching financial statements."""
    symbol: str = Field(..., description="The A-share stock code (e.g., '000001').")
    report_type: str = Field("balance", description="Report type: balance, income, cashflow")
    period: str = Field("latest", description="Period: latest, quarterly, annual")


@tool("get_current_time")
def get_current_time(query: str = "beijing") -> str:
    """Get current time in Beijing timezone.

    Args:
        query: Timezone (default: 'beijing')

    Returns:
        Current time in Beijing timezone as string
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    beijing_tz = ZoneInfo("Asia/Shanghai")
    current_time = datetime.now(beijing_tz)
    return f"Current Beijing time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"


@tool("get_historic_stock_data")
def get_historic_stock_data(query: str) -> str:
    """Retrieve historical A-share stock market data.

        Fetches historical stock market data for Chinese A-share stocks based on the
        provided query parameters. The function accepts a formatted query string and
        returns tabular stock data or an error message.

        Args:
            query (str): Comma-separated query string containing stock parameters in
                key=value format. Required parameters include symbol, start_date,
                end_date, and period. Optional parameters include adjust and timeout.

                Required parameters:
                    - symbol: A-share stock code (e.g., "000001")
                    - start_date: Start date in YYYYMMDD format (e.g., "20230801")
                    - end_date: End date in YYYYMMDD format (e.g., "20230810")
                    - period: Data frequency ("daily", "weekly", or "monthly")

                Optional parameters:
                    - adjust: Price adjustment method ("qfq" for front-adjusted,
                             "hfq" for back-adjusted, or empty for unadjusted,
                             default: empty)
                    - timeout: Request timeout in seconds (default: system default)

        Returns:
            str: Tabular representation of historical stock data, or error message
                 if data retrieval fails.

        Example:
            > query = "symbol=000001,start_date=20230801,end_date=20230810,period=daily,adjust=qfq"
            > result = get_historical_stock_data(query)

        Raises:
            This function handles errors internally and returns error messages as strings
            rather than raising exceptions.
        """
    try:
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip("'\"")

        symbol = params.get('symbol')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        period = params.get('period', 'daily')
        adjust = params.get('adjust')
        timeout = float(params['timeout']) if 'timeout' in params else None

        # Enhanced validation
        if not symbol:
            return "Error: Missing required parameter 'symbol'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not start_date:
            return "Error: Missing required parameter 'start_date'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not end_date:
            return "Error: Missing required parameter 'end_date'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"

        # Validate date format
        if len(start_date) != 8 or not start_date.isdigit():
            return f"Error: Invalid start_date format '{start_date}'. Must be YYYYMMDD format."
        if len(end_date) != 8 or not end_date.isdigit():
            return f"Error: Invalid end_date format '{end_date}'. Must be YYYYMMDD format."

        # Validate period
        valid_periods = {"daily", "weekly", "monthly"}
        if period not in valid_periods:
            return f"Error: Invalid period '{period}'. Must be one of: {', '.join(valid_periods)}"

        # Validate adjust if provided
        if adjust and adjust not in {"qfq", "hfq", ""}:
            return f"Error: Invalid adjust method '{adjust}'. Must be 'qfq', 'hfq', or empty."

        akshare_params = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust if adjust else "",
        }
        if timeout is not None:
            akshare_params["timeout"] = timeout

        df = ak.stock_zh_a_hist(**akshare_params)

        if df.empty:
            return f"No data found for stock {symbol} from {start_date} to {end_date} with period '{period}' and adjust '{adjust}'. Please verify the stock code and date range."

        # Enhanced data summary
        if len(df) > 15:
            summary_df = pd.concat([df.head(5), df.tail(5)])
            summary_stats = f"Records: {len(df)}, Date Range: {df.iloc[0]['日期']} to {df.iloc[-1]['日期']}"
            return f"Data for {symbol} ({summary_stats}):\n{summary_df.to_string()}"
        else:
            date_range = f"{df.iloc[0]['日期']} to {df.iloc[-1]['日期']}" if len(df) > 0 else "No dates"
            return f"Data for {symbol} ({len(df)} records, {date_range}):\n{df.to_string()}"

    except ValueError as ve:
        return f"Validation error: {str(ve)}. Please check parameter formats."
    except ImportError:
        return "Error: Required data library (akshare) not available. Please install akshare package."
    except ConnectionError:
        return "Error: Network connection failed. Please check your internet connection and try again."
    except TimeoutError:
        return "Error: Data request timed out. Please try again with a longer timeout or check network conditions."
    except Exception as e:
        return f"Error fetching data: {str(e)}. Please verify parameters and try again."


@tool("get_stock_notifications")
def get_stock_notifications(query: str) -> str:
    """Retrieve stock announcements and notifications for Chinese A-share stocks.

    Fetches stock announcements and notifications from Chinese A-share market based on
    the provided query parameters. The API requires specific date and announcement type,
    with optional stock code filtering.

    Args:
        query (str): Comma-separated query string containing parameters in
            key=value format. Required parameters include notice_type and date.
            Optional parameter stock_code can be provided to filter specific stock.

            Required parameters:
                - notice_type: Announcement type ("全部", "重大事项", "财务报告",
                              "融资公告", "风险提示", "资产重组", "信息变更", "持股变动")
                - date: Specific date in YYYYMMDD format (e.g., "20220511")

            Optional parameters:
                - stock_code: A-share stock code to filter results (e.g., "000001")

    Returns:
        str: Tabular representation of stock announcements, or error message
             if data retrieval fails.

    Example:
        >>> query = "notice_type=财务报告,date=20220511"
        >>> result = get_stock_notifications(query)
        >>> print(result)
        # Returns announcements for financial reports on 2022-05-11

        >>> query = "notice_type=重大事项,date=20220511,stock_code=000001"
        >>> result = get_stock_notifications(query)
        >>> print(result)
        # Returns major event announcements for stock 000001 on 2022-05-11

    Raises:
        This function handles errors internally and returns error messages as strings
        rather than raising exceptions.
    """
    try:
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip("'\"")

        notice_type = params.get('notice_type')
        date = params.get('date')
        stock_code = params.get('stock_code')

        # Enhanced validation
        if not notice_type:
            return "Error: Missing required parameter 'notice_type'. Format: notice_type=TYPE,date=YYYYMMDD[,stock_code=CODE]"
        if not date:
            return "Error: Missing required parameter 'date'. Format: notice_type=TYPE,date=YYYYMMDD[,stock_code=CODE]"

        # Validate date format
        if len(date) != 8 or not date.isdigit():
            return f"Error: Invalid date format '{date}'. Must be YYYYMMDD format."

        # Validate notice_type
        valid_types = {"全部", "重大事项", "财务报告", "融资公告", "风险提示", "资产重组", "信息变更", "持股变动"}
        if notice_type not in valid_types:
            return f"Error: Invalid notice_type '{notice_type}'. Valid options: {', '.join(sorted(valid_types))}"

        # Validate stock code format if provided
        if stock_code and (len(stock_code) != 6 or not stock_code.isdigit()):
            return f"Error: Invalid stock code format '{stock_code}'. Must be 6-digit A-share code (e.g., '000001')."

        # Call API with required parameters
        df = cast(DataFrame, ak.stock_notice_report(symbol=notice_type, date=date))

        if df.empty:
            return f"No announcements found for {notice_type} on {date}. Please verify the date and announcement type."

        # Filter by stock code if provided
        if stock_code:
            filtered_df = cast(DataFrame, df[df['代码'] == stock_code])

            if filtered_df.empty:
                return f"No announcements found for stock {stock_code} on {date} with type {notice_type}."

            if len(filtered_df) > 8:
                summary_df = filtered_df.head(8)
                return f"Announcements for {stock_code} ({len(filtered_df)} total, showing first 8):\n{summary_df.to_string()}"
            else:
                return f"Announcements for {stock_code} ({len(filtered_df)} total):\n{filtered_df.to_string()}"
        else:
            # Return all announcements for the type/date
            if len(df) > 8:
                summary_df = df.head(8)
                return f"All {notice_type} announcements on {date} ({len(df)} total, showing first 8):\n{summary_df.to_string()}"
            else:
                return f"All {notice_type} announcements on {date} ({len(df)} total):\n{df.to_string()}"

    except ImportError:
        return "Error: Required data library (akshare) not available. Please install akshare package."
    except ConnectionError:
        return "Error: Network connection failed. Please check your internet connection and try again."
    except TimeoutError:
        return "Error: Data request timed out. Please try again later."
    except ValueError as ve:
        return f"Validation error: {str(ve)}. Please check parameter formats."
    except Exception as e:
        return f"Error fetching notifications: {str(e)}. Please verify parameters and try again."


@tool("get_real_time_quote")
def get_real_time_quote(query: str) -> str:
    """Retrieve real-time stock quote for Chinese A-share stocks.

    Fetches current market data including price, volume, and other trading metrics
    for Chinese A-share stocks.

    Args:
        query (str): Comma-separated query string containing stock parameters in
            key=value format. Required parameter: symbol.

            Required parameter:
                - symbol: A-share stock code (e.g., "000001")

    Returns:
        str: Real-time quote information, or error message if data retrieval fails.

    Example:
        > query = "symbol=000001"
        > result = get_real_time_quote(query)
    """
    try:
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip("'\"")

        symbol = params.get('symbol')

        if not symbol:
            return "Error: Missing required parameter 'symbol'. Format: symbol=CODE"

        # Validate stock code format
        if len(symbol) != 6 or not symbol.isdigit():
            return f"Error: Invalid stock code format '{symbol}'. Must be 6-digit A-share code (e.g., '000001')."

        # Fetch real-time data
        df = ak.stock_zh_a_spot_em()
        
        if df.empty:
            return "Error: No real-time data available. Please try again later."

        # Filter for the specific stock
        stock_data = df[df['代码'] == symbol]

        if stock_data.empty:
            return f"Error: No real-time data found for stock {symbol}. Please verify the stock code."

        # Extract relevant information
        stock_info = stock_data.iloc[0]
        return (
            f"Real-time quote for {symbol} ({stock_info['名称']}):\n"
            f"Current Price: {stock_info['最新价']} | Change: {stock_info['涨跌幅']}% | Change Amount: {stock_info['涨跌额']}\n"
            f"Open: {stock_info['今开']} | High: {stock_info['最高']} | Low: {stock_info['最低']} | Previous Close: {stock_info['昨收']}\n"
            f"Volume: {stock_info['成交量']} | Turnover: {stock_info['成交额']}\n"
            f"PE Ratio: {stock_info['市盈率-动态']} | PB Ratio: {stock_info['市净率']}"
        )

    except ImportError:
        return "Error: Required data library (akshare) not available. Please install akshare package."
    except ConnectionError:
        return "Error: Network connection failed. Please check your internet connection and try again."
    except TimeoutError:
        return "Error: Data request timed out. Please try again later."
    except Exception as e:
        return f"Error fetching real-time quote: {str(e)}. Please verify parameters and try again."


@tool("get_financial_statements")
def get_financial_statements(query: str) -> str:
    """Retrieve financial statements for Chinese A-share stocks.

    Fetches balance sheets, income statements, or cash flow statements
    for Chinese A-share companies.

    Args:
        query (str): Comma-separated query string containing parameters in
            key=value format. Required parameter: symbol.
            Optional parameters: report_type, period.

            Required parameter:
                - symbol: A-share stock code (e.g., "000001")

            Optional parameters:
                - report_type: Type of financial statement ("balance", "income", "cashflow")
                - period: Reporting period ("latest", "quarterly", "annual")

    Returns:
        str: Financial statement data, or error message if data retrieval fails.

    Example:
        > query = "symbol=000001,report_type=balance,period=latest"
        > result = get_financial_statements(query)
    """
    try:
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip("'\"")

        symbol = params.get('symbol')
        report_type = params.get('report_type', 'balance')
        period = params.get('period', 'latest')

        if not symbol:
            return "Error: Missing required parameter 'symbol'. Format: symbol=CODE[,report_type=TYPE,period=PERIOD]"

        # Validate stock code format
        if len(symbol) != 6 or not symbol.isdigit():
            return f"Error: Invalid stock code format '{symbol}'. Must be 6-digit A-share code (e.g., '000001')."

        # Validate report_type
        valid_report_types = {"balance", "income", "cashflow"}
        if report_type not in valid_report_types:
            return f"Error: Invalid report_type '{report_type}'. Must be one of: {', '.join(valid_report_types)}"

        # Validate period
        valid_periods = {"latest", "quarterly", "annual"}
        if period not in valid_periods:
            return f"Error: Invalid period '{period}'. Must be one of: {', '.join(valid_periods)}"

        # Use alternative approach for financial data
        try:
            # Get real-time data which contains basic financial metrics
            real_time_df = ak.stock_zh_a_spot_em()
            if real_time_df.empty:
                return "Financial data service currently unavailable. Please try again later."

            stock_data = real_time_df[real_time_df['代码'] == symbol]
            if stock_data.empty:
                return f"No financial data found for stock {symbol}. Please verify the stock code."

            stock_info = stock_data.iloc[0]
            
            # Safely access columns with fallbacks
            stock_name = stock_info.get('名称', 'N/A')
            pe_ratio = stock_info.get('市盈率-动态', 'N/A')
            pb_ratio = stock_info.get('市净率', 'N/A')
            market_cap = stock_info.get('总市值', 'N/A')
            current_price = stock_info.get('最新价', 'N/A')
            
            # Provide basic financial metrics
            basic_financials = (
                f"Basic Financial Metrics for {symbol} ({stock_name}):\n"
                f"Market Cap: {market_cap} | PE Ratio: {pe_ratio}\n"
                f"PB Ratio: {pb_ratio} | Current Price: {current_price}\n"
                f"Note: Detailed financial statements (balance sheet, income statement, cash flow) "
                f"are temporarily unavailable. Basic valuation metrics show "
                f"PE ratio: {pe_ratio}, PB ratio: {pb_ratio}."
            )
            
            return basic_financials

        except Exception as api_error:
            return f"Financial data service error: {str(api_error)}. Detailed financial statements are currently unavailable. Please try basic stock analysis with real-time quotes or historical data."

    except ImportError:
        return "Error: Required data library (akshare) not available. Please install akshare package."
    except ConnectionError:
        return "Error: Network connection failed. Please check your internet connection and try again."
    except TimeoutError:
        return "Error: Data request timed out. Please try again later."
    except Exception as e:
        return f"Error fetching financial statements: {str(e)}. Please verify parameters and try again."


@tool("calculate_investment_performance")
def calculate_investment_performance(query: str) -> str:
    """Calculate investment performance metrics for a stock or portfolio.

    Computes various performance metrics including returns, volatility,
    Sharpe ratio, and other risk-adjusted measures for investment analysis.

    Args:
        query (str): Comma-separated query string containing parameters in
            key=value format. Required parameters: symbol, start_date, end_date.
            Optional parameters: initial_investment, benchmark.

            Required parameters:
                - symbol: A-share stock code (e.g., "000001")
                - start_date: Investment start date in YYYYMMDD format
                - end_date: Investment end date in YYYYMMDD format

            Optional parameters:
                - initial_investment: Initial investment amount (default: 10000)
                - benchmark: Benchmark symbol for comparison (e.g., "000300" for CSI 300)

    Returns:
        str: Performance metrics analysis, or error message if calculation fails.

    Example:
        > query = "symbol=000001,start_date=20230101,end_date=20231231,initial_investment=100000"
        > result = calculate_investment_performance(query)
    """
    try:
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip("'\"")

        symbol = params.get('symbol')
        start_date = params.get('start_date')
        end_date = params.get('end_date')
        initial_investment = float(params.get('initial_investment', 10000))
        benchmark = params.get('benchmark')

        # Validate required parameters
        if not symbol:
            return "Error: Missing required parameter 'symbol'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not start_date:
            return "Error: Missing required parameter 'start_date'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not end_date:
            return "Error: Missing required parameter 'end_date'. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"

        # Validate date format
        if len(start_date) != 8 or not start_date.isdigit():
            return f"Error: Invalid start_date format '{start_date}'. Must be YYYYMMDD format."
        if len(end_date) != 8 or not end_date.isdigit():
            return f"Error: Invalid end_date format '{end_date}'. Must be YYYYMMDD format."

        # Fetch historical data
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        
        if df.empty:
            return f"No historical data found for {symbol} from {start_date} to {end_date}"

        # Calculate performance metrics
        start_price = df.iloc[0]['收盘']
        end_price = df.iloc[-1]['收盘']
        
        # Total return
        total_return = (end_price / start_price - 1) * 100
        
        # Annualized return
        days_held = len(df)
        annualized_return = ((1 + total_return/100) ** (365/days_held) - 1) * 100
        
        # Investment value
        final_value = initial_investment * (1 + total_return/100)
        profit_loss = final_value - initial_investment
        
        # Volatility (daily standard deviation)
        daily_returns = df['收盘'].pct_change().dropna()
        volatility = daily_returns.std() * 100 * np.sqrt(252)  # Annualized volatility
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annualized_return/100 - risk_free_rate) / (volatility/100) if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min() * 100

        performance_analysis = (
            f"Performance Analysis for {symbol} ({start_date} to {end_date}):\n"
            f"Initial Investment: {initial_investment:,.2f} | Final Value: {final_value:,.2f} | P&L: {profit_loss:,.2f}\n"
            f"Total Return: {total_return:.2f}% | Annualized Return: {annualized_return:.2f}%\n"
            f"Volatility (Annualized): {volatility:.2f}% | Sharpe Ratio: {sharpe_ratio:.2f}\n"
            f"Maximum Drawdown: {max_drawdown:.2f}% | Holding Period: {days_held} days\n"
        )

        # Add benchmark comparison if provided
        if benchmark:
            try:
                bench_df = ak.stock_zh_a_hist(symbol=benchmark, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
                if not bench_df.empty:
                    bench_start = bench_df.iloc[0]['收盘']
                    bench_end = bench_df.iloc[-1]['收盘']
                    bench_return = (bench_end / bench_start - 1) * 100
                    alpha = total_return - bench_return
                    
                    performance_analysis += (
                        f"\nBenchmark Comparison ({benchmark}):\n"
                        f"Benchmark Return: {bench_return:.2f}% | Alpha: {alpha:.2f}%\n"
                        f"Outperformance: {'Yes' if alpha > 0 else 'No'} ({alpha:.2f}%)"
                    )
            except Exception:
                performance_analysis += "\nBenchmark comparison unavailable"

        return performance_analysis

    except Exception as e:
        return f"Error calculating performance: {str(e)}. Please verify parameters and try again."


def send_bark_notification(title: str, content: str) -> None:
    """Send notification via Bark service."""
    bark_key = os.getenv('BARK_API_KEY')
    if not bark_key:
        logger.warning("BARK_API_KEY not found. Skipping notification.")
        return

    try:
        url = f"https://api.day.app/{bark_key}/{title}/{content}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("Bark notification sent successfully")
        else:
            logger.warning(f"Failed to send Bark notification: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending Bark notification: {e}")


def send_bark_critical_alert(title: str, content: str) -> None:
    """Send critical alert via Bark service for investment decisions."""
    bark_key = os.getenv('BARK_API_KEY')
    if not bark_key:
        logger.warning("BARK_API_KEY not found. Skipping critical alert.")
        return

    try:
        url = f"https://api.day.app/{bark_key}/{title}/{content}?level=critical&volume=3"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            logger.info("Bark critical alert sent successfully")
        else:
            logger.warning(f"Failed to send Bark critical alert: {response.status_code}")
    except Exception as e:
        logger.error(f"Error sending Bark critical alert: {e}")


def markdown_to_plain_text(markdown_text: str) -> str:
    """Convert markdown text to plain text for Bark notifications."""
    # Remove markdown formatting
    plain_text = markdown_text
    
    # Remove bold/italic: **text** or __text__ -> text
    plain_text = re.sub(r'[*_]{2}(.*?)[*_]{2}', r'\1', plain_text)
    
    # Remove single asterisk/underscore: *text* or _text_ -> text
    plain_text = re.sub(r'[*_](.*?)[*_]', r'\1', plain_text)
    
    # Remove code blocks: `code` -> code
    plain_text = re.sub(r'`(.*?)`', r'\1', plain_text)
    
    # Remove links: [text](url) -> text
    plain_text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', plain_text)
    
    # Remove headers: # Header -> Header
    plain_text = re.sub(r'^#+\s*', '', plain_text, flags=re.MULTILINE)
    
    # Remove blockquotes: > text -> text
    plain_text = re.sub(r'^>\s*', '', plain_text, flags=re.MULTILINE)
    
    # Remove horizontal rules
    plain_text = re.sub(r'^[-*_]{3,}\s*$', '', plain_text, flags=re.MULTILINE)
    
    # Clean up extra spaces and newlines
    plain_text = re.sub(r'\s+', ' ', plain_text)
    plain_text = plain_text.strip()
    
    return plain_text


def send_investment_alert(alert_type: str, symbol: str, message: str, confidence: str = "medium") -> None:
    """Send specific investment alerts with standardized formatting."""
    alert_types = {
        "buy": "🟢 BUY ALERT",
        "sell": "🔴 SELL ALERT", 
        "hold": "🟡 HOLD ALERT",
        "warning": "⚠️ WARNING",
        "opportunity": "🎯 OPPORTUNITY",
        "performance": "📊 PERFORMANCE"
    }
    
    prefix = alert_types.get(alert_type, "📋 INVESTMENT")
    title = f"{prefix} {symbol}"
    
    # Convert markdown to plain text
    plain_message = markdown_to_plain_text(message)
    
    # Add confidence level to message
    if confidence:
        plain_message = f"[{confidence.upper()} CONFIDENCE] {plain_message}"
    
    # Use critical alert for sell signals and warnings
    if alert_type in ["sell", "warning"]:
        send_bark_critical_alert(title, plain_message)
    else:
        send_bark_notification(title, plain_message)


def parse_investment_decisions(response: str) -> list:
    """Parse investment decisions from agent response for targeted alerts."""
    decisions = []
    response_lower = response.lower()
    
    # Look for specific decision patterns
    decision_patterns = [
        ("buy", ["buy", "买入", "long", "做多", "recommend buying", "initiate position"]),
        ("sell", ["sell", "卖出", "short", "做空", "recommend selling", "divest", "reduce exposure"]),
        ("hold", ["hold", "持有", "maintain", "keep position", "continue holding"]),
        ("warning", ["warning", "caution", "risk", "danger", "concern", "problem"]),
        ("opportunity", ["opportunity", "undervalued", "attractive", "potential", "upside"])
    ]
    
    # Remove confidence level markers from context to avoid duplication
    clean_context = response
    confidence_patterns = [
        r"\[HIGH CONFIDENCE\]",
        r"\[MEDIUM CONFIDENCE\]", 
        r"\[LOW CONFIDENCE\]",
        r"\[INVALID\]",
        r"high confidence",
        r"medium confidence",
        r"low confidence"
    ]
    
    for pattern in confidence_patterns:
        clean_context = re.sub(pattern, "", clean_context, flags=re.IGNORECASE)
    
    # Clean up extra spaces
    clean_context = re.sub(r'\s+', ' ', clean_context).strip()
    
    for alert_type, keywords in decision_patterns:
        for keyword in keywords:
            if keyword in response_lower:
                # Extract stock symbol if mentioned
                symbol_match = None
                if alert_type != "performance":
                    # Look for 6-digit stock codes
                    symbol_match = re.search(r'\b(\d{6})\b', response)
                
                symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
                
                # Extract confidence level
                confidence = "medium"
                if "high confidence" in response_lower:
                    confidence = "high"
                elif "low confidence" in response_lower:
                    confidence = "low"
                
                decisions.append({
                    "type": alert_type,
                    "symbol": symbol,
                    "confidence": confidence,
                    "context": clean_context[:200] + "..." if len(clean_context) > 200 else clean_context
                })
                break  # Only need one match per type
    
    return decisions


class ConfidenceAwareOutputParser(AgentOutputParser):
    """Custom output parser that continues execution for low confidence responses."""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # Check if this is a high confidence final answer
        if "[HIGH CONFIDENCE]" in text and "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        
        # For medium/low/invalid confidence, continue execution
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, text, re.DOTALL)
        
        if not match:
            # If no action found but we have Final Answer with low confidence, 
            # we should still continue to get more data
            if "Final Answer:" in text:
                # Extract the part after Final Answer to use as context
                text.split("Final Answer:")[-1].strip()
                # Return an action to get more data
                return AgentAction(
                    tool="get_historic_stock_data", 
                    tool_input="symbol=000001,start_date=20230101,end_date=20231231,period=daily",
                    log=text
                )
            raise ValueError(f"Could not parse LLM output: `{text}`")
        
        action = match.group(1).strip()
        action_input = match.group(2)
        
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=text
        )


def check_critical_decision(response: str) -> bool:
    """Check if response contains long/short investment decisions."""
    response_lower = response.lower()

    # Keywords that indicate investment decisions
    decision_keywords = [
        'long', 'short', 'buy', 'sell', 'hold', 'position',
        '买入', '卖出', '持有', '做多', '做空', '建仓', '平仓'
    ]

    # Check for investment decision patterns
    for keyword in decision_keywords:
        if keyword in response_lower:
            return True

    return False


def create_agent():
    """Create and configure the investment analysis agent."""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        logger.error("GEMINI_API_KEY not found. Set your API key in .env file.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    # Core financial data tools
    tools = [
        get_historic_stock_data, 
        get_current_time, 
        get_stock_notifications,
        get_real_time_quote,
        get_financial_statements,
        calculate_investment_performance
    ]

    try:
        tools.extend([YahooFinanceNewsTool(), ShellTool()])
    except Exception as e:
        logger.warning(f"Could not load some tools: {e}")

    try:
        additional_tools = load_tools(["wikipedia", "llm-math"], llm=llm)
        tools.extend(additional_tools)
    except Exception as e:
        logger.warning(f"Could not load additional tools: {e}")

    prompt = PromptTemplate(
        template=moe_prompt,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
    )

    # Create LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Create custom output parser
    output_parser = ConfidenceAwareOutputParser()
    
    # Create single action agent with custom output parser
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=15  # Increased for more iterations with low confidence
    )


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Investment Analysis Agent for A-share Market")
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Investment query to analyze (e.g., 'Get historical data for stock 000001 from 20230801 to 20230810')"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="enable verbose logging"
    )

    args = parser.parse_args()

    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    try:
        agent = create_agent()
        logger.info("Agent initialized successfully")

        result = agent.invoke({"input": args.query})
        print("\n" + "="*50)
        print("ANALYSIS RESULT")
        print("="*50)
        print(result['output'])

        # Parse and send targeted investment alerts
        decisions = parse_investment_decisions(result['output'])
        
        if decisions:
            # Send specific alerts for each decision
            for decision in decisions:
                send_investment_alert(
                    decision['type'],
                    decision['symbol'],
                    decision['context'],
                    decision['confidence']
                )
            logger.info(f"Sent {len(decisions)} targeted investment alerts")
        else:
            # Fallback to general notification
            if check_critical_decision(result['output']):
                title = "CRITICAL INVESTMENT DECISION"
                content = result['output'][:300] + "..." if len(result['output']) > 300 else result['output']
                send_bark_critical_alert(title, content)
                logger.info("Critical investment decision detected - sent critical alert")
            else:
                # Send normal notification
                title = "Investment Analysis Complete"
                content = result['output'][:200] + "..." if len(result['output']) > 200 else result['output']
                send_bark_notification(title, content)

    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        # Send error notification
        send_bark_notification("Investment Analysis Failed", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
