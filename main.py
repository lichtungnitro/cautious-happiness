#!/usr/bin/env python3
"""Investment Analysis Agent for A-share Market Data"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd
from pandas import DataFrame
from typing import cast
import akshare as ak
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.tools import ShellTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
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

        if not all([symbol, start_date, end_date]):
            return "Error: Missing required parameters. Format: symbol=CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"

        akshare_params = {
            "symbol": symbol,
            "period": period,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }
        if timeout is not None:
            akshare_params["timeout"] = timeout

        df = ak.stock_zh_a_hist(**akshare_params)

        if df.empty:
            return f"No data found for {symbol} from {start_date} to {end_date}"

        if len(df) > 10:
            summary_df = pd.concat([df.head(3), df.tail(3)])
            return f"Data for {symbol} ({len(df)} records):\n{summary_df.to_string()}"
        else:
            return f"Data for {symbol}:\n{df.to_string()}"

    except Exception as e:
        return f"Error fetching data: {str(e)}"


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

        if not notice_type or not date:
            return "Error: Missing required parameters. Format: notice_type=TYPE,date=YYYYMMDD[,stock_code=CODE]"

        # Validate notice_type
        valid_types = {"全部", "重大事项", "财务报告", "融资公告", "风险提示", "资产重组", "信息变更", "持股变动"}
        if notice_type not in valid_types:
            return f"Error: Invalid notice_type. Valid options: {', '.join(valid_types)}"

        # Call API with required parameters
        df = cast(DataFrame, ak.stock_notice_report(symbol=notice_type, date=date))

        if df.empty:
            return f"No announcements found for {notice_type} on {date}"

        # Filter by stock code if provided
        if stock_code:
            filtered_df = cast(DataFrame, df[df['代码'] == stock_code])

            if filtered_df.empty:
                return f"No announcements found for stock {stock_code} on {date}"

            if len(filtered_df) > 10:
                summary_df = filtered_df.head(10)
                return f"Announcements for {stock_code} ({len(filtered_df)} total):\n{summary_df.to_string()}"
            else:
                return f"Announcements for {stock_code}:\n{filtered_df.to_string()}"
        else:
            # Return all announcements for the type/date
            if len(df) > 10:
                summary_df = df.head(10)
                return f"All {notice_type} announcements on {date} ({len(df)} total):\n{summary_df.to_string()}"
            else:
                return f"All {notice_type} announcements on {date}:\n{df.to_string()}"

    except Exception as e:
        return f"Error fetching notifications: {str(e)}"


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

    tools = [get_historic_stock_data, get_current_time, get_stock_notifications]

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

    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
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

        # Check for critical investment decisions
        if check_critical_decision(result['output']):
            title = "CRITICALwINVESTMENT DECISION"
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
