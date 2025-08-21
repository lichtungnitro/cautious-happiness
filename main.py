#!/usr/bin/env python3
"""Investment Analysis Agent for A-share Market Data"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd
import akshare as ak
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


@tool("get_historic_stock_data")
def get_historic_stock_data(query: str) -> str:
    """Retrieve historic A-share stock market data.

    Input format: "symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD,period=daily,adjust=hfq"
    Example: "symbol=000001,start_date=20230801,end_date=20230810,period=daily,adjust=hfq"
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


def create_agent():
    """Create and configure the investment analysis agent."""
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')

    if not api_key:
        logger.error("GEMINI_API_KEY not found. Set your API key in .env file.")
        sys.exit(1)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0
    )

    tools = [get_historic_stock_data]

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

    except Exception as e:
        logger.error(f"Failed to process query: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
