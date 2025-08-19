import pandas as pd
import akshare as ak
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import tool
from typing import Optional
import os

# Core LangChain imports for agent creation
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor

# Load other tools
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.agents import load_tools
from langchain_community.tools import ShellTool

# Import the LangChain Hub for pre-built prompts
from langchain import hub

# --- Tool Input Schema ---
class StockHistoryInput(BaseModel):
    """Input for fetching historical stock data."""
    symbol: str = Field(..., description="The A-share stock code (e.g., '000001').")
    start_date: str = Field(..., description="Start date in YYYYMMDD format (e.g., '20230101').")
    end_date: str = Field(..., description="End date in YYYYMMDD format (e.g., '20230616').")
    period: str = Field('daily', description="The data frequency. Must be 'daily', 'weekly', or 'monthly'.")
    adjust: Optional[str] = Field('hfq', description="The adjustment method. 'qfq' for front-adjusted, 'hfq' for back-adjusted (recommended for quantitative analysis), or '' for no adjustment.")
    timeout: Optional[float] = Field(None, description="Timeout for the data fetching request in seconds.")

# --- The Tool Function ---
@tool("get_historic_stock_data")
def get_historic_stock_data(query: str) -> str:
    """
    Retrieves historic A-share stock market data (daily, weekly, or monthly).

    This tool fetches historical stock data from the Chinese A-share market.
    Input should be in the format: "symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD,period=daily,adjust=hfq"

    Example: "symbol=000001,start_date=20230801,end_date=20230810,period=daily,adjust=hfq"

    Args:
        query (str): A comma-separated string with parameters in key=value format.
                    Required: symbol, start_date, end_date
                    Optional: period (default: daily), adjust (default: hfq), timeout

    Returns:
        A string representation of the historic stock data in a table format, or an
        error message if the data could not be retrieved.
    """
    try:
        # Parse the input string
        params = {}
        for item in query.split(','):
            if '=' in item:
                key, value = item.strip().split('=', 1)
                params[key.strip()] = value.strip().strip("'\"")

        # Extract required parameters
        symbol = params.get('symbol')
        start_date = params.get('start_date')
        end_date = params.get('end_date')

        # Extract optional parameters with defaults
        period = params.get('period', 'daily')
        adjust = params.get('adjust', 'hfq')
        timeout = params.get('timeout')
        if timeout:
            timeout = float(timeout)

        # Validate required parameters
        if not symbol:
            return "Error: 'symbol' parameter is required. Format: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not start_date:
            return "Error: 'start_date' parameter is required. Format: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"
        if not end_date:
            return "Error: 'end_date' parameter is required. Format: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"

        print(f"Fetching data for symbol={symbol}, start_date={start_date}, end_date={end_date}, period={period}, adjust={adjust}")

        # Fetch data using akshare
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            timeout=timeout
        )

        if df.empty:
            return f"No historic data found for stock {symbol} from {start_date} to {end_date} with period '{period}' and adjust '{adjust}'."

        # Convert the DataFrame to a string representation that the LLM can easily read and summarize.
        # Limit the output to avoid overwhelming the model
        if len(df) > 10:
            summary_df = pd.concat([df.head(5), df.tail(5)])
            return f"Historical data for {symbol} (showing first 5 and last 5 records out of {len(df)} total):\n{summary_df.to_string()}"
        else:
            return f"Historical data for {symbol}:\n{df.to_string()}"

    except Exception as e:
        # A robust tool should always handle exceptions gracefully and provide a clear
        # message back to the agent, explaining why the tool failed.
        return f"An error occurred while fetching stock data: {e}. Please ensure the format is: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD"

def main():
    # --- 1. Set up API key ---
    # Replace with your actual API key or set as environment variable
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')  # or use userdata.get('GEMINI_API_KEY') in Colab

    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Please set your API key.")
        # For testing purposes, you can set it directly here:
        # api_key = "your-api-key-here"
        return

    # --- 2. Initialize the model ---
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0
    )

    # --- 3. Define the tools ---
    # Start with our custom stock data tool
    tools = [get_historic_stock_data]

    # Add other tools (optional - comment out if you don't need them)
    try:
        finance_tool = YahooFinanceNewsTool()
        tools.append(finance_tool)
    except Exception as e:
        print(f"Could not load Yahoo Finance tool: {e}")

    # Load additional tools
    try:
        additional_tools = load_tools(["wikipedia", "llm-math"], llm=llm)
        tools.extend(additional_tools)
    except Exception as e:
        print(f"Could not load additional tools: {e}")

    # --- 4. Create the correct prompt ---
    # Use the standard ReAct prompt from the hub, which handles agent_scratchpad correctly
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception as e:
        print(f"Could not pull prompt from hub: {e}")
        # Fallback to a custom prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful financial assistant. Use the available tools to answer questions about the stock market, historical stock data, and financial news.

You have access to the following tools:

{tools}

When using the get_historic_stock_data tool, format your input as: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD,period=daily,adjust=hfq

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (for get_historic_stock_data, use format: symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""),
        ])

    # --- 5. Initialize the agent ---
    agent = create_react_agent(
        tools=tools,
        llm=llm,
        prompt=prompt
    )

    # --- 6. Create the AgentExecutor ---
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )

    # --- 7. Test the agent ---
    print("--- Agent is ready! ---")

    # Test query 1: Get specific date data
    try:
        print("\n=== Testing stock data query ===")
        result = agent_executor.invoke({
            "input": "What was the closing price of stock 600960 on August 18, 2023?"
        })
        print("Final result:", result['output'])
    except Exception as e:
        print(f"Error in first query: {e}")

    # Test query 2: Get recent data range
    try:
        print("\n=== Testing recent stock data query ===")
        result2 = agent_executor.invoke({
            "input": "Get the historical stock data for stock code 000001 from August 1, 2023 to August 10, 2023 using daily frequency."
        })
        print("Final result:", result2['output'])
    except Exception as e:
        print(f"Error in second query: {e}")

if __name__ == "__main__":
    main()
