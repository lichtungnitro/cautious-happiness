# Monism Agent - AI-Powered Investment Analysis System

A sophisticated multi-ego expert system for Chinese A-share market analysis, combining Google Gemini AI with collaborative decision-making.

## Features

- Multi-Ego Architecture: Four specialized investment personas with weighted decision-making
- Real-time Data: A-share market data integration via akshare
- AI Analysis: Google Gemini 2.0 Flash for intelligent insights
- Automated Notifications: Bark service integration for alerts
- Comprehensive Tools: Market data, news, calculations, and research capabilities

## Decision Framework

| Persona | Weight | Responsibility |
|---------|--------|----------------|
| Portfolio Manager | 50% | Strategic oversight and final decisions |
| Investment Analyst | 35% | Fundamental research and recommendations |
| Quantitative Analyst | 10% | Data-driven models and technical analysis |
| Trader | 5% | Execution and market mechanics |

## Quick Start

### Installation
```bash
uv pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
echo "BARK_API_KEY=your_bark_key_here" >> .env
```

### Usage
```bash
python main.py -q "Analyze recent price trends for 600960"
python main.py -q "symbol=000001,start_date=20230801,end_date=20230810,period=daily,adjust=hfq"
```

## Input Format

For stock data queries:
```
symbol=STOCK_CODE,start_date=YYYYMMDD,end_date=YYYYMMDD,period=daily,adjust=hfq
```

Examples:
- symbol=000001,start_date=20230101,end_date=20231231,period=daily,adjust=hfq
- symbol=600519,start_date=20230801,end_date=20230810,period=daily

## Tools Available

- **get_historic_stock_data**: A-share historical market data
- **get_stock_notifications**: Stock announcements and notifications from Chinese A-share market
- **yahoo_finance_news**: Financial news and updates
- **wikipedia**: General knowledge research
- **llm-math**: Mathematical calculations
- **terminal**: System command execution

### Stock Notifications Tool
The new `get_stock_notifications` tool provides access to Chinese A-share market announcements:

**Parameters:**
- `notice_type`: Announcement type (required)
  - Choices: `"全部"`, `"重大事项"`, `"财务报告"`, `"融资公告"`, `"风险提示"`, `"资产重组"`, `"信息变更"`, `"持股变动"`
- `date`: Specific date in YYYYMMDD format (required)
- `stock_code`: A-share stock code for filtering (optional)

**Usage Examples:**
```bash
# Get all financial reports for a specific date
python main.py -q "Get notifications for notice_type=财务报告,date=20220511"

# Get major events for specific stock on a date
python main.py -q "Get notifications for notice_type=重大事项,date=20220511,stock_code=000001"
```

## Notifications

The system includes Bark notification integration with two notification levels:

### Standard Notifications
Regular completion alerts for general analysis:

<img src="https://s2.loli.net/2025/08/21/6v7KNBHJDG3UCRo.jpg" alt="Standard Notification" width="300"/>

### Critical Investment Alerts
**Enhanced critical alerts** triggered when analysis contains investment decisions (buy/sell/hold recommendations):

- **Critical-level urgency** with `level=critical` parameter
- **Volume 3** for attention-grabbing alerts
- **Extended content** (300 chars vs 200 chars for normal notifications)
- **Special keywords**: Detects `long`, `short`, `buy`, `sell`, `hold`, `买入`, `卖出`, `持有`, etc.

<div style="display: flex; gap: 10px;">
  <img src="https://s2.loli.net/2025/08/21/JdL9tqjQvWE8K31.jpg" alt="Critical Alert Notification" width="300"/>
  <img src="https://s2.loli.net/2025/08/21/4yfLMgtOxlsCTDm.jpg" alt="Bark Message List" width="300"/>
</div>

### Setup
Configure your Bark API key:
```bash
echo "BARK_API_KEY=your_bark_key_here" >> .env
```

Critical alerts are automatically triggered when investment decisions are detected in the analysis output.

## Project Structure

```
monism-agent/
├── main.py              # CLI entry point
├── util/moe.py         # Multi-ego system configuration
├── sample/fetch_hist.py # Example implementation
├── requirements.txt    # Dependencies
└── pyproject.toml      # Project configuration
```

## Disclaimer

All analysis and investment suggestions are for informational purposes only and do not constitute financial advice. Always conduct thorough research before making investment decisions.

## License

MIT License - free to use, modify, and distribute. See LICENSE file for details.