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

- get_historic_stock_data: A-share historical market data
- yahoo_finance_news: Financial news and updates
- wikipedia: General knowledge research
- llm-math: Mathematical calculations
- terminal: System command execution

## Notifications

The system includes Bark notification integration for analysis completion alerts:

![Bark Notification](https://s2.loli.net/2025/08/21/6v7KNBHJDG3UCRo.jpg)

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