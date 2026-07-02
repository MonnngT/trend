# Gold Trend Monitor

一个用于实时跟踪黄金走势的 Streamlit 小程序，支持：

- XAU/USD、COMEX 黄金期货、GLD ETF 行情跟踪
- K 线图、SMA、布林带、RSI、MACD、ATR 技术指标
- 自动刷新和手动刷新
- 可选 DeepSeek AI 分析按钮，基于最新行情输出趋势和交易计划

> 本项目仅用于行情研究和学习，不构成投资建议。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 配置 DeepSeek API

不要把真实 API key 写进代码或提交到 GitHub。推荐使用环境变量：

```powershell
$env:DEEPSEEK_API_KEY="your_deepseek_api_key"
streamlit run app.py
```

也可以在 Streamlit Cloud 或本地 `.streamlit/secrets.toml` 中配置：

```toml
DEEPSEEK_API_KEY = "your_deepseek_api_key"
```

当前代码使用 DeepSeek 官方 OpenAI 兼容接口：

- `base_url`: `https://api.deepseek.com`
- 默认模型: `deepseek-v4-pro`
- 可选模型: `deepseek-v4-flash`

## 说明

行情数据来自 Yahoo Finance，分钟级数据可能延迟或缺失。交易前请核对券商、交易所和宏观事件数据。
