import io
import os
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh


SYMBOLS = {
    "COMEX gold futures": {"yahoo": "GC=F", "stooq": "gc.f"},
    "XAU/USD spot": {"yahoo": "XAUUSD=X", "stooq": "xauusd"},
    "SPDR Gold Shares ETF": {"yahoo": "GLD", "stooq": "gld.us"},
}

PERIOD_INTERVALS = {
    "1d": "1m",
    "5d": "5m",
    "1mo": "30m",
    "3mo": "1h",
    "6mo": "1h",
    "1y": "1d",
    "2y": "1d",
    "5y": "1wk",
}

TROY_OUNCE_GRAMS = 31.1034768


st.set_page_config(
    page_title="Gold Trend Monitor",
    page_icon="G",
    layout="wide",
)


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close"]
    for column in required:
        if column not in data.columns:
            raise ValueError(f"Missing {column} column")

    if "Volume" not in data.columns:
        data["Volume"] = 0

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=["Close"]).sort_index()


def load_yahoo_chart(symbol: str, period: str, interval: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    response = requests.get(
        url,
        params={"range": period, "interval": interval, "includePrePost": "false"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error"):
        raise RuntimeError(chart["error"])

    results = chart.get("result") or []
    if not results:
        raise RuntimeError("Yahoo returned no chart result")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    if not timestamps or not quotes:
        raise RuntimeError("Yahoo returned no price rows")

    index = pd.to_datetime(timestamps, unit="s", utc=True)
    timezone = result.get("meta", {}).get("exchangeTimezoneName")
    if timezone:
        try:
            index = index.tz_convert(timezone)
        except Exception:
            pass
    index = index.tz_localize(None)

    data = pd.DataFrame(
        {
            "Open": quotes.get("open"),
            "High": quotes.get("high"),
            "Low": quotes.get("low"),
            "Close": quotes.get("close"),
            "Volume": quotes.get("volume"),
        },
        index=index,
    )
    return normalize_ohlcv(data)


def stooq_cutoff(period: str) -> pd.Timestamp:
    days = {
        "1d": 60,
        "5d": 90,
        "1mo": 180,
        "3mo": 240,
        "6mo": 365,
        "1y": 540,
        "2y": 900,
        "5y": 2000,
    }.get(period, 365)
    return pd.Timestamp.today().normalize() - pd.Timedelta(days=days)


def load_stooq_daily(symbol: str, period: str) -> pd.DataFrame:
    response = requests.get(
        "https://stooq.com/q/d/l/",
        params={"s": symbol, "i": "d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    if not response.text.strip() or "No data" in response.text[:80]:
        raise RuntimeError("Stooq returned no data")

    data = pd.read_csv(io.StringIO(response.text))
    if data.empty:
        raise RuntimeError("Stooq returned an empty CSV")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date")
    data = data[data.index >= stooq_cutoff(period)]
    return normalize_ohlcv(data)


@st.cache_data(ttl=300, show_spinner=False)
def load_usdcny_rate() -> tuple[float | None, str, str]:
    try:
        data = load_yahoo_chart("CNY=X", "5d", "1d")
        if not data.empty:
            return float(data["Close"].iloc[-1]), "Yahoo Finance (CNY=X)", ""
        return None, "No source", "Yahoo Finance returned 0 rows for CNY=X."
    except Exception as exc:
        return None, "No source", f"USD/CNY rate failed: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def load_market_data(symbol_name: str, period: str, interval: str) -> tuple[pd.DataFrame, str, str]:
    config = SYMBOLS[symbol_name]
    errors: list[str] = []

    try:
        data = load_yahoo_chart(config["yahoo"], period, interval)
        if not data.empty:
            return data, f"Yahoo Finance ({config['yahoo']})", ""
        errors.append("Yahoo Finance returned 0 rows.")
    except Exception as exc:
        errors.append(f"Yahoo Finance failed: {exc}")

    try:
        data = load_stooq_daily(config["stooq"], period)
        if not data.empty:
            return data, f"Stooq daily fallback ({config['stooq']})", "\n".join(errors)
        errors.append("Stooq returned 0 rows.")
    except Exception as exc:
        errors.append(f"Stooq fallback failed: {exc}")

    return pd.DataFrame(), "No source", "\n".join(errors)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    close = df["Close"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA26"] = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI14"] = rsi(close)

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    df["BB_UPPER"] = middle + 2 * std
    df["BB_LOWER"] = middle - 2 * std

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = true_range.rolling(14).mean()
    return df


def latest_snapshot(df: pd.DataFrame) -> dict:
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    change = latest["Close"] - previous["Close"]
    change_pct = change / previous["Close"] * 100 if previous["Close"] else 0

    trend = "震荡"
    if latest["Close"] > latest.get("SMA20", np.nan) > latest.get("SMA50", np.nan):
        trend = "偏多"
    elif latest["Close"] < latest.get("SMA20", np.nan) < latest.get("SMA50", np.nan):
        trend = "偏空"

    rsi_value = latest.get("RSI14", np.nan)
    momentum = "中性"
    if pd.notna(rsi_value) and rsi_value >= 70:
        momentum = "超买"
    elif pd.notna(rsi_value) and rsi_value <= 30:
        momentum = "超卖"

    return {
        "time": df.index[-1],
        "price": float(latest["Close"]),
        "change": float(change),
        "change_pct": float(change_pct),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "volume": float(latest.get("Volume", 0) or 0),
        "sma20": float(latest.get("SMA20", np.nan)),
        "sma50": float(latest.get("SMA50", np.nan)),
        "rsi14": float(rsi_value),
        "macd": float(latest.get("MACD", np.nan)),
        "macd_signal": float(latest.get("MACD_SIGNAL", np.nan)),
        "atr14": float(latest.get("ATR14", np.nan)),
        "trend": trend,
        "momentum": momentum,
    }


def build_chart(df: pd.DataFrame, symbol_name: str) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.62, 0.2, 0.18],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#16825d",
            decreasing_line_color="#b23a48",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(color="#f0b429", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(color="#2f80ed", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"], name="BB upper", line=dict(color="#8a8f98", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"], name="BB lower", line=dict(color="#8a8f98", width=1, dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", line=dict(color="#7c3aed", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#b23a48", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#16825d", row=2, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["MACD"] - df["MACD_SIGNAL"], name="MACD hist", marker_color="#8792a2"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#111827", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"], name="Signal", line=dict(color="#dc6803", width=1.4)), row=3, col=1)

    fig.update_layout(
        title=f"{symbol_name} trend",
        height=720,
        margin=dict(l=20, r=20, t=56, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def compact_market_table(df: pd.DataFrame, rows: int = 60) -> str:
    columns = ["Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "RSI14", "MACD", "MACD_SIGNAL", "ATR14"]
    table = df[columns].tail(rows).copy()
    table.index = table.index.astype(str)
    return table.round(4).to_csv()


def run_deepseek_analysis(
    api_key: str,
    model: str,
    symbol_name: str,
    period: str,
    interval: str,
    snapshot: dict,
    market_csv: str,
    thinking_enabled: bool,
) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    system_prompt = (
        "你是一个谨慎的黄金市场分析助手。请基于用户给出的行情和技术指标分析，"
        "给出趋势、关键支撑阻力、风险点和交易计划。必须强调这不是投资建议，"
        "不要承诺收益，不要编造不存在的数据。"
    )
    user_prompt = f"""
品种: {symbol_name}
周期: {period}
K线间隔: {interval}
最新快照: {snapshot}

最近行情与指标 CSV:
{market_csv}

请用中文输出：
1. 当前趋势判断
2. 多空关键价位
3. 可能的交易思路，包含入场触发、止损、止盈和仓位风险
4. 需要警惕的数据质量或宏观事件风险
"""

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1500,
        "temperature": 0.2,
        "stream": False,
    }
    if thinking_enabled:
        kwargs["reasoning_effort"] = "high"
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = client.chat.completions.create(**kwargs)
    if not response.choices:
        raise RuntimeError("DeepSeek returned no choices.")

    message = response.choices[0].message
    content = message.content or ""
    if not content.strip():
        extra = getattr(message, "model_extra", {}) or {}
        content = extra.get("reasoning_content", "") or extra.get("reasoning", "")
    if not content.strip():
        raise RuntimeError("DeepSeek returned an empty response. Try deepseek-v4-pro with Thinking mode off.")
    return content


def format_number(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def converted_price_text(symbol_name: str, price_usd: float, usdcny: float | None) -> tuple[str, str]:
    if not usdcny:
        return "-", "USD/CNY unavailable"

    cny_price = price_usd * usdcny
    if symbol_name == "SPDR Gold Shares ETF":
        return f"¥{cny_price:,.2f}", "CNY/share"

    cny_per_gram = cny_price / TROY_OUNCE_GRAMS
    return f"¥{cny_per_gram:,.2f}", "CNY/g"


def main() -> None:
    st.title("Gold Trend Monitor")

    with st.sidebar:
        st.header("Market")
        symbol_name = st.selectbox("Symbol", list(SYMBOLS.keys()))
        period = st.selectbox("Period", list(PERIOD_INTERVALS.keys()), index=2)
        interval = PERIOD_INTERVALS[period]
        st.caption(f"Interval: {interval}")

        auto_refresh = st.toggle("Auto refresh", value=True)
        refresh_seconds = st.slider("Refresh seconds", 30, 600, 60, step=30)
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.header("DeepSeek AI")
        ai_enabled = st.toggle("Enable AI analysis", value=False)
        model = st.selectbox("Model", ["deepseek-v4-pro", "deepseek-v4-flash"], index=0)
        thinking_enabled = st.toggle("Thinking mode", value=False)

    if auto_refresh:
        st_autorefresh(interval=refresh_seconds * 1000, key="gold_market_refresh")

    data, data_source, data_errors = load_market_data(symbol_name, period, interval)
    if data.empty:
        st.error("No market data returned. Try another symbol, period, or interval.")
        if data_errors:
            st.code(data_errors, language="text")
        st.info(
            "If this keeps happening, your network may be blocking the market data provider. "
            "COMEX gold futures and GLD usually have the best availability."
        )
        return

    df = add_indicators(data)
    snapshot = latest_snapshot(df)
    usdcny_rate, usdcny_source, usdcny_error = load_usdcny_rate()
    cny_text, cny_unit = converted_price_text(symbol_name, snapshot["price"], usdcny_rate)
    st.caption(f"Data source: {data_source}")
    if data_errors:
        with st.expander("Fallback details", expanded=False):
            st.code(data_errors, language="text")

    metric_cols = st.columns(6)
    metric_cols[0].metric("USD price", format_number(snapshot["price"]), f'{snapshot["change"]:+.2f} ({snapshot["change_pct"]:+.2f}%)')
    metric_cols[1].metric(cny_unit, cny_text)
    metric_cols[2].metric("USD/CNY", format_number(usdcny_rate, 4) if usdcny_rate else "-")
    metric_cols[3].metric("Trend", snapshot["trend"])
    metric_cols[4].metric("RSI14", format_number(snapshot["rsi14"]))
    metric_cols[5].metric("Updated", pd.Timestamp(snapshot["time"]).strftime("%Y-%m-%d %H:%M"))
    if usdcny_rate:
        st.caption(f"FX source: {usdcny_source}. CNY/g uses 1 troy ounce = {TROY_OUNCE_GRAMS:.4f} g.")
    elif usdcny_error:
        st.warning(usdcny_error)

    st.plotly_chart(build_chart(df, symbol_name), use_container_width=True)

    with st.expander("Latest data", expanded=False):
        st.dataframe(df.tail(100).round(4), use_container_width=True)

    if ai_enabled:
        api_key = get_secret("DEEPSEEK_API_KEY")
        if not api_key:
            st.warning("Set DEEPSEEK_API_KEY in your environment or Streamlit secrets before running AI analysis.")
        elif st.button("Run DeepSeek AI analysis", type="primary", use_container_width=True):
            with st.spinner("DeepSeek is analyzing the latest gold data..."):
                try:
                    analysis_snapshot = {
                        **snapshot,
                        "usd_cny": usdcny_rate,
                        "cny_price": cny_text,
                        "cny_unit": cny_unit,
                        "data_source": data_source,
                    }
                    analysis = run_deepseek_analysis(
                        api_key=api_key,
                        model=model,
                        symbol_name=symbol_name,
                        period=period,
                        interval=interval,
                        snapshot=analysis_snapshot,
                        market_csv=compact_market_table(df),
                        thinking_enabled=thinking_enabled,
                    )
                except Exception as exc:
                    st.session_state["last_ai_error"] = str(exc)
                    st.session_state.pop("last_ai_analysis", None)
                else:
                    st.session_state["last_ai_analysis"] = analysis
                    st.session_state.pop("last_ai_error", None)

        if st.session_state.get("last_ai_error"):
            st.error(f"DeepSeek analysis failed: {st.session_state['last_ai_error']}")
        if st.session_state.get("last_ai_analysis"):
            st.subheader("AI analysis")
            st.write(st.session_state["last_ai_analysis"])

    st.caption(
        "Market data may be delayed and can contain gaps. This tool is for research only and is not financial advice."
    )


if __name__ == "__main__":
    main()
