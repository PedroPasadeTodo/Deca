"""
Bot de backtesting para Bitcoin (BTC/USD) en Coinbase.
Estrategia: cruce de medias moviles (SMA) con filtro RSI.

Requisitos (instalar en tu propio entorno con internet):
    pip install ccxt pandas numpy matplotlib

USO:
    python btc_bot.py

Esto es solo BACKTESTING sobre datos historicos. No ejecuta ordenes reales.
No es asesoramiento financiero: prueba, ajusta y entiende los riesgos antes
de operar con dinero real.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURACION
# ---------------------------------------------------------------------------
EXCHANGE_ID = "coinbase"      # exchange de ccxt
SYMBOL = "BTC/USD"
TIMEFRAME = "1d"              # 1d, 4h, 1h, etc.
SINCE_DAYS = 730               # cuantos dias hacia atras descargar

SMA_FAST = 20
SMA_SLOW = 50
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70           # no comprar si el RSI supera esto
RSI_OVERSOLD = 30             # no vender en corto (no usado, solo long) si esta muy sobrevendido

INITIAL_CAPITAL = 10_000.0
FEE_PCT = 0.006                # ~0.6% por operacion (fee tipico de Coinbase, ajustalo)


# ---------------------------------------------------------------------------
# 1. DESCARGA DE DATOS
# ---------------------------------------------------------------------------
def fetch_ohlcv_ccxt(exchange_id, symbol, timeframe, since_days):
    """Descarga velas historicas usando ccxt. Requiere conexion a internet."""
    import ccxt
    import time

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    since = exchange.milliseconds() - since_days * 24 * 60 * 60 * 1000

    all_ohlcv = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=300)
        if not batch:
            break
        all_ohlcv += batch
        since = batch[-1][0] + 1
        if len(batch) < 300:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def fetch_synthetic_data(n=730, seed=42):
    """Genera datos sinteticos (random walk) solo para probar que el codigo
    funciona quando no hay conexion a internet disponible."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0006, scale=0.03, size=n)
    price = 20000 * np.exp(np.cumsum(returns))
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="D")
    df = pd.DataFrame({"close": price}, index=dates)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    df["volume"] = rng.uniform(100, 1000, size=n)
    return df


# ---------------------------------------------------------------------------
# 2. INDICADORES
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def add_indicators(df):
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(SMA_SLOW).mean()
    df["rsi"] = compute_rsi(df["close"], RSI_PERIOD)
    return df


# ---------------------------------------------------------------------------
# 3. LOGICA DE LA ESTRATEGIA (solo posiciones largas: comprar / vender, sin apalancamiento)
# ---------------------------------------------------------------------------
def generate_signals(df):
    df = df.copy()
    df["signal"] = 0  # 1 = estar en posicion (comprado), 0 = fuera del mercado

    bullish_cross = (df["sma_fast"] > df["sma_slow"]) & (df["rsi"] < RSI_OVERBOUGHT)
    df.loc[bullish_cross, "signal"] = 1

    bearish_cross = df["sma_fast"] < df["sma_slow"]
    df.loc[bearish_cross, "signal"] = 0

    # La posicion deseada cada dia es directamente la señal (1 = dentro, 0 = fuera)
    df["position"] = df["signal"]
    return df


# ---------------------------------------------------------------------------
# 4. BACKTEST
# ---------------------------------------------------------------------------
def run_backtest(df, initial_capital=INITIAL_CAPITAL, fee_pct=FEE_PCT):
    df = df.copy()
    df["market_return"] = df["close"].pct_change().fillna(0)

    # La posicion de HOY se decide con datos de AYER (evita look-ahead bias)
    df["position_shifted"] = df["position"].shift(1).fillna(0)

    df["trade"] = df["position_shifted"].diff().fillna(0) != 0
    df["strategy_return"] = df["position_shifted"] * df["market_return"]
    df.loc[df["trade"], "strategy_return"] -= fee_pct  # coste por cambiar de posicion

    df["equity_strategy"] = initial_capital * (1 + df["strategy_return"]).cumprod()
    df["equity_buyhold"] = initial_capital * (1 + df["market_return"]).cumprod()

    return df


def compute_metrics(df, periods_per_year=365):
    strat = df["strategy_return"]
    equity = df["equity_strategy"]

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_periods = len(df)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()

    sharpe = 0.0
    if strat.std() > 0:
        sharpe = (strat.mean() / strat.std()) * np.sqrt(periods_per_year)

    trades = df[df["trade"]]
    n_trades = len(trades)

    return {
        "Retorno total estrategia": f"{total_return:.2%}",
        "Retorno anualizado (aprox)": f"{annualized_return:.2%}",
        "Drawdown maximo": f"{max_drawdown:.2%}",
        "Sharpe ratio (aprox)": f"{sharpe:.2f}",
        "Numero de operaciones": n_trades,
        "Retorno comprar-y-mantener": f"{(df['equity_buyhold'].iloc[-1] / df['equity_buyhold'].iloc[0] - 1):.2%}",
    }


# ---------------------------------------------------------------------------
# 5. GRAFICO
# ---------------------------------------------------------------------------
def plot_results(df, output_path="equity_curve.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df.index, df["equity_strategy"], label="Estrategia (SMA + RSI)", linewidth=1.8)
    ax.plot(df.index, df["equity_buyhold"], label="Comprar y mantener (Buy & Hold)", linewidth=1.2, alpha=0.7)
    ax.set_title("Backtest: Estrategia SMA+RSI vs Buy & Hold - BTC/USD")
    ax.set_ylabel("Capital ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Grafico guardado en: {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(use_real_data=True):
    print("Descargando/generando datos...")
    if use_real_data:
        try:
            df = fetch_ohlcv_ccxt(EXCHANGE_ID, SYMBOL, TIMEFRAME, SINCE_DAYS)
        except Exception as e:
            print(f"No se pudo descargar de {EXCHANGE_ID} ({e}). Usando datos sinteticos de prueba.")
            df = fetch_synthetic_data(SINCE_DAYS)
    else:
        df = fetch_synthetic_data(SINCE_DAYS)

    print(f"Datos cargados: {len(df)} velas, desde {df.index[0]} hasta {df.index[-1]}")

    df = add_indicators(df)
    df = generate_signals(df)
    df = run_backtest(df)

    metrics = compute_metrics(df)
    print("\n=== RESULTADOS DEL BACKTEST ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    plot_results(df)
    df.to_csv("backtest_resultado.csv")
    print("\nDetalle completo guardado en: backtest_resultado.csv")


if __name__ == "__main__":
    # Cambia a use_real_data=True cuando tengas ccxt instalado y conexion a internet.
    main(use_real_data=True)
