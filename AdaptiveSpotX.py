import logging
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    merge_informative_pair,
    IntParameter,
    DecimalParameter,
    CategoricalParameter,
)
from pandas import DataFrame
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

#############################################################################################################
##  AdaptiveSpotX — Multi-Regime Adaptive Spot Strategy                                                    ##
##  Inspired by NostalgiaForInfinity (https://github.com/iterativv/NostalgiaForInfinity)                   ##
##                                                                                                         ##
##  HYPEROPT QUICK START                                                                                    ##
##  -------------------                                                                                    ##
##  Buy-space only (fastest):                                                                               ##
##    freqtrade hyperopt --strategy AdaptiveSpotX --hyperopt-loss SharpeHyperOptLoss \                     ##
##      --spaces buy --timerange 20220101-20240101 -e 500                                                  ##
##                                                                                                         ##
##  Full space (buy + sell + roi + stoploss):                                                               ##
##    freqtrade hyperopt --strategy AdaptiveSpotX --hyperopt-loss CalmarHyperOptLoss \                     ##
##      --spaces buy sell roi stoploss --timerange 20220101-20240101 -e 1000                               ##
##                                                                                                         ##
##  RECOMMENDED SETUP                                                                                      ##
##  -----------------                                                                                      ##
##  • max_open_trades: 3      • stake_amount: 20 (total per trade ≤ 30 with DCA)                           ##
##  • stake_currency: USDT/USDC                                                                            ##
##  • pairlist: VolumePairList 40-80 pairs, AgeFilter >= 30 days                                           ##
##  • Blacklist: *BULL *BEAR *UP *DOWN leveraged tokens                                                    ##
##  • use_exit_signal: true  •  exit_profit_only: false                                                    ##
##  • ignore_roi_if_entry_signal: true                                                                     ##
#############################################################################################################


class AdaptiveSpotX(IStrategy):
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v2.0.0"

    # =========================================================================
    # HYPEROPT DEFAULT VALUES
    # Applied when running live/backtest without a hyperopt result file.
    # Freqtrade overwrites these automatically after a successful hyperopt run.
    # =========================================================================

    buy_params = {
        # BTC macro filter
        "btc_filter_rsi_min":   45,
        "btc_filter_ema_ratio": 0.97,
        # Condition 1 — pullback
        "cond1_enabled":        True,
        "cond1_rsi_1h_min":     30,
        "cond1_rsi_1h_max":     55,
        "cond1_rsi_5m_max":     45,
        "cond1_cmf_1h_min":    -0.15,
        "cond1_volume_mult":    1.1,
        "cond1_cti_max":        0.5,
        # Condition 2 — mean-reversion
        "cond2_enabled":        True,
        "cond2_rsi_1h_max":     35,
        "cond2_rsi_5m_max":     32,
        "cond2_cmf_1h_min":    -0.35,
        # Condition 3 — bull breakout
        "cond3_enabled":        True,
        "cond3_rsi_4h_min":     50,
        "cond3_rsi_4h_max":     65,
        "cond3_rsi_5m_min":     52,
        "cond3_rsi_5m_max":     72,
        "cond3_volume_mult":    2.0,
        # DCA — trigger ottimizzabili, importi fissi (25 e 30 USDC)
        "dca1_trigger": -0.08,
        "dca2_trigger": -0.14,
    }

    sell_params = {
        "exit_rsi_1h_overbought": 70,
        "exit_bb_proximity":      0.985,
    }

    # =========================================================================
    # HYPEROPT PARAMETERS
    # =========================================================================

    # --- BTC macro filter ---
    btc_filter_rsi_min   = IntParameter(35, 55, default=45,   space="buy",  optimize=True, load=True)
    btc_filter_ema_ratio = DecimalParameter(0.93, 1.00, default=0.97, decimals=2, space="buy", optimize=True, load=True)

    # --- Condition 1: Trend Pullback ---
    cond1_enabled    = CategoricalParameter([True, False], default=True, space="buy", optimize=True, load=True)
    cond1_rsi_1h_min = IntParameter(20, 42, default=30, space="buy", optimize=True, load=True)
    cond1_rsi_1h_max = IntParameter(44, 65, default=55, space="buy", optimize=True, load=True)
    cond1_rsi_5m_max = IntParameter(35, 55, default=45, space="buy", optimize=True, load=True)
    cond1_cmf_1h_min = DecimalParameter(-0.30,  0.00, default=-0.15, decimals=2, space="buy", optimize=True, load=True)
    cond1_volume_mult = DecimalParameter(1.0,   2.0,  default=1.1,   decimals=1, space="buy", optimize=True, load=True)
    cond1_cti_max     = DecimalParameter(0.3,   0.8,  default=0.5,   decimals=1, space="buy", optimize=True, load=True)

    # --- Condition 2: Mean-Reversion Bounce ---
    cond2_enabled    = CategoricalParameter([True, False], default=True, space="buy", optimize=True, load=True)
    cond2_rsi_1h_max = IntParameter(25, 45, default=35, space="buy", optimize=True, load=True)
    cond2_rsi_5m_max = IntParameter(22, 40, default=32, space="buy", optimize=True, load=True)
    cond2_cmf_1h_min = DecimalParameter(-0.50, -0.10, default=-0.35, decimals=2, space="buy", optimize=True, load=True)

    # --- Condition 3: Bull Breakout ---
    cond3_enabled     = CategoricalParameter([True, False], default=True, space="buy", optimize=True, load=True)
    cond3_rsi_4h_min  = IntParameter(45, 60, default=50, space="buy", optimize=True, load=True)
    cond3_rsi_4h_max  = IntParameter(55, 75, default=65, space="buy", optimize=True, load=True)
    cond3_rsi_5m_min  = IntParameter(45, 62, default=52, space="buy", optimize=True, load=True)
    cond3_rsi_5m_max  = IntParameter(65, 80, default=72, space="buy", optimize=True, load=True)
    cond3_volume_mult = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space="buy", optimize=True, load=True)

    # --- Exit ---
    exit_rsi_1h_overbought = IntParameter(65, 82,   default=70,    space="sell", optimize=True, load=True)
    exit_bb_proximity      = DecimalParameter(0.970, 1.000, default=0.985, decimals=3, space="sell", optimize=True, load=True)

    # --- DCA ---
    # I trigger sono ottimizzabili; gli importi sono fissi (25 e 30 USDC)
    dca1_trigger = DecimalParameter(-0.15, -0.05, default=-0.08, decimals=2, space="buy", optimize=True, load=True)
    dca2_trigger = DecimalParameter(-0.25, -0.10, default=-0.14, decimals=2, space="buy", optimize=True, load=True)

    # Importi fissi per ogni DCA — in USDC (non percentuale)
    # Entry: 20 USDC (stake_amount nel config) + DCA1: 25 + DCA2: 30 = 75 USDC max
    dca1_amount: float = 25.0
    dca2_amount: float = 30.0

    # =========================================================================
    # FIXED STRATEGY SETTINGS
    # =========================================================================

    stoploss = -0.15

    trailing_stop = False
    use_custom_stoploss = False

    timeframe = "5m"
    info_timeframes = ["1h", "4h"]

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    startup_candle_count: int = 480

    position_adjustment_enable = True

    minimal_roi = {
        "0":    0.10,
        "240":  0.05,
        "720":  0.03,
        "1440": 0.015,
    }

    # =========================================================================
    # INFORMATIVE PAIRS
    # =========================================================================

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = [(pair, tf) for pair in pairs for tf in self.info_timeframes]

        stake = self.config.get("stake_currency", "USDT")
        stable = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"
        informative.append((btc_pair, "4h"))

        return informative

    # =========================================================================
    # INDICATOR HELPERS
    # =========================================================================

    def _indicators_4h(self, df: DataFrame) -> DataFrame:
        df["ema_20"]  = ta.EMA(df, timeperiod=20)
        df["ema_50"]  = ta.EMA(df, timeperiod=50)
        df["ema_200"] = ta.EMA(df, timeperiod=200)
        df["rsi"]     = ta.RSI(df, timeperiod=14)
        df["adx"]     = ta.ADX(df, timeperiod=14)

        bull = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]) & (df["rsi"] > 50)
        bear = (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]) & (df["rsi"] < 50)
        df["regime"] = np.where(bull, 1, np.where(bear, -1, 0))

        return df

    def _indicators_1h(self, df: DataFrame) -> DataFrame:
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["rsi"]    = ta.RSI(df, timeperiod=14)
        df["willr"]  = ta.WILLR(df, timeperiod=14)

        bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"] = bb["upperband"]
        df["bb_lower"] = bb["lowerband"]

        df["cmf"]       = pta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)

        return df

    def _btc_indicators_4h(self, df: DataFrame) -> DataFrame:
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["rsi"]    = ta.RSI(df, timeperiod=14)
        df.rename(columns=lambda c: f"btc_{c}" if c != "date" else c, inplace=True)
        return df

    # =========================================================================
    # POPULATE INDICATORS
    # =========================================================================

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        stake  = self.config.get("stake_currency", "USDT")
        stable = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"

        # 4h informative (current pair)
        info_4h = self.dp.get_pair_dataframe(metadata["pair"], "4h")
        if not info_4h.empty:
            info_4h = self._indicators_4h(info_4h)
            df = merge_informative_pair(df, info_4h, self.timeframe, "4h", ffill=True)
            df.drop(columns=[c for c in df.columns if c == "date_4h"], inplace=True)

        # 1h informative (current pair)
        info_1h = self.dp.get_pair_dataframe(metadata["pair"], "1h")
        if not info_1h.empty:
            info_1h = self._indicators_1h(info_1h)
            df = merge_informative_pair(df, info_1h, self.timeframe, "1h", ffill=True)
            df.drop(columns=[c for c in df.columns if c == "date_1h"], inplace=True)

        # BTC 4h macro filter
        if metadata["pair"] != btc_pair:
            btc_4h = self.dp.get_pair_dataframe(btc_pair, "4h")
            if not btc_4h.empty:
                btc_4h = self._btc_indicators_4h(btc_4h)
                df = merge_informative_pair(df, btc_4h, self.timeframe, "4h", ffill=True)
                df.drop(columns=[c for c in df.columns if c == "date_4h"], inplace=True)

        # 5m base indicators (periods fixed — only thresholds are hyperoptimized)
        df["rsi"]       = ta.RSI(df, timeperiod=14)
        df["ema_8"]     = ta.EMA(df, timeperiod=8)
        df["ema_21"]    = ta.EMA(df, timeperiod=21)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)

        bb5 = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"] = bb5["upperband"]
        df["bb_lower"] = bb5["lowerband"]

        df["cti"] = pta.cti(df["close"], length=20)

        return df

    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"]  = ""

        # Helpers to read informative columns safely
        def col(name, default):
            return df[name] if name in df.columns else pd.Series(default, index=df.index)

        # BTC macro gate
        btc_ok = (
            col("btc_ema_20_4h", df["close"]) >= col("btc_ema_50_4h", df["close"]) * self.btc_filter_ema_ratio.value
        ) | (col("btc_rsi_4h", 50) > self.btc_filter_rsi_min.value)

        # ── Condition 1: Trend Pullback ───────────────────────────────────────
        cond1 = (
            self.cond1_enabled.value
            & btc_ok
            & (col("regime_4h", 0) >= 0)
            & (col("rsi_4h", 50)   < 65)
            & col("rsi_1h", 50).between(self.cond1_rsi_1h_min.value, self.cond1_rsi_1h_max.value)
            & (col("cmf_1h", 0.0)  > self.cond1_cmf_1h_min.value)
            & (df["close"]         > col("ema_50_1h", df["close"]) * 0.985)
            & (df["rsi"]           < self.cond1_rsi_5m_max.value)
            & (df["volume"]        > df["volume_ma"] * self.cond1_volume_mult.value)
            & (df["cti"]           < self.cond1_cti_max.value)
        )

        # ── Condition 2: Mean-Reversion Bounce ───────────────────────────────
        cond2 = (
            self.cond2_enabled.value
            & btc_ok
            & (col("rsi_1h", 50)  < self.cond2_rsi_1h_max.value)
            & (df["close"]        <= col("bb_lower_1h", df["close"]) * 1.02)
            & (col("cmf_1h", 0.0) > self.cond2_cmf_1h_min.value)
            & (df["rsi"]          < self.cond2_rsi_5m_max.value)
            & (df["close"]        <= df["bb_lower"] * 1.01)
            & (~cond1)
        )

        # ── Condition 3: Bull Breakout ────────────────────────────────────────
        cond3 = (
            self.cond3_enabled.value
            & btc_ok
            & (col("regime_4h", 0) == 1)
            & col("rsi_4h", 50).between(self.cond3_rsi_4h_min.value, self.cond3_rsi_4h_max.value)
            & (df["volume"] > df["volume_ma"] * self.cond3_volume_mult.value)
            & df["rsi"].between(self.cond3_rsi_5m_min.value, self.cond3_rsi_5m_max.value)
            & (df["close"] > df["ema_8"])
            & (df["ema_8"] > df["ema_21"])
            & (~cond1)
            & (~cond2)
        )

        df.loc[cond1, ["enter_long", "enter_tag"]] = [1, "pullback"]
        df.loc[cond2, ["enter_long", "enter_tag"]] = [1, "mean_rev"]
        df.loc[cond3, ["enter_long", "enter_tag"]] = [1, "bull_breakout"]

        return df

    # =========================================================================
    # EXIT SIGNALS
    # =========================================================================

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        df["exit_tag"]  = ""

        def col(name, default):
            return df[name] if name in df.columns else pd.Series(default, index=df.index)

        overbought = (
            (col("rsi_1h", 50) > self.exit_rsi_1h_overbought.value)
            & (df["close"] >= col("bb_upper_1h", df["close"]) * self.exit_bb_proximity.value)
        )

        ema_cross_down = (
            (col("ema_20_1h", df["close"])         < col("ema_50_1h", df["close"]))
            & (col("ema_20_1h", df["close"]).shift(1) >= col("ema_50_1h", df["close"]).shift(1))
            & (col("rsi_1h", 50) < 50)
        )

        df.loc[overbought,                  ["exit_long", "exit_tag"]] = [1, "overbought"]
        df.loc[ema_cross_down & ~overbought, ["exit_long", "exit_tag"]] = [1, "ema_cross_down"]

        return df

    # =========================================================================
    # POSITION ADJUSTMENT (DCA)
    # =========================================================================

    def adjust_trade_position(
        self,
        trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        buys = trade.nr_of_successful_entries

        if buys >= 3:
            return None

        if buys == 1 and current_profit < self.dca1_trigger.value:
            return self.dca1_amount   # 25 USDC fissi

        if buys == 2 and current_profit < self.dca2_trigger.value:
            return self.dca2_amount   # 30 USDC fissi

        return None
