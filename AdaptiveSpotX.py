import logging
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

#############################################################################################################
##  AdaptiveSpotX — Multi-Regime Adaptive Spot Strategy                                                    ##
##  Inspired by NostalgiaForInfinity (https://github.com/iterativv/NostalgiaForInfinity)                   ##
##                                                                                                         ##
##  DESIGN PHILOSOPHY                                                                                      ##
##  -----------------                                                                                      ##
##  No spot strategy can profit in every market condition — in a sustained bear market all longs lose.     ##
##  This strategy instead aims to:                                                                         ##
##    • Catch meaningful moves in bull and sideways markets with three distinct entry modes.                ##
##    • Block most entries in confirmed bear regimes using a macro BTC filter.                             ##
##    • Reduce drawdowns through signal-based exits rather than hard stops.                                ##
##    • Recover stuck trades with controlled DCA (max 2 rebuys).                                          ##
##                                                                                                         ##
##  THREE ENTRY MODES                                                                                      ##
##  -----------------                                                                                      ##
##  1. "pullback"      — Trend pullback in a bullish/sideways 4h context.                                  ##
##  2. "mean_rev"      — Deep oversold bounce; works in sideways and early bear recoveries.                ##
##  3. "bull_breakout" — High-volume momentum breakout in a confirmed bull regime.                         ##
##                                                                                                         ##
##  RECOMMENDED SETUP                                                                                      ##
##  -----------------                                                                                      ##
##  • timeframe: 5m                                                                                        ##
##  • max_open_trades: 6-10                                                                                ##
##  • stake_currency: USDT or USDC                                                                         ##
##  • pairlist: VolumePairList (40-80 pairs), AgeFilter >= 30 days                                         ##
##  • Blacklist: leveraged tokens (*BULL, *BEAR, *UP, *DOWN)                                               ##
##  • use_exit_signal: true                                                                                 ##
##  • exit_profit_only: false                                                                               ##
##  • ignore_roi_if_entry_signal: true                                                                     ##
#############################################################################################################


class AdaptiveSpotX(IStrategy):
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v1.0.0"

    # Hard stoploss — last resort safety net; strategy prefers signal-based exits
    stoploss = -0.15

    trailing_stop = False
    use_custom_stoploss = False

    timeframe = "5m"
    info_timeframes = ["1h", "4h"]

    process_only_new_candles = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # 480 x 5m = 40h of base data; 4h EMA200 needs data from the informative layer
    startup_candle_count: int = 480

    # DCA — up to two additional buys per trade
    position_adjustment_enable = True

    # Tiered ROI; signal-based exits override this when a valid entry signal is still active
    minimal_roi = {
        "0": 0.10,      # anytime if 10% is reached
        "240": 0.05,    # 4 h
        "720": 0.03,    # 12 h
        "1440": 0.015,  # 24 h
    }

    # =========================================================================
    # INFORMATIVE PAIRS
    # =========================================================================

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative = []
        for pair in pairs:
            for tf in self.info_timeframes:
                informative.append((pair, tf))

        stake = self.config.get("stake_currency", "USDT")
        stable = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"
        informative.append((btc_pair, "4h"))

        return informative

    # =========================================================================
    # INDICATOR HELPERS (one per timeframe)
    # =========================================================================

    def _indicators_4h(self, df: DataFrame) -> DataFrame:
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["ema_200"] = ta.EMA(df, timeperiod=200)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["adx"] = ta.ADX(df, timeperiod=14)

        # Market regime: 1 = bull, 0 = sideways, -1 = bear
        bull = (df["ema_20"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]) & (df["rsi"] > 50)
        bear = (df["ema_20"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]) & (df["rsi"] < 50)
        df["regime"] = np.where(bull, 1, np.where(bear, -1, 0))

        return df

    def _indicators_1h(self, df: DataFrame) -> DataFrame:
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["willr"] = ta.WILLR(df, timeperiod=14)

        bb = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"] = bb["upperband"]
        df["bb_lower"] = bb["lowerband"]

        df["cmf"] = pta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)

        return df

    def _btc_indicators_4h(self, df: DataFrame) -> DataFrame:
        """Compute indicators on BTC 4h then rename all columns with btc_ prefix."""
        df["ema_20"] = ta.EMA(df, timeperiod=20)
        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        # 3-candle (12h) momentum
        df["mom"] = df["close"].pct_change(3)

        # Prefix every column except date
        df.rename(columns=lambda c: f"btc_{c}" if c != "date" else c, inplace=True)
        return df

    # =========================================================================
    # POPULATE INDICATORS
    # =========================================================================

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        stake = self.config.get("stake_currency", "USDT")
        stable = {"USDT", "USDC", "BUSD", "DAI", "FDUSD", "USD"}
        btc_pair = f"BTC/{stake}" if stake in stable else "BTC/USDT"

        # --- 4h informative (current pair) ---
        info_4h = self.dp.get_pair_dataframe(metadata["pair"], "4h")
        if not info_4h.empty:
            info_4h = self._indicators_4h(info_4h)
            df = merge_informative_pair(df, info_4h, self.timeframe, "4h", ffill=True)
            if "date_4h" in df.columns:
                df.drop(columns=["date_4h"], inplace=True)

        # --- 1h informative (current pair) ---
        info_1h = self.dp.get_pair_dataframe(metadata["pair"], "1h")
        if not info_1h.empty:
            info_1h = self._indicators_1h(info_1h)
            df = merge_informative_pair(df, info_1h, self.timeframe, "1h", ffill=True)
            if "date_1h" in df.columns:
                df.drop(columns=["date_1h"], inplace=True)

        # --- BTC 4h (macro market filter) ---
        if metadata["pair"] != btc_pair:
            btc_4h = self.dp.get_pair_dataframe(btc_pair, "4h")
            if not btc_4h.empty:
                btc_4h = self._btc_indicators_4h(btc_4h)
                # After btc_ prefix: columns are btc_open, btc_close, btc_ema_20 …
                # merge_informative_pair adds _4h suffix → btc_open_4h, btc_ema_20_4h …
                df = merge_informative_pair(df, btc_4h, self.timeframe, "4h", ffill=True)
                if "date_4h" in df.columns:
                    df.drop(columns=["date_4h"], inplace=True)

        # --- 5m base indicators ---
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["ema_8"] = ta.EMA(df, timeperiod=8)
        df["ema_21"] = ta.EMA(df, timeperiod=21)
        df["volume_ma"] = ta.SMA(df["volume"], timeperiod=20)

        bb5 = ta.BBANDS(df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df["bb_upper"] = bb5["upperband"]
        df["bb_lower"] = bb5["lowerband"]

        # CTI — measures how strongly price is trending linearly (range -1..1)
        df["cti"] = pta.cti(df["close"], length=20)

        return df

    # =========================================================================
    # ENTRY SIGNALS
    # =========================================================================

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0
        df["enter_tag"] = ""

        # ── Macro gate: block entries when BTC is in confirmed downtrend ──────
        # btc_ema_20_4h and btc_ema_50_4h come from BTC pair merge
        has_btc = "btc_ema_20_4h" in df.columns
        if has_btc:
            btc_ok = (df["btc_ema_20_4h"] >= df["btc_ema_50_4h"] * 0.97) | (df["btc_rsi_4h"] > 45)
        else:
            btc_ok = pd.Series(True, index=df.index)

        # ── Condition 1 · Trend Pullback ──────────────────────────────────────
        # Buy the dip inside a 4h uptrend: price eases back toward EMA support,
        # RSI cools off, and volume shows the pullback is not panic selling.
        cond1 = (
            btc_ok
            & (df.get("regime_4h", pd.Series(0, index=df.index)) >= 0)
            & (df.get("rsi_4h", pd.Series(50, index=df.index)) < 65)
            & (df.get("rsi_1h", pd.Series(50, index=df.index)).between(30, 55))
            & (df.get("cmf_1h", pd.Series(0.0, index=df.index)) > -0.15)
            & (df["close"] > df.get("ema_50_1h", df["close"]) * 0.985)
            & (df["rsi"] < 45)
            & (df["volume"] > df["volume_ma"] * 1.1)
            & (df["cti"] < 0.5)   # not already in a strong 5m upswing
        )

        # ── Condition 2 · Mean-Reversion Bounce ──────────────────────────────
        # Deep oversold on both 1h and 5m with price at lower Bollinger Band.
        # Works in sideways markets and at capitulation lows in bear markets.
        cond2 = (
            btc_ok
            & (df.get("rsi_1h", pd.Series(50, index=df.index)) < 35)
            & (df["close"] <= df.get("bb_lower_1h", df["close"]) * 1.02)
            & (df.get("cmf_1h", pd.Series(0.0, index=df.index)) > -0.35)
            & (df["rsi"] < 32)
            & (df["close"] <= df["bb_lower"] * 1.01)
            & (~cond1)
        )

        # ── Condition 3 · Bull Breakout ───────────────────────────────────────
        # High-volume momentum entry; only in a confirmed 4h bull regime.
        cond3 = (
            btc_ok
            & (df.get("regime_4h", pd.Series(0, index=df.index)) == 1)
            & (df.get("rsi_4h", pd.Series(50, index=df.index)).between(50, 65))
            & (df["volume"] > df["volume_ma"] * 2.0)
            & (df["rsi"].between(52, 72))
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
        df["exit_tag"] = ""

        # Exit 1: Overbought on 1h while price touches the upper Bollinger Band
        overbought = (
            (df.get("rsi_1h", pd.Series(50, index=df.index)) > 70)
            & (df["close"] >= df.get("bb_upper_1h", df["close"]) * 0.985)
        )

        # Exit 2: 1h EMA bearish crossover — trend is turning
        ema_cross_down = (
            (df.get("ema_20_1h", df["close"]) < df.get("ema_50_1h", df["close"]))
            & (df.get("ema_20_1h", df["close"]).shift(1) >= df.get("ema_50_1h", df["close"]).shift(1))
            & (df.get("rsi_1h", pd.Series(50, index=df.index)) < 50)
        )

        df.loc[overbought, ["exit_long", "exit_tag"]] = [1, "overbought"]
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
        """
        Two DCA levels — 25% of original stake each time.
        Only fires once per level; a third buy never happens.
        """
        buys = trade.nr_of_successful_entries

        if buys >= 3:
            return None

        # First rebuy: price dropped 8% from initial entry
        if buys == 1 and current_profit < -0.08:
            return trade.stake_amount * 0.25

        # Second rebuy: price dropped 14% (evaluated on the current average cost)
        if buys == 2 and current_profit < -0.14:
            return trade.stake_amount * 0.25

        return None
