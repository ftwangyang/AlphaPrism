"""AlphaPrism: Category-Guided LLM Alpha Mining with Dual-Branch Causal Combination.

Usage:
  python alphaprism.py eval --market all --pool_dir pools --qlib_path /path/to/qlib_data/cn_data
  python alphaprism.py eval --market all --pool_dir pools --qlib_path /path/to/qlib_data/cn_data --ablations --sensitivity
  python alphaprism.py mine --market csi300 --seeds 42 --qlib_path /path/to/qlib_data/cn_data --api_url <YOUR_API_URL> --model Qwen3-32B-AWQ --out_dir pools
  python alphaprism.py run --market csi300 --seeds 42 --qlib_path /path/to/qlib_data/cn_data --api_url <YOUR_API_URL> --model Qwen3-32B-AWQ
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
import time
import warnings
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import torch
from torch import Tensor


# Qlib market data loader

class FeatureType(IntEnum):
    OPEN = 0
    CLOSE = 1
    HIGH = 2
    LOW = 3
    VOLUME = 4
    VWAP = 5


class _SequentialParallel:
    """Sequential executor for Qlib data loading."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, iterable):
        return [fn(*a, **kw) for fn, a, kw in iterable]


def load_instruments(qlib_path: str, market: str) -> List[str]:
    inst_file = os.path.join(qlib_path, "instruments", f"{market.lower()}.txt")
    if not os.path.exists(inst_file):
        raise FileNotFoundError(f"Instrument list not found: {inst_file}")
    codes: List[str] = []
    with open(inst_file, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0].startswith(("SH", "SZ")):
                codes.append(parts[0])
    return sorted(set(codes))


class StockData:
    """Loads OHLCV + VWAP panel from Qlib into a GPU/CPU tensor [T_padded, F, S]."""

    _qlib_initialized: bool = False
    _qlib_current_path: str = ""

    def __init__(
        self,
        instrument: Union[str, List[str]],
        start_time: str,
        end_time: str,
        qlib_path: str,
        max_backtrack_days: int = 100,
        max_future_days: int = 30,
        features: Optional[List[FeatureType]] = None,
        device: torch.device = torch.device("cpu"),
        freq: str = "day",
        filter_stocks: bool = True,
        stock_quality_threshold: float = 0.5,
        smart_fill: bool = True,
    ) -> None:
        qlib_path = os.path.expanduser(qlib_path)
        if not os.path.exists(qlib_path):
            raise FileNotFoundError(
                f"Qlib data directory does not exist: {qlib_path}. "
                "Please download Qlib cn_data and pass --qlib_path."
            )

        self._ensure_metadata(qlib_path)
        self._init_qlib(qlib_path)

        self._instrument = instrument
        self._start_time = start_time
        self._end_time = end_time
        self.max_backtrack_days = max_backtrack_days
        self.max_future_days = max_future_days
        self._features = features if features is not None else list(FeatureType)
        self.device = device
        self.freq = freq
        self.filter_stocks = filter_stocks
        self.stock_quality_threshold = stock_quality_threshold
        self.smart_fill = smart_fill

        self.data, self._dates, self._stock_ids = self._get_data()

    @property
    def n_days(self) -> int:
        return int(self.data.shape[0] - self.max_backtrack_days - self.max_future_days)

    @property
    def n_stocks(self) -> int:
        return int(self.data.shape[2])

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.to_datetime(
            list(self._dates)[self.max_backtrack_days : len(self._dates) - self.max_future_days]
        )

    @staticmethod
    def _ensure_metadata(qlib_path: str) -> None:
        meta_dir = os.path.join(qlib_path, "features", ".meta")
        os.makedirs(meta_dir, exist_ok=True)
        meta_file = os.path.join(meta_dir, "dir.txt")
        if not os.path.exists(meta_file):
            with open(meta_file, "w", encoding="utf-8") as f:
                f.write("day\n")

    @classmethod
    def _init_qlib(cls, qlib_path: str) -> None:
        import qlib
        import qlib.data.data as qdd
        import qlib.utils.paral as qp
        from qlib.config import REG_CN, REG_US

        qp.ParallelExt = _SequentialParallel
        qdd.ParallelExt = _SequentialParallel

        if cls._qlib_initialized and cls._qlib_current_path == qlib_path:
            return
        region = REG_US if "us" in qlib_path.lower() else REG_CN
        if cls._qlib_current_path:
            qlib.init(provider_uri=qlib_path, region=region, override=True)
        else:
            qlib.init(provider_uri=qlib_path, region=region)
        cls._qlib_initialized = True
        cls._qlib_current_path = qlib_path

    def _load_exprs(self, exprs: List[str]) -> pd.DataFrame:
        from qlib.data import D
        from qlib.data.dataset.loader import QlibDataLoader

        cal: np.ndarray = D.calendar(freq=self.freq)
        start_idx = int(cal.searchsorted(pd.Timestamp(self._start_time)))
        end_idx = int(cal.searchsorted(pd.Timestamp(self._end_time)))
        if start_idx >= len(cal):
            raise ValueError(f"Start time {self._start_time} exceeds Qlib calendar range.")
        if end_idx >= len(cal):
            end_idx = len(cal) - 1

        real_start = cal[max(0, start_idx - self.max_backtrack_days)]
        real_end = cal[min(len(cal) - 1, end_idx + self.max_future_days)]
        return QlibDataLoader(config=exprs, freq=self.freq).load(self._instrument, real_start, real_end)

    @staticmethod
    def _smart_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
        df_filled = df.ffill(limit=5).bfill(limit=2)
        df_filled = df_filled.interpolate(method="linear", limit=3, limit_area="inside")
        return df_filled.fillna(0)

    def _filter_stocks_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.filter_stocks:
            return df
        stocks = df.index.get_level_values(1).unique()
        valid_stocks = []
        for stock in stocks:
            s_df = df.xs(stock, level=1)
            qual = float(s_df.notna().sum().sum()) / max(s_df.size, 1)
            if qual >= self.stock_quality_threshold:
                valid_stocks.append(stock)
        if not valid_stocks:
            return df
        return df.loc[(slice(None), valid_stocks), :]

    def _filter_stocks_tensor(
        self, tensor: Tensor, stock_ids: pd.Index
    ) -> Tuple[Tensor, pd.Index]:
        if not self.filter_stocks:
            return tensor, stock_ids
        valid_ratio = ((tensor != 0) & ~torch.isnan(tensor)).float().mean(dim=(0, 1))
        mask = valid_ratio >= self.stock_quality_threshold
        if int(mask.sum().item()) == 0:
            return tensor, stock_ids
        return tensor[:, :, mask], stock_ids[mask.cpu().numpy()]

    def _get_data(self) -> Tuple[Tensor, pd.Index, pd.Index]:
        features = ["$" + f.name.lower() for f in self._features]
        df = self._load_exprs(features)
        df = self._filter_stocks_df(df)
        df = self._smart_fill_missing(df) if self.smart_fill else df.fillna(0)

        df_unstacked = df.stack().unstack(level=1)
        if df_unstacked.isnull().sum().sum() > 0:
            df_unstacked = (
                self._smart_fill_missing(df_unstacked) if self.smart_fill else df_unstacked.fillna(0)
            )

        dates = (
            df_unstacked.index.levels[0]
            if isinstance(df_unstacked.index, pd.MultiIndex)
            else df_unstacked.index.unique()
        )
        stock_ids = df_unstacked.columns
        values = df_unstacked.values
        n_features = len(features)
        n_dates = values.shape[0] // n_features
        values = values.reshape((n_dates, n_features, len(stock_ids)))
        if len(dates) != n_dates:
            dates = dates[:n_dates]

        tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        tensor, stock_ids = self._filter_stocks_tensor(tensor, stock_ids)
        return tensor, dates, stock_ids


# Prefix expression DSL, parser, AST validator, and evaluator

class Expression(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor: ...

    def __repr__(self) -> str:
        return str(self)


class Feature(Expression):
    def __init__(self, feature: FeatureType) -> None:
        self._feature = feature

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        if period.start < -data.max_backtrack_days or period.stop - 1 > data.max_future_days:
            raise IndexError("Expression evaluation slice out of padded StockData range.")
        start = period.start + data.max_backtrack_days
        stop = period.stop + data.max_backtrack_days + data.n_days - 1
        return data.data[start:stop, int(self._feature), :]

    def __str__(self) -> str:
        return "$" + self._feature.name.lower()


class Constant(Expression):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        days = period.stop - period.start - 1 + data.n_days
        return torch.full(
            (days, data.n_stocks), self.value, dtype=data.data.dtype, device=data.data.device
        )

    def __str__(self) -> str:
        return str(self.value)


class UnaryOperator(Expression):
    def __init__(self, operand: Union[Expression, float]) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._operand.evaluate(data, period))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand})"


class BinaryOperator(Expression):
    def __init__(self, lhs: Union[Expression, float], rhs: Union[Expression, float]) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._lhs.evaluate(data, period), self._rhs.evaluate(data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs})"


class RollingOperator(Expression):
    def __init__(self, operand: Union[Expression, float], delta_time: int) -> None:
        self._operand = operand if isinstance(operand, Expression) else Constant(operand)
        self._delta_time = int(delta_time)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        values = self._operand.evaluate(data, slice(period.start - self._delta_time + 1, period.stop))
        return self._apply(values.unfold(0, self._delta_time, 1))

    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._operand},{self._delta_time}d)"


class PairRollingOperator(Expression):
    def __init__(
        self,
        lhs: Union[Expression, float],
        rhs: Union[Expression, float],
        delta_time: int,
    ) -> None:
        self._lhs = lhs if isinstance(lhs, Expression) else Constant(lhs)
        self._rhs = rhs if isinstance(rhs, Expression) else Constant(rhs)
        self._delta_time = int(delta_time)

    def _unfold(self, expr: Expression, data: StockData, period: slice) -> Tensor:
        values = expr.evaluate(data, slice(period.start - self._delta_time + 1, period.stop))
        return values.unfold(0, self._delta_time, 1)

    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._apply(self._unfold(self._lhs, data, period), self._unfold(self._rhs, data, period))

    @abstractmethod
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor: ...

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._lhs},{self._rhs},{self._delta_time}d)"


# 22 operators

class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.abs()


class Log(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.log()


class Add(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs + rhs


class Sub(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs - rhs


class Mul(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs * rhs


class Div(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return lhs / rhs


class Greater(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.maximum(lhs, rhs)


class Less(BinaryOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        return torch.minimum(lhs, rhs)


class Ref(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        return self._operand.evaluate(
            data, slice(period.start - self._delta_time, period.stop - self._delta_time)
        )

    def _apply(self, operand: Tensor) -> Tensor:
        return operand


class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)


class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sum(dim=-1)


class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.std(dim=-1)


class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.var(dim=-1)


class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]


class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]


class Med(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.median(dim=-1)[0]


class Mad(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return (operand - operand.mean(dim=-1, keepdim=True)).abs().mean(dim=-1)


class Delta(RollingOperator):
    def evaluate(self, data: StockData, period: slice = slice(0, 1)) -> Tensor:
        values = self._operand.evaluate(data, slice(period.start - self._delta_time, period.stop))
        return values[self._delta_time :] - values[: -self._delta_time]

    def _apply(self, operand: Tensor) -> Tensor:
        return operand


class WMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        weights = torch.arange(n, dtype=operand.dtype, device=operand.device)
        weights = weights / weights.sum()
        return (weights * operand).sum(dim=-1)


class EMA(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        n = operand.shape[-1]
        alpha = 1.0 - 2.0 / (1.0 + n)
        power = torch.arange(n, 0, -1, dtype=operand.dtype, device=operand.device)
        weights = alpha ** power
        weights = weights / weights.sum()
        return (weights * operand).sum(dim=-1)


class Cov(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        n = lhs.shape[-1]
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        return (clhs * crhs).sum(dim=-1) / (n - 1)


class Corr(PairRollingOperator):
    def _apply(self, lhs: Tensor, rhs: Tensor) -> Tensor:
        clhs = lhs - lhs.mean(dim=-1, keepdim=True)
        crhs = rhs - rhs.mean(dim=-1, keepdim=True)
        ncov = (clhs * crhs).sum(dim=-1)
        nlvar = (clhs ** 2).sum(dim=-1)
        nrvar = (crhs ** 2).sum(dim=-1)
        stdmul = (nlvar * nrvar).sqrt()
        stdmul[(nlvar < 1e-6) | (nrvar < 1e-6)] = 1.0
        return ncov / stdmul


ALLOWED_OPS: Tuple[Type[Expression], ...] = (
    Abs, Log,
    Add, Sub, Mul, Div, Greater, Less,
    Ref, Mean, Sum, Std, Var, Max, Min, Med, Mad, Delta, WMA, EMA,
    Cov, Corr,
)

OP_REGISTRY: Dict[str, Type[Expression]] = {cls.__name__.lower(): cls for cls in ALLOWED_OPS}
for _alias, _cls in [
    ("tsmean", Mean), ("tssum", Sum), ("tsstd", Std), ("tsvar", Var),
    ("tsmax", Max), ("tsmin", Min), ("tsmed", Med), ("tsmad", Mad),
    ("tsdelta", Delta), ("tswma", WMA), ("tsema", EMA), ("tscov", Cov), ("tscorr", Corr),
]:
    OP_REGISTRY[_alias] = _cls

FEAT_MAP: Dict[str, FeatureType] = {
    "$open": FeatureType.OPEN,
    "$close": FeatureType.CLOSE,
    "$high": FeatureType.HIGH,
    "$low": FeatureType.LOW,
    "$volume": FeatureType.VOLUME,
    "$vwap": FeatureType.VWAP,
}

_TOKEN_RE = re.compile(
    r"\s*(?:(\$[a-zA-Z_]+)|([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?d?)|([a-zA-Z_][a-zA-Z0-9_]*)|([(),]))"
)


def parse_expr(text: str) -> Expression:
    """Parse a prefix-notation formula string into an Expression AST."""
    tokens: List[Tuple[str, str]] = []
    pos = 0
    s = text.strip()
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(f"Unexpected syntax near: {s[pos:pos + 20]!r}")
        if m.group(1):
            tokens.append(("FEAT", m.group(1).lower()))
        elif m.group(2):
            tokens.append(("NUM", m.group(2)))
        elif m.group(3):
            tokens.append(("IDENT", m.group(3).lower()))
        else:
            tokens.append(("PUNCT", m.group(4)))
        pos = m.end()

    idx = [0]

    def parse_node() -> Expression:
        if idx[0] >= len(tokens):
            raise ValueError("Unexpected end of expression.")
        kind, val = tokens[idx[0]]
        idx[0] += 1
        if kind == "FEAT":
            if val not in FEAT_MAP:
                raise ValueError(f"Unsupported feature: {val}")
            return Feature(FEAT_MAP[val])
        if kind == "NUM":
            if val.endswith("d"):
                raise ValueError(f"Window token {val} outside rolling argument.")
            return Constant(float(val))
        if kind == "IDENT":
            if val not in OP_REGISTRY:
                raise ValueError(f"Disallowed or unknown operator: {val}")
            op_cls = OP_REGISTRY[val]
            if idx[0] >= len(tokens) or tokens[idx[0]] != ("PUNCT", "("):
                raise ValueError(f"Expected '(' after operator {val}.")
            idx[0] += 1
            args: List[Union[Expression, int]] = []
            if issubclass(op_cls, UnaryOperator):
                args.append(parse_node())
            elif issubclass(op_cls, BinaryOperator):
                args.append(parse_node())
                _expect(",")
                args.append(parse_node())
            elif issubclass(op_cls, RollingOperator):
                args.append(parse_node())
                _expect(",")
                args.append(parse_window())
            elif issubclass(op_cls, PairRollingOperator):
                args.append(parse_node())
                _expect(",")
                args.append(parse_node())
                _expect(",")
                args.append(parse_window())
            _expect(")")
            return op_cls(*args)  # type: ignore[arg-type]
        raise ValueError(f"Unexpected token: {kind}={val}")

    def parse_window() -> int:
        if idx[0] >= len(tokens) or tokens[idx[0]][0] != "NUM":
            raise ValueError("Expected integer window argument.")
        raw = tokens[idx[0]][1]
        idx[0] += 1
        raw = raw[:-1] if raw.endswith("d") else raw
        val = float(raw)
        if not val.is_integer():
            raise ValueError(f"Rolling window must be an integer, got {raw}")
        return int(val)

    def _expect(ch: str) -> None:
        if idx[0] >= len(tokens) or tokens[idx[0]] != ("PUNCT", ch):
            raise ValueError(f"Expected '{ch}'.")
        idx[0] += 1

    node = parse_node()
    if idx[0] != len(tokens):
        raise ValueError(f"Trailing tokens after expression: {tokens[idx[0]:]}")
    return node


def validate_expr_ast(
    expr: Expression, max_window: int = 40, max_depth: int = 12, max_nodes: int = 35
) -> Tuple[bool, str]:
    """Verify allowed 22-op grammar, windows in [1, max_window], finite constants, and bounded AST."""
    nodes = [0]
    has_feat = [False]

    def walk(node: Expression, depth: int) -> Optional[str]:
        nodes[0] += 1
        if depth > max_depth:
            return "depth"
        if nodes[0] > max_nodes:
            return "nodes"
        if isinstance(node, Feature):
            has_feat[0] = True
            return None
        if isinstance(node, Constant):
            return None if np.isfinite(node.value) else "const_inf"
        if not isinstance(node, ALLOWED_OPS):
            return f"disallowed_{type(node).__name__}"
        if isinstance(node, UnaryOperator):
            return walk(node._operand, depth + 1)
        if isinstance(node, BinaryOperator):
            return walk(node._lhs, depth + 1) or walk(node._rhs, depth + 1)
        if isinstance(node, RollingOperator):
            if node._delta_time < 1 or node._delta_time > max_window:
                return f"window_{node._delta_time}"
            return walk(node._operand, depth + 1)
        if isinstance(node, PairRollingOperator):
            if node._delta_time < 1 or node._delta_time > max_window:
                return f"window_{node._delta_time}"
            return walk(node._lhs, depth + 1) or walk(node._rhs, depth + 1)
        return "unknown"

    err = walk(expr, 1)
    if err is not None:
        return False, err
    if not has_feat[0]:
        return False, "no_feature"
    return True, "ok"


# Cross-sectional normalization and evaluation metrics

def fast_rank_data(x: Tensor, nan_mask: Tensor) -> Tensor:
    """Compute cross-sectional mid-ranks along the stock axis."""
    x_clean = torch.where(nan_mask, torch.full_like(x, float("inf")), x)
    _, idx_asc = torch.sort(x_clean, dim=1, stable=True)
    r_asc = torch.empty_like(x_clean)
    arange = torch.arange(x.shape[1], device=x.device, dtype=x.dtype).expand_as(x)
    r_asc.scatter_(1, idx_asc, arange)
    x_flip = torch.flip(x_clean, [1])
    _, idx_flip = torch.sort(x_flip, dim=1, stable=True)
    r_flip = torch.empty_like(x_clean)
    r_flip.scatter_(1, idx_flip, arange)
    r_desc = torch.flip(r_flip, [1])
    rank = 0.5 * (r_asc + r_desc)
    rank[nan_mask] = 0.0
    return rank


def rankz_2d(x: Tensor) -> Tensor:
    """Cross-sectional rank normalization RankZ(x) -> zero mean, unit std (0 on invalid entries)."""
    nan_mask = ~torch.isfinite(x)
    r = fast_rank_data(x, nan_mask)
    valid = (~nan_mask).float()
    n = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (r * valid).sum(dim=1, keepdim=True) / n
    diff = (r - mean) * valid
    std = ((diff * diff).sum(dim=1, keepdim=True) / n).sqrt()
    z = torch.where(std > 1e-6, diff / std.clamp(min=1e-6), torch.zeros_like(diff))
    z[nan_mask] = 0.0
    return z


def cs_rank_zscore_nan(x: Tensor, min_valid: int = 10) -> Tensor:
    """Mid-rank cross-sectional z-score preserving NaNs on invalid entries (used during Stage 1 mining)."""
    x = torch.where(torch.isfinite(x), x, torch.full_like(x, float("nan")))
    mask = torch.isfinite(x)
    d, s = x.shape
    src = torch.arange(s, device=x.device, dtype=x.dtype).unsqueeze(0).expand(d, s)

    order_asc = torch.argsort(torch.where(mask, x, torch.full_like(x, float("inf"))), dim=1, stable=True)
    ranks_asc = torch.empty_like(x).scatter_(1, order_asc, src)

    order_desc = torch.argsort(torch.where(mask, -x, torch.full_like(x, float("inf"))), dim=1, stable=True)
    ranks_desc = torch.empty_like(x).scatter_(1, order_desc, src)

    n_valid = mask.sum(dim=1, keepdim=True).to(x.dtype)
    mid = torch.where(mask, 0.5 * (ranks_asc + (n_valid - 1.0 - ranks_desc)), torch.full_like(x, float("nan")))
    mean = torch.where(mask, mid, torch.zeros_like(mid)).sum(dim=1, keepdim=True) / n_valid.clamp_min(1.0)
    diff = torch.where(mask, mid - mean, torch.zeros_like(mid))
    std = torch.sqrt((diff * diff).sum(dim=1, keepdim=True) / n_valid.clamp_min(1.0))
    ok_day = (n_valid >= min_valid) & (std > 1e-6)
    z = (mid - mean) / std.clamp_min(1e-6)
    return torch.where(mask & ok_day, z, torch.full_like(z, float("nan")))


def clipz_2d(x: Tensor, clip: float = 4.0) -> Tensor:
    """Cross-sectional standardization, clipping to [-clip, clip], and restandardization."""
    nan_mask = ~torch.isfinite(x)
    valid = (~nan_mask).float()
    x0 = torch.where(nan_mask, torch.zeros_like(x), x)
    n = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (x0 * valid).sum(dim=1, keepdim=True) / n
    diff = (x0 - mean) * valid
    std = ((diff * diff).sum(dim=1, keepdim=True) / n).sqrt()
    z = torch.where(std > 1e-6, diff / std.clamp(min=1e-6), torch.zeros_like(diff))
    z = z.clamp(-clip, clip) * valid
    mean2 = (z * valid).sum(dim=1, keepdim=True) / n
    diff2 = (z - mean2) * valid
    std2 = ((diff2 * diff2).sum(dim=1, keepdim=True) / n).sqrt()
    z2 = torch.where(std2 > 1e-6, diff2 / std2.clamp(min=1e-6), torch.zeros_like(diff2))
    z2[nan_mask] = 0.0
    return z2


def normalize_by_day_safe(x: Tensor) -> Tensor:
    x = torch.where(torch.isfinite(x), x, torch.full_like(x, float("nan")))
    mask = torch.isnan(x)
    n = (~mask).sum(dim=1).clamp_min(1).to(x.dtype)
    xc = torch.where(mask, torch.zeros_like(x), x)
    mean = xc.sum(dim=1) / n
    std = ((((xc - mean[:, None]) * (~mask)) ** 2).sum(dim=1) / n).sqrt()
    out = (x - mean[:, None]) / std[:, None]
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def batch_pearsonr(x: Tensor, y: Tensor) -> Tensor:
    nan_mask = x.isnan() | y.isnan()
    valid = (~nan_mask).float()
    n = valid.sum(dim=1)
    xc = torch.where(nan_mask, torch.zeros_like(x), x)
    yc = torch.where(nan_mask, torch.zeros_like(y), y)
    xm = xc.sum(dim=1) / n.clamp(min=1.0)
    ym = yc.sum(dim=1) / n.clamp(min=1.0)
    x_std = (((xc - xm[:, None]) * valid) ** 2).sum(dim=1).div(n.clamp(min=1.0)).sqrt()
    y_std = (((yc - ym[:, None]) * valid) ** 2).sum(dim=1).div(n.clamp(min=1.0)).sqrt()
    cov = (xc * yc).sum(dim=1) / n.clamp(min=1.0) - xm * ym
    stdmul = x_std * y_std
    stdmul[(x_std < 1e-3) | (y_std < 1e-3)] = 1.0
    res = cov / stdmul
    res[res.isnan()] = 0.0
    return res


def batch_spearmanr(x: Tensor, y: Tensor) -> Tensor:
    nan_mask = x.isnan() | y.isnan()
    rx = fast_rank_data(x, nan_mask)
    ry = fast_rank_data(y, nan_mask)
    rx = torch.where(nan_mask, torch.full_like(rx, float("nan")), rx)
    ry = torch.where(nan_mask, torch.full_like(ry, float("nan")), ry)
    return batch_pearsonr(rx, ry)


def daily_pearson_masked(x: Tensor, y: Tensor) -> Tensor:
    mask = torch.isfinite(x) & torch.isfinite(y)
    n = mask.sum(dim=1).to(x.dtype)
    xc = torch.where(mask, x, torch.zeros_like(x))
    yc = torch.where(mask, y, torch.zeros_like(y))
    xm = xc.sum(dim=1, keepdim=True) / n.clamp_min(1.0).unsqueeze(1)
    ym = yc.sum(dim=1, keepdim=True) / n.clamp_min(1.0).unsqueeze(1)
    dx = torch.where(mask, x - xm, torch.zeros_like(x))
    dy = torch.where(mask, y - ym, torch.zeros_like(y))
    cov = (dx * dy).sum(dim=1) / n.clamp_min(1.0)
    sx = torch.sqrt((dx * dx).sum(dim=1) / n.clamp_min(1.0))
    sy = torch.sqrt((dy * dy).sum(dim=1) / n.clamp_min(1.0))
    ok = (n >= 10) & (sx > 1e-4) & (sy > 1e-4)
    out = torch.where(ok, cov / (sx * sy).clamp_min(1e-8), torch.zeros_like(cov))
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_metrics(
    pred: Tensor, tgt: Tensor, label_days: int = 20, test_years: float = 3.0, risk_free_rate: float = 0.0
) -> Tuple[Dict[str, float], Tensor]:
    """Compute IC, ICIR, RankIC, RankICIR, AR, MDD, and SR on top-20% daily-equivalent returns."""
    if pred.dim() > 2:
        pred = pred.squeeze(-1)
    if tgt.dim() > 2:
        tgt = tgt.squeeze(-1)

    ic_s = torch.nan_to_num(batch_pearsonr(pred, tgt), nan=0.0)
    ric_chunks = [batch_spearmanr(pred[i : i + 400], tgt[i : i + 400]) for i in range(0, len(pred), 400)]
    ric_s = torch.nan_to_num(torch.cat(ric_chunks, dim=0), nan=0.0)

    nan_mask = pred.isnan() | tgt.isnan()
    ret_raw = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    for d in range(pred.shape[0]):
        vm = ~nan_mask[d]
        vc = int(vm.sum().item())
        if vc == 0:
            continue
        top_k = max(1, int(vc * 0.2))
        _, top_idx = torch.topk(pred[d][vm], top_k, largest=True)
        ret_raw[d] = tgt[d][vm][top_idx].mean()
    ret_s = torch.nan_to_num(ret_raw, nan=0.0) / float(label_days)

    ic_m = ic_s.mean().item()
    ic_sd = ic_s.std().item() if ic_s.std().item() > 1e-6 else 1.0
    ric_m = ric_s.mean().item()
    ric_sd = ric_s.std().item() if ric_s.std().item() > 1e-6 else 1.0
    ret_m = ret_s.mean().item()
    ret_sd = ret_s.std().item() if ret_s.std().item() > 1e-6 else 1.0

    excess = ret_s - (risk_free_rate / 252.0)
    sr = 0.0 if excess.std().item() < 1e-8 else (excess.mean() / excess.std() * math.sqrt(252.0)).item()

    cum = 1.0 + ret_s.cumsum(dim=0)
    run_max = torch.cummax(cum, dim=0)[0]
    mdd = float(-((cum - run_max) / run_max).min().item())

    return {
        "ic": ic_m,
        "ic_std": ic_sd,
        "icir": ic_m / ic_sd,
        "ric": ric_m,
        "ric_std": ric_sd,
        "ricir": ric_m / ric_sd,
        "ret": ret_m * len(ret_s) / test_years,
        "ret_std": ret_sd,
        "retir": ret_m / ret_sd,
        "ret_sharpe": sr,
        "ret_mdd": mdd,
    }, ret_s


# Stage 1: category-guided residual LLM alpha mining

BAND_NAMES: List[str] = [
    "reversal_trend",
    "volatility_dispersion",
    "liquidity_scale",
    "price_volume_coupling",
    "intraday_cross",
]

DSL_DOC = """STRICT PREFIX FUNCTION-CALL SYNTAX (DO NOT use infix +, -, *, /):
Allowed 22 operators ONLY:
- Unary: Abs(x), Log(x)
- Binary: Add(x, y), Sub(x, y), Mul(x, y), Div(x, y), Greater(x, y), Less(x, y)
- Rolling (integer window d MUST be in 1..40): Ref(x, d), Mean(x, d), Sum(x, d), Std(x, d), Var(x, d), Max(x, d), Min(x, d), Med(x, d), Mad(x, d), Delta(x, d), WMA(x, d), EMA(x, d)
- Rolling pair (integer window d MUST be in 1..40): Cov(x, y, d), Corr(x, y, d)
Allowed features ONLY: $open, $high, $low, $close, $volume, $vwap
Numeric constants like 0.01, 0.5, 1, 2, 5, 10, -1 are allowed.
FORBIDDEN operators (DO NOT USE): CSRank, Rank, Skew, Kurt, Sign, Pow, Square, Sqrt.
FORBIDDEN windows: any window > 40 or < 1.
SMOOTHNESS RULE: NEVER put raw $volume, Var($volume,d), Std($volume,d), or Mad($volume,d) directly in a denominator Div(..., raw_vol) because raw volume spans 5 orders of magnitude and creates extreme outliers. Always wrap volume or dollar volume in Log(Add(..., 1.0)) or use dimensionless ratios like Div($volume, Add(Mean($volume, d), 1.0))."""

BAND_SPECS: List[Tuple[str, str]] = [
    (
        "reversal_trend",
        """Category 1: PRICE REVERSAL & TREND STRUCTURE
Focus on cross-sectional return reversal, log-price / moving-average deviation (Mean/EMA/WMA/Med), rolling high-low channel position, and multi-horizon return spread over windows 5, 10, 20, 40.
Examples:
Div(Sub($close, Mean($close, 20)), Add(Std($close, 20), 0.01))
Div(Sub($close, Min($low, 20)), Add(Sub(Max($high, 20), Min($low, 20)), 0.01))
Log(Div($close, EMA($close, 20)))""",
    ),
    (
        "volatility_dispersion",
        """Category 2: VOLATILITY & RANGE DISPERSION
Focus on idiosyncratic return volatility (Std/Mad of daily returns or log prices), Parkinson log high-low range Log(Div($high, $low)), short-vs-long volatility ratios, and log-volatility level over windows 5, 10, 20, 40.
Examples:
Log(Add(Std(Div(Sub($close, Ref($close, 1)), Ref($close, 1)), 20), 0.001))
Mean(Log(Div($high, $low)), 20)
Div(Std(Log($close), 5), Add(Std(Log($close), 20), 0.001))""",
    ),
    (
        "liquidity_scale",
        """Category 3: LIQUIDITY, LOG DOLLAR-VOLUME & TURNOVER SCALE
Focus on smooth log dollar-volume Log(Add(Mul($vwap, $volume), 1.0)), rolling mean/median/EMA/WMA/Std/Mad of log dollar-volume or log share volume, log Amihud illiquidity Log(Add(Mean(Div(Abs(Sub($close, Ref($close, 1))), Add(Mul($vwap, $volume), 1.0)), 20), 0.0001)), and relative turnover surges Div(Mean($volume, 5), Add(Mean($volume, 20), 1.0)) over windows 5, 10, 20, 40.
Examples:
Mean(Log(Add(Mul($vwap, $volume), 1.0)), 20)
Std(Log(Add(Mul($vwap, $volume), 1.0)), 20)
Sub(Log(Add(Mul($vwap, $volume), 1.0)), Mean(Log(Add(Mul($vwap, $volume), 1.0)), 20))""",
    ),
    (
        "price_volume_coupling",
        """Category 4: PRICE-VOLUME COUPLING & MICROSTRUCTURE
Focus on rolling Corr and Cov between prices/returns/intraday range and volume/log-dollar-volume/relative volume, plus intraday VWAP execution pressure Div(Sub($vwap, $close), Add(Sub($high, $low), 0.01)) and volume-weighted return signals over windows 5, 10, 20, 40.
Examples:
Corr($close, Log(Add($volume, 1.0)), 20)
Corr(Div(Sub($close, Ref($close, 5)), Ref($close, 5)), Div($volume, Add(Mean($volume, 10), 1.0)), 20)
Mean(Div(Sub($vwap, $close), Add(Sub($high, $low), 0.01)), 10)""",
    ),
    (
        "intraday_cross",
        """Category 5: INTRADAY-OVERNIGHT SPLIT & SMOOTH CROSS-CATEGORY INTERACTIONS
Focus on overnight gap Log(Div($open, Ref($close, 1))), intraday body Log(Div($close, $open)), VWAP-to-high/low ratios Div($vwap, $high), upper/lower candlestick shadow imbalance, and SMOOTH cross-category interactions combining Log dollar-volume, volatility, and intraday returns over windows 5, 10, 20, 40.
Examples:
Mean(Log(Div($open, Ref($close, 1))), 20)
Mean(Div($vwap, $high), 10)
Mul(Mean(Log(Div($high, $low)), 10), Mean(Log(Add(Mul($vwap, $volume), 1.0)), 20))""",
    ),
]

SYSTEM_PROMPT = (
    "You are an expert quantitative finance researcher designing smooth, well-conditioned formulaic alpha factors "
    "for Chinese A-share cross-sectional stock selection (20-day forward return). "
    "Output ONLY raw prefix-notation formulas, one per line. No numbering, no bullets, no markdown, no explanations."
)

PROMPT_R0 = """{dsl}

{band_desc}

Task: Generate {n} diverse, economically meaningful, cross-sectionally SMOOTH formulaic alpha factors strictly within this category.
- Every formula must be syntactically valid prefix notation using ONLY the 22 allowed operators and windows in 1..40.
- Avoid repeating these already-generated formulas:
{avoid}
Output {n} lines, one formula per line, nothing else."""

PROMPT_RESID = """{dsl}

{band_desc}

Closed-loop Residual Refinement (Round {rnd}, training period 2010-2020 only):
We fit a linear combination of our current multi-category alpha pool and computed the UNEXPLAINED RESIDUAL cross-sectional return.
Your goal is to propose {n} NEW cross-sectionally SMOOTH formulas in THIS category that explain the residual return and are DECORRELATED from the existing pool.

Strongest smooth formulas in this category so far (train RankIC, RankICIR, linear IC, yearly consistency, residual RankIC):
{good}

Weak, unstable, or overly skewed formulas to AVOID mimicking:
{bad}

Do NOT repeat any already-tested formula:
{avoid}

Generate {n} novel, smooth, valid prefix formulas (windows in 1..40, ONLY the 22 allowed operators), one per line, nothing else."""


def _parse_llm_lines(text: str) -> List[str]:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    out: List[str] = []
    for raw in text.splitlines():
        s = raw.strip().strip("`").strip()
        s = re.sub(r"^\d+[\.\)\-:]\s*", "", s)
        s = re.sub(r"^[\-\*]\s*", "", s)
        if "#" in s:
            s = s.split("#", 1)[0].strip()
        if not s or "$" not in s or "(" not in s or ")" not in s:
            continue
        if re.search(r"\$[a-z]+\s*[\+\-\*/]\s*\$?[a-z0-9]", s):
            continue
        out.append(s)
    return out


def _ask_llm(
    api_url: str,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    api_key: str = "",
) -> Tuple[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": 0.92,
        "max_tokens": 1100,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_err = ""
    for _ in range(3):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=180)
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or ""), ""
        except Exception as exc:
            last_err = str(exc)
            time.sleep(3)
    return "", last_err


def _factor_train_stats(
    f_raw: Tensor, y_raw: Tensor, y_rz: Tensor, years: np.ndarray
) -> Optional[Dict[str, float]]:
    f_rz = cs_rank_zscore_nan(f_raw)
    valid_days = torch.isfinite(f_rz).any(dim=1).float().mean().item()
    if valid_days < 0.70:
        return None
    f_norm = normalize_by_day_safe(f_raw)
    ric_s = daily_pearson_masked(f_rz, y_rz)
    ic_s = daily_pearson_masked(f_norm, y_raw)

    ric = ric_s.mean().item()
    ricir = ric / max(ric_s.std().item(), 1e-6)
    ic = ic_s.mean().item()
    icir = ic / max(ic_s.std().item(), 1e-6)

    ric_np = ric_s.detach().cpu().numpy()
    sign = 1.0 if ric >= 0 else -1.0
    yr_means = [float(ric_np[years == yr].mean()) for yr in np.unique(years) if (years == yr).sum() >= 20]
    yr_cons = float(np.mean([(m * sign) > 0 for m in yr_means])) if yr_means else 0.0

    n_days = len(ric_np)
    block_irs = []
    for b in range(5):
        seg = ric_np[(b * n_days) // 5 : ((b + 1) * n_days) // 5] * sign
        block_irs.append(float(seg.mean() / max(seg.std(), 1e-6)))
    return {
        "valid_days": float(valid_days),
        "ric": float(ric),
        "ricir": float(ricir),
        "ic": float(ic),
        "icir": float(icir),
        "yr_cons": float(yr_cons),
        "min_block_ricir": float(min(block_irs)),
    }


def _quality_score(rec: Dict[str, float]) -> float:
    ricir, icir = abs(rec["ricir"]), abs(rec["icir"])
    smooth_ir = 0.6 * min(ricir, icir) + 0.4 * (0.5 * ricir + 0.5 * icir)
    return smooth_ir * (0.6 + 0.4 * rec["yr_cons"]) + 0.25 * max(0.0, rec["min_block_ricir"])


def _compute_orthogonal_residual(
    archive: List[Dict],
    fnorm_cache: Dict[str, Tensor],
    frz_cache: Dict[str, Tensor],
    y_rz: Tensor,
    max_pool: int = 15,
    corr_cap: float = 0.68,
) -> Tuple[List[str], Tensor]:
    cands = [
        a for a in archive
        if abs(a["ric"]) >= 0.015 and abs(a["ic"]) >= 0.012
        and abs(a["ricir"]) >= 0.15 and abs(a["icir"]) >= 0.12
        and a["yr_cons"] >= 0.55
    ]
    cands = (
        sorted(archive, key=_quality_score, reverse=True)[:20]
        if len(cands) < 3
        else sorted(cands, key=_quality_score, reverse=True)
    )

    selected: List[str] = []
    sel_rz: List[Tensor] = []
    for a in cands:
        e = a["expr"]
        frz = frz_cache[e]
        if any(abs(daily_pearson_masked(frz, prev).mean().item()) > corr_cap for prev in sel_rz):
            continue
        selected.append(e)
        sel_rz.append(frz)
        if len(selected) >= max_pool:
            break

    if not selected:
        return [], y_rz

    X = torch.stack([fnorm_cache[e] for e in selected], dim=-1)
    d_len, s_len, k_len = X.shape
    Xf = X.reshape(d_len * s_len, k_len)
    yf = y_rz.reshape(d_len * s_len)
    ok = torch.isfinite(yf) & torch.isfinite(Xf).all(dim=-1)
    Xv, yv = Xf[ok], yf[ok]
    lam = 1e-2 * Xv.shape[0]
    w = torch.linalg.solve(Xv.T @ Xv + lam * torch.eye(k_len, device=X.device, dtype=X.dtype), Xv.T @ yv)
    pred = (Xf @ w).reshape(d_len, s_len)
    resid = torch.where(torch.isfinite(y_rz), y_rz - pred, torch.full_like(y_rz, float("nan")))
    return selected, cs_rank_zscore_nan(resid)


def mine_pool(
    market: str,
    seed: int,
    qlib_path: str,
    api_url: str,
    model: str,
    out_dir: str,
    api_key: str = "",
    train_start: str = "2010-01-01",
    train_end: str = "2020-12-31",
    label_days: int = 20,
    temperature: float = 0.85,
    r0_batches: int = 2,
    r0_batch_size: int = 10,
    resid_rounds: int = 3,
    resid_batch_size: int = 12,
    device_str: str = "cuda:0",
) -> Path:
    """Run Stage 1 Category-Guided Residual LLM Alpha Mining (25 queries per seed)."""
    t0 = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    codes = load_instruments(qlib_path, market)
    data = StockData(codes, train_start, train_end, qlib_path=qlib_path, device=device)
    close = Feature(FeatureType.CLOSE)
    y_full = Sub(Div(Ref(close, -label_days), close), Constant(1.0)).evaluate(data).float()
    y_full = torch.where(torch.isfinite(y_full), y_full, torch.full_like(y_full, float("nan")))

    n_keep = y_full.shape[0] - (label_days + 1)
    y_raw = y_full[:n_keep]
    y_rz = cs_rank_zscore_nan(y_raw)
    years = data.dates.year.values[:n_keep].astype(np.int32)
    print(f"[Mine {market} s{seed}] causal train: {n_keep}d x {y_raw.shape[1]} stocks", flush=True)

    seen_canon = set()
    archive: List[Dict] = []
    fnorm_cache: Dict[str, Tensor] = {}
    frz_cache: Dict[str, Tensor] = {}
    rejected: Dict[str, List[str]] = {b: [] for b, _ in BAND_SPECS}
    call_idx = 0

    def ingest(lines: List[str], band_name: str, rnd: int) -> Tuple[int, int]:
        n_parsed, n_added = 0, 0
        for s in lines:
            try:
                expr = parse_expr(s)
            except Exception:
                continue
            ok_ast, _ = validate_expr_ast(expr, max_window=40)
            if not ok_ast:
                continue
            canon = str(expr)
            if canon in seen_canon:
                continue
            seen_canon.add(canon)
            n_parsed += 1
            try:
                f_raw = expr.evaluate(data).float()
            except Exception:
                continue
            if f_raw.ndim != 2 or torch.isinf(f_raw).any().item():
                continue
            f_raw = torch.where(torch.isfinite(f_raw), f_raw, torch.full_like(f_raw, float("nan")))[:n_keep]
            stats = _factor_train_stats(f_raw, y_raw, y_rz, years)
            if stats is None:
                continue
            fnorm = normalize_by_day_safe(f_raw)
            outlier_rate = float((fnorm.abs() > 4.0).float().mean().item())
            frz = cs_rank_zscore_nan(f_raw)
            archive.append({"expr": canon, "band": band_name, "round": rnd, "outlier_rate": outlier_rate, **stats})
            fnorm_cache[canon] = fnorm
            frz_cache[canon] = frz
            n_added += 1
            if abs(stats["icir"]) < 0.12 or abs(stats["ricir"]) < 0.12 or stats["yr_cons"] < 0.55 or outlier_rate > 0.025:
                rejected[band_name].append(
                    f"{canon} (ric={stats['ric']:+.3f}, ic={stats['ic']:+.3f}, ricir={stats['ricir']:+.2f}, yr={stats['yr_cons']:.2f})"
                )
        return n_parsed, n_added

    def run_parallel(tasks: List[Tuple[str, str, str, int]]) -> List[Tuple[str, str]]:
        results: Dict[str, Tuple[str, str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = {
                ex.submit(_ask_llm, api_url, model, prompt, temperature, s, api_key): (b, tag)
                for (b, tag, prompt, s) in tasks
            }
            for fut in concurrent.futures.as_completed(futs):
                b, tag = futs[fut]
                txt, _ = fut.result()
                results[tag] = (b, txt)
        return [results[tag] for (_, tag, _, _) in tasks]

    # Round 0: 2 batches x 5 categories = 10 calls
    for batch in range(r0_batches):
        tasks = []
        for band_name, band_desc in BAND_SPECS:
            call_idx += 1
            band_seen = [a["expr"] for a in archive if a["band"] == band_name][-15:]
            prompt = PROMPT_R0.format(
                dsl=DSL_DOC, band_desc=band_desc, n=r0_batch_size,
                avoid="\n".join(band_seen) if band_seen else "(none)",
            )
            tasks.append((band_name, f"r0_{band_name}_b{batch}", prompt, seed * 1000 + call_idx))
        for band_name, txt in run_parallel(tasks):
            np_, na_ = ingest(_parse_llm_lines(txt), band_name, rnd=0)
            print(f"  [R0 {band_name:21s} b{batch}] parsed={np_:2d} added={na_:2d} total={len(archive)}", flush=True)

    # Rounds 1..3: 3 rounds x 5 categories = 15 calls
    for rnd in range(1, resid_rounds + 1):
        pool_exprs, y_perp = _compute_orthogonal_residual(archive, fnorm_cache, frz_cache, y_rz)
        for a in archive:
            rr = daily_pearson_masked(frz_cache[a["expr"]], y_perp)
            ri = daily_pearson_masked(fnorm_cache[a["expr"]], y_perp)
            a["resid_ric"] = float(rr.mean().item())
            a["resid_ricir"] = float(rr.mean().item() / max(rr.std().item(), 1e-6))
            a["resid_icir"] = float(ri.mean().item() / max(ri.std().item(), 1e-6))

        tasks = []
        for band_name, band_desc in BAND_SPECS:
            call_idx += 1
            band_items = [a for a in archive if a["band"] == band_name and a["outlier_rate"] <= 0.03]
            if not band_items:
                band_items = [a for a in archive if a["band"] == band_name]
            band_items.sort(
                key=lambda a: 0.5 * (abs(a["resid_ricir"]) + abs(a["resid_icir"])) + _quality_score(a),
                reverse=True,
            )
            good_lines = [
                f"  {a['expr']}  | ric={a['ric']:+.3f} ricir={a['ricir']:+.2f} ic={a['ic']:+.3f} yr={a['yr_cons']:.2f} resid={a['resid_ric']:+.3f}"
                for a in band_items[:6]
            ]
            bad_lines = rejected[band_name][-4:]
            avoid_lines = [a["expr"] for a in archive if a["band"] == band_name][-20:]
            prompt = PROMPT_RESID.format(
                dsl=DSL_DOC, band_desc=band_desc, rnd=rnd, n=resid_batch_size,
                good="\n".join(good_lines) if good_lines else "  (none yet)",
                bad="\n".join("  " + x for x in bad_lines) if bad_lines else "  (none)",
                avoid="\n".join(avoid_lines) if avoid_lines else "(none)",
            )
            tasks.append((band_name, f"r{rnd}_{band_name}", prompt, seed * 1000 + call_idx))
        for band_name, txt in run_parallel(tasks):
            np_, na_ = ingest(_parse_llm_lines(txt), band_name, rnd=rnd)
            print(f"  [R{rnd} {band_name:21s}] parsed={np_:2d} added={na_:2d} total={len(archive)}", flush=True)

    pool_file = out_path / f"prism_llm_{market.lower()}_s{seed}.json"
    payload = {
        "exprs": [a["expr"] for a in archive],
        "meta": [{"expr": a["expr"], "band": a["band"], "round": a["round"]} for a in archive],
        "band_counts": {b: sum(1 for a in archive if a["band"] == b) for b in BAND_NAMES},
        "market": market.lower(),
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(pool_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"[Mine {market} s{seed}] saved {len(archive)} unique formulas -> {pool_file}", flush=True)
    return pool_file


# Stage 2: dual-branch causal factor combination

def _get_r0_count(pdict: Dict) -> int:
    meta = pdict.get("meta", [])
    if meta and "round" in meta[0]:
        return sum(1 for m in meta if int(m["round"]) == 0)
    bands = [m["band"] for m in meta]
    rle: List[List] = []
    for b in bands:
        if rle and rle[-1][0] == b:
            rle[-1][1] += 1
        else:
            rle.append([b, 1])
    return sum(x[1] for x in rle[:10])


def run_linear_branch(
    X_rz: Tensor,
    y_rz: Tensor,
    y_valid: Tensor,
    bands: Optional[Sequence[str]],
    oos_start: int,
    lam_list: Sequence[float] = (0.1,),
    mode: str = "category",
    k_band: int = 4,
    k_flat: int = 20,
    tau_corr: float = 0.65,
    alpha_fast: float = 0.25,
    hl_fast: float = 750.0,
) -> Tuple[Dict[float, Tensor], float]:
    """Category-wise orthogonal Ridge branch over multi-scale causal moments on H_t (tau <= t - 21)."""
    t_len, s_len, p_len = X_rz.shape
    device = X_rz.device
    shift = 21
    decay = math.exp(-math.log(2.0) / hl_fast)

    Gl = torch.zeros((p_len, p_len), device=device, dtype=torch.float64)
    bl = torch.zeros((p_len,), device=device, dtype=torch.float64)
    rl = torch.zeros((p_len,), device=device, dtype=torch.float64)
    r2l = torch.zeros((p_len,), device=device, dtype=torch.float64)
    Gf, bf, rf, r2f = Gl.clone(), bl.clone(), rl.clone(), r2l.clone()
    cnt, wf = 0.0, 0.0

    def update_day(tau: int) -> None:
        nonlocal cnt, wf, Gl, bl, rl, r2l, Gf, bf, rf, r2f
        m = y_valid[tau]
        if int(m.sum().item()) < 10:
            return
        xt = X_rz[tau, m].double()
        yt = y_rz[tau, m].double()
        e_t = float(xt.shape[0])
        gt = (xt.T @ xt) / e_t
        bt = (xt.T @ yt) / e_t
        xm = xt - xt.mean(0, keepdim=True)
        ym = yt - yt.mean()
        xs = xm.std(0, unbiased=False)
        ys = ym.std(unbiased=False)
        rt = torch.where(
            (xs > 1e-3) & (ys > 1e-3),
            (xm * ym[:, None]).mean(0) / (xs * ys).clamp(min=1e-6),
            torch.zeros_like(bt),
        )
        cnt += 1.0
        Gl += gt
        bl += bt
        rl += rt
        r2l += rt * rt
        if alpha_fast > 0:
            Gf = decay * Gf + gt
            bf = decay * bf + bt
            rf = decay * rf + rt
            r2f = decay * r2f + rt * rt
            wf = decay * wf + 1.0

    for tau in range(oos_start - shift):
        update_day(tau)

    uniq_bands = [b for b in BAND_NAMES if b in set(bands)] if bands is not None else []
    use_category = mode == "category" and len(uniq_bands) > 1
    band_idx_map = (
        {b: [i for i, bb in enumerate(bands) if bb == b] for b in uniq_bands} if use_category else {}
    )

    n_oos = t_len - oos_start
    out = {lam: torch.zeros((n_oos, s_len), device=device, dtype=torch.float32) for lam in lam_list}
    nsel_hist: List[int] = []

    for idx_d, cur in enumerate(range(oos_start, t_len)):
        update_day(cur - shift)
        if alpha_fast > 0:
            G = (1.0 - alpha_fast) * (Gl / cnt) + alpha_fast * (Gf / wf)
            b = (1.0 - alpha_fast) * (bl / cnt) + alpha_fast * (bf / wf)
            rm = (1.0 - alpha_fast) * (rl / cnt) + alpha_fast * (rf / wf)
            r2 = (1.0 - alpha_fast) * (r2l / cnt) + alpha_fast * (r2f / wf)
        else:
            G, b, rm, r2 = Gl / cnt, bl / cnt, rl / cnt, r2l / cnt

        psi = rm / (r2 - rm * rm).clamp(min=1e-12).sqrt()
        diag = G.diag().clamp(min=1e-12).sqrt()
        corr = G / (diag[:, None] * diag[None, :])
        xc = X_rz[cur].double()

        if use_category:
            band_sels: List[List[int]] = []
            tot_sel = 0
            for bn in uniq_bands:
                ct = torch.tensor(band_idx_map[bn], device=device)
                ord_b = ct[torch.argsort(psi[ct].abs(), descending=True)].tolist()
                sel_b: List[int] = []
                for ii in ord_b:
                    if all(abs(corr[ii, q].item()) <= tau_corr for q in sel_b):
                        sel_b.append(ii)
                        if len(sel_b) >= k_band:
                            break
                if sel_b:
                    band_sels.append(sel_b)
                    tot_sel += len(sel_b)
            nsel_hist.append(tot_sel)
            for lam in lam_list:
                bpreds, bw = [], []
                for sel_b in band_sels:
                    Gb = G[sel_b][:, sel_b]
                    bb = b[sel_b]
                    eye = torch.eye(len(sel_b), device=device, dtype=torch.float64)
                    wb = torch.linalg.solve(Gb + lam * Gb.diag().mean() * eye, bb)
                    bp = xc[:, sel_b] @ wb
                    bp = (bp - bp.mean()) / bp.std().clamp(min=1e-6)
                    bpreds.append(bp)
                    ic_b = float((wb @ bb) / torch.sqrt((wb @ Gb @ wb).clamp(min=1e-12)))
                    bw.append(max(ic_b, 0.01))
                bwt = torch.tensor(bw, device=device, dtype=torch.float64)
                bwt = bwt / bwt.sum()
                out[lam][idx_d] = (bwt[:, None] * torch.stack(bpreds, dim=0)).sum(0).float()
        else:
            ok = torch.where((rm.abs() > 0.015) & (psi.abs() > 0.15))[0]
            if len(ok) == 0:
                ok = torch.tensor([int(psi.abs().argmax().item())], device=device)
            ord_fl = ok[torch.argsort(psi[ok].abs(), descending=True)].tolist()
            sel: List[int] = []
            for ii in ord_fl:
                if all(abs(corr[ii, q].item()) <= tau_corr for q in sel):
                    sel.append(ii)
                    if len(sel) >= k_flat:
                        break
            nsel_hist.append(len(sel))
            Gs = G[sel][:, sel]
            bs = b[sel]
            dbar = Gs.diag().mean()
            eye = torch.eye(len(sel), device=device, dtype=torch.float64)
            xcs = xc[:, sel]
            for lam in lam_list:
                w = torch.linalg.solve(Gs + lam * dbar * eye, bs)
                out[lam][idx_d] = (xcs @ w).float()

    return {lam: clipz_2d(v, 4.0) for lam, v in out.items()}, float(np.mean(nsel_hist))


def run_nonlinear_branch(
    X_np: np.ndarray,
    y_np: np.ndarray,
    yv_np: np.ndarray,
    years: np.ndarray,
    oos_years: Sequence[int],
    seed: int,
    depth: int = 3,
    leaves: int = 7,
    n_est: int = 120,
    lr: float = 0.04,
    mcs: int = 1200,
    l1: float = 5.0,
    l2: float = 50.0,
    hl: float = 1000.0,
    n_jobs: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Annual walk-forward shallow regularized LightGBM with strict 21-day purge gap."""
    t_len, s_len, p_len = X_np.shape
    start_day = int(np.where(years == oos_years[0])[0][0])
    pred_nl = np.zeros((t_len - start_day, s_len), dtype=np.float32)

    for yr in oos_years:
        yr_idx = np.where(years == yr)[0]
        if len(yr_idx) == 0:
            continue
        tr_end = int(yr_idx[0]) - 21
        taus = np.arange(tr_end)
        day_w = (
            np.power(2.0, -(tr_end - 1 - taus) / hl).astype(np.float32)
            if np.isfinite(hl)
            else np.ones(tr_end, dtype=np.float32)
        )
        mask_tr = yv_np[:tr_end]
        X_tr = X_np[:tr_end][mask_tr]
        y_tr = y_np[:tr_end][mask_tr]
        w_tr = np.broadcast_to(day_w[:, None], (tr_end, s_len))[mask_tr]

        model = lgb.LGBMRegressor(
            max_depth=depth,
            num_leaves=leaves,
            n_estimators=n_est,
            learning_rate=lr,
            min_child_samples=mcs,
            subsample=0.75,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=l1,
            reg_lambda=l2,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        pred_nl[yr_idx - start_day] = model.predict(X_np[yr_idx].reshape(-1, p_len)).reshape(len(yr_idx), s_len)

    return clipz_2d(torch.tensor(pred_nl, device=device, dtype=torch.float32), 4.0)


def fuse_causal_maxir(
    s_lin: Tensor,
    s_nl: Tensor,
    y_hyb_oos: Tensor,
    y_rz_oos: Tensor,
    val_len: int,
    gamma: float = 0.05,
    w_min: float = 0.15,
    hl: float = 750.0,
    alpha_fast: float = 0.25,
    fixed_w_nl: Optional[float] = None,
) -> Tuple[Tensor, float]:
    """Strictly causal bounded Max-IR fusion over realized out-of-sample history H_t (tau <= t - 21)."""
    t_len = s_lin.shape[0]
    device = s_lin.device
    if fixed_w_nl is not None:
        return (1.0 - fixed_w_nl) * s_lin + fixed_w_nl * s_nl, float(fixed_w_nl)

    ic0 = torch.nan_to_num(batch_pearsonr(s_lin, y_hyb_oos), nan=0.0).cpu().numpy()
    ic1 = torch.nan_to_num(batch_pearsonr(s_nl, y_hyb_oos), nan=0.0).cpu().numpy()
    ric0 = torch.nan_to_num(batch_spearmanr(s_lin, y_rz_oos), nan=0.0).cpu().numpy()
    ric1 = torch.nan_to_num(batch_spearmanr(s_nl, y_rz_oos), nan=0.0).cpu().numpy()
    arr = 0.5 * (np.stack([ic0, ic1], axis=1) + np.stack([ric0, ric1], axis=1)).astype(np.float64)

    shift = 21
    decay = math.exp(-math.log(2.0) / hl)
    s1l, s2l, cl = np.zeros(2), np.zeros((2, 2)), 0.0
    s1f, s2f, wf = np.zeros(2), np.zeros((2, 2)), 0.0
    mu_o = np.zeros((t_len, 2))
    cov_o = np.zeros((t_len, 2, 2))

    for t in range(t_len):
        v = arr[t]
        vv = np.outer(v, v)
        cl += 1.0
        s1l += v
        s2l += vv
        wf = decay * wf + 1.0
        s1f = decay * s1f + v
        s2f = decay * s2f + vv
        m1 = (1.0 - alpha_fast) * (s1l / cl) + alpha_fast * (s1f / wf)
        m2 = (1.0 - alpha_fast) * (s2l / cl) + alpha_fast * (s2f / wf)
        mu_o[t] = m1
        cov_o[t] = m2 - np.outer(m1, m1)

    w_arr = np.empty((t_len, 2), dtype=np.float32)
    for cur in range(t_len):
        te = cur - shift
        if te < 20:
            w_nl = 0.5
        else:
            mu, cov = mu_o[te], cov_o[te]
            dbar = max(float(np.trace(cov)) / 2.0, 1e-6)
            w_raw = np.linalg.solve(cov + gamma * dbar * np.eye(2), mu)
            w_pos = np.maximum(w_raw, 0.0)
            s = float(w_pos.sum())
            if s <= 1e-12:
                w_nl = 1.0 - w_min if mu[1] > mu[0] else w_min
            else:
                w_nl = float(np.clip(w_pos[1] / s, w_min, 1.0 - w_min))
        w_arr[cur] = [1.0 - w_nl, w_nl]

    wt = torch.tensor(w_arr, device=device, dtype=torch.float32)
    pred = wt[:, 0:1] * s_lin + wt[:, 1:2] * s_nl
    return pred, float(w_arr[val_len:, 1].mean())


def evaluate_markets(
    markets: Sequence[str],
    seeds: Sequence[int],
    pool_dir: str,
    qlib_path: str,
    out_dir: str,
    run_ablations: bool = False,
    run_sensitivity: bool = False,
    n_jobs: int = 8,
    device_str: str = "cuda:0",
) -> None:
    """Run Stage 2 Dual-Branch Causal Combination and report 5-seed metrics."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    oos_years = [2021, 2022, 2023, 2024]
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for mk in markets:
        t0 = time.time()
        codes = load_instruments(qlib_path, mk)
        data = StockData(codes, "2010-01-01", "2024-12-31", qlib_path=qlib_path, device=device)
        years = data.dates.year.values
        oos_start = int(np.where(years == 2021)[0][0])
        test_start = int(np.where(years == 2022)[0][0])
        val_len = test_start - oos_start

        close = Feature(FeatureType.CLOSE)
        tgt = Sub(Div(Ref(close, -20), close), Constant(1.0)).evaluate(data)
        tgt = torch.where(torch.isfinite(tgt), tgt, torch.full_like(tgt, float("nan")))
        y_rz = rankz_2d(tgt)
        y_hyb = 0.7 * clipz_2d(tgt, 4.0) + 0.3 * y_rz
        y_valid = torch.isfinite(tgt)
        y_np, yv_np = y_hyb.cpu().numpy(), y_valid.cpu().numpy()

        tgt_oos = tgt[oos_start:]
        y_hyb_oos = torch.where(y_valid[oos_start:], y_hyb[oos_start:], torch.full_like(tgt_oos, float("nan")))
        y_rz_oos = torch.where(y_valid[oos_start:], y_rz[oos_start:], torch.full_like(tgt_oos, float("nan")))

        print(f"\n=== {mk.upper()} loaded ({data.n_days}d x {data.n_stocks} stocks) in {time.time() - t0:.1f}s ===", flush=True)
        summary: Dict[str, List[Tuple[Dict[str, float], float]]] = {}

        def record(tag: str, seed: int, pred_oos: Tensor, w_nl: float, extra: Optional[Dict] = None) -> None:
            vres, _ = evaluate_metrics(pred_oos[:val_len], tgt_oos[:val_len], label_days=20, test_years=1.0)
            tres, ret_s = evaluate_metrics(pred_oos[val_len:], tgt_oos[val_len:], label_days=20, test_years=3.0)
            payload = {"method": tag, "market": mk, "seed": seed, "valid": vres, "test": tres, "w_nl_mean": w_nl}
            if extra:
                payload.update(extra)
            out_f = out_root / f"{tag}_{mk}_s{seed}.json"
            with open(out_f, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=1)
            np.save(str(out_f) + ".ret_s.npy", ret_s.cpu().numpy())
            summary.setdefault(tag, []).append((tres, w_nl))

        for s in seeds:
            ts = time.time()
            pool_file = Path(pool_dir) / f"prism_llm_{mk}_s{s}.json"
            if not pool_file.exists():
                raise FileNotFoundError(f"Factor pool not found: {pool_file}")
            pdict = json.load(open(pool_file, encoding="utf-8"))
            bands = [m["band"] for m in pdict["meta"]] if "meta" in pdict else None
            exprs = [parse_expr(e) for e in pdict["exprs"]]

            X_rz = torch.empty((data.n_days, data.n_stocks, len(exprs)), device=device, dtype=torch.float32)
            for p, ex in enumerate(exprs):
                X_rz[:, :, p] = rankz_2d(ex.evaluate(data))
            X_np = X_rz.cpu().numpy()

            lam_list = (0.05, 0.1, 0.5, 1.0, 2.0) if run_sensitivity else (0.1,)
            lin_cat, nsel = run_linear_branch(X_rz, y_rz, y_valid, bands, oos_start, lam_list=lam_list, mode="category", k_band=4)
            nl_d3 = run_nonlinear_branch(X_np, y_np, yv_np, years, oos_years, s, depth=3, leaves=7, n_jobs=n_jobs, device=device)

            prism_corr = float(torch.nan_to_num(batch_pearsonr(lin_cat[0.1][val_len:], nl_d3[val_len:]), nan=0.0).mean().item())
            p_full, w_full = fuse_causal_maxir(lin_cat[0.1], nl_d3, y_hyb_oos, y_rz_oos, val_len, gamma=0.05, w_min=0.15)
            record("AlphaPrism", s, p_full, w_full, {"n_pool": len(exprs), "n_selected_mean": nsel, "prism_corr": prism_corr})

            if run_ablations:
                r0_n = _get_r0_count(pdict)
                lin_r0, _ = run_linear_branch(X_rz[:, :, :r0_n], y_rz, y_valid, bands[:r0_n] if bands else None, oos_start, lam_list=(0.1,), mode="category", k_band=4)
                nl_r0 = run_nonlinear_branch(X_np[:, :, :r0_n], y_np, yv_np, years, oos_years, s, depth=3, leaves=7, n_jobs=n_jobs, device=device)
                p_r0, w_r0 = fuse_causal_maxir(lin_r0[0.1], nl_r0, y_hyb_oos, y_rz_oos, val_len, gamma=0.05, w_min=0.15)
                record("wo_residual", s, p_r0, w_r0, {"n_pool": r0_n})

                lin_fl20, nsel_fl20 = run_linear_branch(X_rz, y_rz, y_valid, bands, oos_start, lam_list=(0.1,), mode="flat", k_flat=20)
                p_fl20, w_fl20 = fuse_causal_maxir(lin_fl20[0.1], nl_d3, y_hyb_oos, y_rz_oos, val_len, gamma=0.05, w_min=0.15)
                record("global_top20", s, p_fl20, w_fl20, {"n_selected_mean": nsel_fl20})

                record("linear_only", s, lin_cat[0.1], 0.0)
                record("nonlinear_only", s, nl_d3, 1.0)

            if run_sensitivity:
                for lam, tag in [(0.05, "lam_0p05"), (0.5, "lam_0p5"), (1.0, "lam_1p0"), (2.0, "lam_2p0")]:
                    pl, wl = fuse_causal_maxir(lin_cat[lam], nl_d3, y_hyb_oos, y_rz_oos, val_len, gamma=0.05, w_min=0.15)
                    record(tag, s, pl, wl)
                for depth, leaves, tag in [(2, 3, "depth_2"), (4, 15, "depth_4")]:
                    nl_cur = run_nonlinear_branch(X_np, y_np, yv_np, years, oos_years, s, depth=depth, leaves=leaves, n_jobs=n_jobs, device=device)
                    pd_, wd = fuse_causal_maxir(lin_cat[0.1], nl_cur, y_hyb_oos, y_rz_oos, val_len, gamma=0.05, w_min=0.15)
                    record(tag, s, pd_, wd)
                for w_fix, tag in [(0.0, "wnl_0p00"), (0.25, "wnl_0p25"), (0.50, "wnl_0p50"), (0.75, "wnl_0p75"), (1.0, "wnl_1p00")]:
                    pw, _ = fuse_causal_maxir(lin_cat[0.1], nl_d3, y_hyb_oos, y_rz_oos, val_len, fixed_w_nl=w_fix)
                    record(tag, s, pw, w_fix)

            print(f"  [{mk} s{s}] done in {time.time() - ts:.1f}s", flush=True)

        print(f"\n--- {mk.upper()} SUMMARY ({len(seeds)}-seed mean) ---")
        print(f"{'Variant':<16s} | {'IC':>6s} {'ICIR':>6s} {'RankIC':>7s} {'RICIR':>6s} {'AR(%)':>11s} {'MDD(%)':>8s} {'SR':>5s} {'w_nl':>5s}")
        print("-" * 82)
        for tag, items in summary.items():
            rs = [x[0] for x in items]
            ars = [r["ret"] * 100.0 for r in rs]
            print(
                f"{tag:<16s} | "
                f"{np.mean([r['ic'] for r in rs]):6.3f} "
                f"{np.mean([r['icir'] for r in rs]):6.3f} "
                f"{np.mean([r['ric'] for r in rs]):7.3f} "
                f"{np.mean([r['ricir'] for r in rs]):6.3f} "
                f"{np.mean(ars):5.2f}+-{np.std(ars):4.2f} "
                f"{-np.mean([r['ret_mdd'] * 100.0 for r in rs]):7.2f}% "
                f"{np.mean([r['ret_sharpe'] for r in rs]):5.2f} "
                f"{np.mean([x[1] for x in items]):5.2f}"
            )


# CLI

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaPrism: Category-Guided LLM Alpha Mining & Dual-Branch Causal Combination")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--market", choices=["csi300", "csi500", "all"], default="all")
        p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
        p.add_argument("--qlib_path", default=os.environ.get("QLIB_DATA_PATH", "/path/to/qlib_data/cn_data"))
        p.add_argument("--device", default="cuda:0")

    p_mine = sub.add_parser("mine", help="Run Stage 1 Category-Guided Residual LLM Alpha Mining")
    add_common(p_mine)
    p_mine.add_argument("--api_url", default=os.environ.get("OPENAI_API_URL", "http://localhost:8000/v1/chat/completions"))
    p_mine.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p_mine.add_argument("--model", default="Qwen3-32B-AWQ")
    p_mine.add_argument("--temperature", type=float, default=0.85)
    p_mine.add_argument("--out_dir", default="pools")

    p_eval = sub.add_parser("eval", help="Run Stage 2 Dual-Branch Causal Combination & Evaluation")
    add_common(p_eval)
    p_eval.add_argument("--pool_dir", default="pools")
    p_eval.add_argument("--out_dir", default="results")
    p_eval.add_argument("--n_jobs", type=int, default=8)
    p_eval.add_argument("--ablations", action="store_true", help="Also run ablation variants")
    p_eval.add_argument("--sensitivity", action="store_true", help="Also run sensitivity sweeps")

    p_run = sub.add_parser("run", help="Run Stage 1 mining followed by Stage 2 evaluation")
    add_common(p_run)
    p_run.add_argument("--api_url", default=os.environ.get("OPENAI_API_URL", "http://localhost:8000/v1/chat/completions"))
    p_run.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    p_run.add_argument("--model", default="Qwen3-32B-AWQ")
    p_run.add_argument("--temperature", type=float, default=0.85)
    p_run.add_argument("--pool_dir", default="pools")
    p_run.add_argument("--out_dir", default="results")
    p_run.add_argument("--n_jobs", type=int, default=8)
    p_run.add_argument("--ablations", action="store_true")
    p_run.add_argument("--sensitivity", action="store_true")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    markets = ["csi300", "csi500"] if args.market == "all" else [args.market]

    if args.cmd == "mine":
        for mk in markets:
            for s in args.seeds:
                mine_pool(
                    market=mk, seed=s, qlib_path=args.qlib_path,
                    api_url=args.api_url, model=args.model, out_dir=args.out_dir,
                    api_key=args.api_key, temperature=args.temperature, device_str=args.device,
                )
    elif args.cmd == "eval":
        evaluate_markets(
            markets=markets, seeds=args.seeds, pool_dir=args.pool_dir,
            qlib_path=args.qlib_path, out_dir=args.out_dir,
            run_ablations=args.ablations, run_sensitivity=args.sensitivity,
            n_jobs=args.n_jobs, device_str=args.device,
        )
    elif args.cmd == "run":
        for mk in markets:
            for s in args.seeds:
                mine_pool(
                    market=mk, seed=s, qlib_path=args.qlib_path,
                    api_url=args.api_url, model=args.model, out_dir=args.pool_dir,
                    api_key=args.api_key, temperature=args.temperature, device_str=args.device,
                )
        evaluate_markets(
            markets=markets, seeds=args.seeds, pool_dir=args.pool_dir,
            qlib_path=args.qlib_path, out_dir=args.out_dir,
            run_ablations=args.ablations, run_sensitivity=args.sensitivity,
            n_jobs=args.n_jobs, device_str=args.device,
        )


if __name__ == "__main__":
    main()
