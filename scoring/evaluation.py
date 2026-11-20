from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


MISSING_BIN = -1
OTHER_CATEGORY = '___other___'
MISSING_CATEGORY = '___missing___'


@dataclass(frozen=True)
class Bins:
    edges: np.ndarray | None = None
    categories: frozenset[str] | None = None

    @classmethod
    def fit(cls, values):
        if not pd.api.types.is_numeric_dtype(values):
            counts = values.value_counts()
            frequent = counts[counts >= max(50, len(values) * 0.005)].index
            return cls(categories=frozenset(frequent))

        observed = values.dropna().to_numpy(dtype=float)
        unique = np.unique(observed)
        if len(unique) <= 10:
            return cls(edges=unique[:-1])
        edges = np.unique(np.quantile(observed, np.arange(1, 10) / 10))
        return cls(edges=edges[edges < unique[-1]])

    def transform(self, values):
        if self.categories is not None:
            categories = values.where(values.isin(self.categories), OTHER_CATEGORY)
            return categories.where(values.notna(), MISSING_CATEGORY)

        numeric = values.to_numpy(dtype=float, na_value=np.nan)
        groups = np.searchsorted(self.edges, numeric, side='left')
        groups[np.isnan(numeric)] = MISSING_BIN
        return pd.Series(groups, index=values.index)


class WoEEncoder:
    def fit(self, features, target):
        self.bins = {name: Bins.fit(features[name]) for name in features}
        self.weights = {}
        for name, bins in self.bins.items():
            grouped = target.groupby(bins.transform(features[name])).agg(['sum', 'size'])
            # Сглаживание для групп без наблюдений одного из классов.
            bad = grouped['sum'] + 0.5
            good = grouped['size'] - grouped['sum'] + 0.5
            self.weights[name] = np.log((bad / bad.sum()) / (good / good.sum()))
        return self

    def transform(self, features):
        return pd.DataFrame({
            name: bins.transform(features[name]).map(self.weights[name]).fillna(0)
            for name, bins in self.bins.items()
        }, index=features.index)


def metrics(target, probabilities):
    auc = float(roc_auc_score(target, probabilities))
    return {
        'roc_auc': auc,
        'gini': 2 * auc - 1,
        'brier': float(brier_score_loss(target, probabilities)),
        'log_loss': float(log_loss(target, probabilities)),
    }


def population_stability(reference_groups, current_groups):
    shares = pd.concat([
        reference_groups.value_counts().rename('reference'),
        current_groups.value_counts().rename('current'),
    ], axis=1).fillna(0).astype(float)
    shares += 0.5
    shares = shares / shares.sum(axis=0)
    difference = shares['current'] - shares['reference']
    return float((difference * np.log(shares['current'] / shares['reference'])).sum())


def decision_table(target, scores, thresholds):
    """Сводка политики: одобряем заявки, для которых PD <= порога."""
    order = np.argsort(scores)
    approved = np.searchsorted(scores[order], thresholds, side='right')
    bad_approved = np.r_[0, np.cumsum(np.asarray(target)[order])][approved]
    expected_bad = np.r_[0.0, np.cumsum(scores[order])][approved]
    denominator = np.where(approved > 0, approved, np.nan)
    rate = bad_approved / denominator

    # 95%-й интервал Уилсона для доли TARGET = 1 среди одобренных.
    z = 1.959963984540054
    adjustment = 1 + z**2 / denominator
    centre = (rate + z**2 / (2 * denominator)) / adjustment
    radius = z * np.sqrt(rate * (1 - rate) / denominator + z**2 / (4 * denominator**2)) / adjustment
    return pd.DataFrame({
        'threshold': thresholds,
        'approved': approved,
        'approval_rate': approved / len(target),
        'default_rate': rate,
        'expected_default_rate': expected_bad / denominator,
        'default_low': np.maximum(centre - radius, 0),
        'default_high': np.minimum(centre + radius, 1),
        'good_rejected': int((target == 0).sum()) - (approved - bad_approved),
    })
