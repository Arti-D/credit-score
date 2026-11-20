import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from scoring.data import PROCESSED, SEED, SPLIT, TARGET, load_data
from scoring.evaluation import Bins, WoEEncoder, decision_table, metrics, population_stability


WOE = 'Logistic WoE'
LIGHTGBM = 'LightGBM'
RISK_LIMIT = 0.05
THRESHOLDS = np.linspace(0, 1, 1001)
RESULTS = PROCESSED / 'scoring_results.json'


def fit_models(features, target, train_index):
    train = features.loc[train_index]
    train_target = target.loc[train_index]
    encoder = WoEEncoder().fit(train, train_target)
    logistic = LogisticRegression(max_iter=1000).fit(encoder.transform(train), train_target)

    # Категории определяются на train, новые значения LightGBM обрабатывает как пропуски.
    tree_features = features.copy()
    for name in train.select_dtypes(exclude='number'):
        categories = sorted(train[name].dropna().unique())
        tree_features[name] = pd.Categorical(tree_features[name], categories=categories)
    for name in train.select_dtypes(include='number'):
        tree_features[name] = tree_features[name].astype('float32')

    boosting = lgb.LGBMClassifier(
        objective='binary', n_estimators=420, learning_rate=0.035, num_leaves=31,
        min_child_samples=100, reg_lambda=2, colsample_bytree=0.85,
        deterministic=True, force_col_wise=True, random_state=SEED, n_jobs=4, verbosity=-1,
    )
    boosting.fit(tree_features.loc[train_index], train_target)
    return encoder, logistic, tree_features, boosting.booster_


def compare_models(target, scores, calibration_index, policy_index):
    results = {}
    calibrators = {}
    policy_probabilities = {}
    for name, raw_scores in scores.items():
        calibrators[name] = LogisticRegression(C=1e6).fit(
            raw_scores[calibration_index].reshape(-1, 1), target.loc[calibration_index],
        )
        predictions = predict_probabilities(raw_scores[policy_index], calibrators[name])
        validation = {
            method: metrics(target.loc[policy_index], probability)
            for method, probability in predictions.items()
        }
        chosen_method = min(validation, key=lambda method: validation[method]['log_loss'])
        results[name] = {'calibration': chosen_method, 'validation': validation}
        policy_probabilities[name] = predictions[chosen_method]
    return results, calibrators, policy_probabilities


def predict_probabilities(scores, calibrator):
    return {'raw': expit(scores), 'sigmoid': calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]}


def evaluate_test(models, target, scores, calibrators):
    predictions = {}
    for name, model in models.items():
        predictions[name] = predict_probabilities(scores[name], calibrators[name])
        model['test'] = {
            method: metrics(target, probability)
            for method, probability in predictions[name].items()
        }
        probability = predictions[name][model['calibration']]
        false_positive, true_positive, _ = roc_curve(target, probability)
        points = np.linspace(0, len(false_positive) - 1, 350, dtype=int)
        model['roc'] = [false_positive[points].tolist(), true_positive[points].tolist()]
    return predictions


def explain_boosting(boosting, features, test_index, case_index):
    sample = features.loc[test_index].sample(n=min(2000, len(test_index)), random_state=SEED)
    sample_contributions = boosting.predict(sample, pred_contrib=True, num_threads=4)
    importance = pd.Series(np.abs(sample_contributions[:, :-1]).mean(axis=0), index=features.columns)
    case = features.loc[[case_index]]
    contributions = boosting.predict(case, pred_contrib=True, num_threads=4)[0]
    raw_score = float(boosting.predict(case, raw_score=True, num_threads=4)[0])
    np.testing.assert_allclose(contributions.sum(), raw_score, atol=1e-7)
    local = pd.Series(contributions[:-1], index=features.columns)
    return {
        'sample_size': len(sample),
        'importance': importance.nlargest(12).to_dict(),
        'case_contributions': local.loc[local.abs().nlargest(8).index].to_dict(),
        'case_baseline': float(contributions[-1]),
        'case_raw_score': raw_score,
    }


def main():
    print('DuckDB: сборка признаков и проверка разбиения', flush=True)
    data = load_data()
    target = data.pop(TARGET)
    splits = data.pop(SPLIT)
    train_index = data.index[splits.eq('train')]
    validation_index = data.index[splits.eq('val')]
    test_index = data.index[splits.eq('test')]
    calibration_index, policy_index = train_test_split(
        validation_index, test_size=0.5, stratify=target.loc[validation_index], random_state=SEED,
    )

    print(f'Обучение WoE и LightGBM на {len(train_index):,} заявках', flush=True)
    encoder, logistic, tree_features, boosting = fit_models(data, target, train_index)
    scores = {
        WOE: logistic.decision_function(encoder.transform(data)),
        LIGHTGBM: boosting.predict(tree_features, raw_score=True, num_threads=4),
    }
    models, calibrators, probabilities = compare_models(target, scores, calibration_index, policy_index)
    selected = max(models, key=lambda name: models[name]['validation'][models[name]['calibration']]['roc_auc'])
    policy_scores = probabilities[selected]

    print('Подбор порога на validation, итоговая оценка на test и TreeSHAP', flush=True)
    policy = decision_table(target.loc[policy_index], policy_scores, THRESHOLDS)
    admissible = policy.loc[policy['default_high'].le(RISK_LIMIT)]
    if admissible.empty:
        raise RuntimeError('На validation нет порога с верхней границей доли проблемных заявок до 5%')
    chosen = admissible.sort_values(['approved', 'threshold'], ascending=[False, True]).iloc[0]
    threshold = float(chosen['threshold'])
    test_target = target.loc[test_index]
    test_predictions = evaluate_test(models, test_target, {name: raw[test_index] for name, raw in scores.items()}, calibrators)
    test_scores = test_predictions[selected][models[selected]['calibration']]
    test_result = decision_table(test_target, test_scores, np.array([threshold])).iloc[0]

    reference_scores, current_scores = pd.Series(policy_scores), pd.Series(test_scores)
    score_bins = Bins.fit(reference_scores)
    score_psi = population_stability(score_bins.transform(reference_scores), score_bins.transform(current_scores))
    feature_psi = {
        name: population_stability(bins.transform(data.loc[policy_index, name]), bins.transform(data.loc[test_index, name]))
        for name, bins in encoder.bins.items()
    }

    calibration_plot = {}
    for name, method in [('До калибровки', 'raw'), ('После калибровки', 'sigmoid')]:
        probability = test_predictions[selected][method]
        observed, predicted = calibration_curve(test_target, probability, n_bins=10, strategy='quantile')
        calibration_plot[name] = [predicted.tolist(), observed.tolist()]

    result = {
        'counts': {'train': len(train_index), 'calibration': len(calibration_index), 'policy': len(policy_index), 'test': len(test_index)},
        'feature_count': len(data.columns),
        'test_base_rate': float(target.loc[test_index].mean()),
        'models': models,
        'selected': selected,
        'risk_limit': RISK_LIMIT,
        'validation_policy': chosen.to_dict(),
        'test_policy': test_result.to_dict(),
        'policy_curve': json.loads(policy.to_json(orient='records')),
        'score_psi': score_psi,
        'feature_psi': feature_psi,
        'calibration_curve': calibration_plot,
        'shap': explain_boosting(
            boosting, tree_features, test_index,
            policy_index[np.argmin(np.abs(probabilities[LIGHTGBM] - threshold))],
        ),
    }
    temporary = RESULTS.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(RESULTS)
    print(f'Результаты сохранены в {RESULTS}', flush=True)


if __name__ == '__main__':
    main()
