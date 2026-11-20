import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from scoring.data import ROOT, PROCESSED


REPORTS = ROOT / 'reports'
BLUE = '#2354ac'
TEAL = '#007b70'
RED = '#ad4a35'
GRAY = '#697589'


def quality_chart(results):
    fig, ((roc, calibration), (policy, stability)) = plt.subplots(2, 2, figsize=(12, 8.8), layout='constrained')
    fig.suptitle('Кредитный скоринг Home Credit', fontsize=17, fontweight='bold')
    roc.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1)
    for color, (name, model) in zip((GRAY, BLUE), results['models'].items()):
        auc = model['test'][model['calibration']]['roc_auc']
        roc.plot(*model['roc'], color=color, label=f'{name} · {auc:.3f}')
    roc.set(title='ROC на тестовой выборке', xlabel='Доля ложных срабатываний', ylabel='Доля найденных TARGET = 1')
    roc.legend(loc='lower right', frameon=False)

    calibration.plot([0, 0.4], [0, 0.4], '--', color=GRAY, linewidth=1)
    for color, (name, curve) in zip((GRAY, BLUE), results['calibration_curve'].items()):
        calibration.plot(*curve, 'o-', label=name, color=color, markersize=4)
    calibration.set(title=f'Калибровка на тесте · {results["selected"]}', xlabel='Средний прогноз', ylabel='Наблюдаемая доля TARGET = 1')
    calibration.xaxis.set_major_formatter(PercentFormatter(1))
    calibration.yaxis.set_major_formatter(PercentFormatter(1))
    calibration.legend(loc='upper left', frameon=False)

    sweep = pd.DataFrame(results['policy_curve']).dropna(subset=['default_rate'])
    policy.plot(sweep['approval_rate'], sweep['default_rate'], color=BLUE, label='Доля TARGET = 1 среди одобренных')
    policy.plot(sweep['approval_rate'], sweep['default_high'], '--', color=GRAY, label='Верхняя граница 95% ДИ')
    policy.axhline(results['risk_limit'], color=RED, linestyle=':', label='Лимит 5%')
    chosen = results['validation_policy']
    policy.scatter(chosen['approval_rate'], chosen['default_rate'], color=BLUE, edgecolor='white', s=75, zorder=3)
    policy.set(title='Компромисс при выборе порога · validation', xlabel='Доля одобренных заявок', ylabel='Доля TARGET = 1', xlim=(0.02, 1.01), ylim=(0, 0.10))
    policy.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    policy.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    policy.legend(loc='upper left', frameon=False, fontsize=9)

    top_psi = pd.Series(results['feature_psi']).nlargest(5)
    top_psi.loc['Скор модели'] = results['score_psi']
    top_psi = top_psi.sort_values()
    stability.barh(top_psi.index, top_psi.values, color=[BLUE if name == 'Скор модели' else TEAL for name in top_psi.index])
    stability.set(title='PSI · случайные validation и test', xlabel='PSI')
    for axis in fig.axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y' if axis is not stability else 'x', color='#dde4ec', linewidth=0.6)
        axis.set_axisbelow(True)
    fig.savefig(REPORTS / 'quality.png', dpi=140)
    plt.close(fig)


def shap_chart(results):
    shap = results['shap']
    importance = pd.Series(shap['importance']).nlargest(10).sort_values()
    local = pd.Series(shap['case_contributions']).sort_values()
    fig, (global_plot, example) = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    fig.suptitle('Интерпретация LightGBM · TreeSHAP', fontsize=17, fontweight='bold')
    global_plot.barh(importance.index, importance, color=BLUE)
    global_plot.set(title=f'Важность признаков · {shap["sample_size"]} заявок из test', xlabel='Средний модуль SHAP в логарифме шансов')
    example.barh(local.index, local, color=[RED if value > 0 else TEAL for value in local])
    example.axvline(0, color=GRAY, linewidth=1)
    example.set(title='Пример заявки из validation рядом с порогом', xlabel='Вклад в логарифм шансов проблем с платежами')
    for axis in fig.axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='x', color='#dde4ec', linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis='y', labelsize=9)
    fig.savefig(REPORTS / 'shap.png', dpi=140)
    plt.close(fig)


def threshold_dashboard(results):
    template = ROOT / 'scoring' / 'threshold.html'
    dashboard = {key: results[key] for key in ('selected', 'risk_limit', 'policy_curve', 'validation_policy', 'test_policy')}
    dashboard['policy_curve'] = [
        [row['approval_rate'], row['default_rate'], row['default_high'], row['good_rejected']]
        for row in dashboard['policy_curve']
    ]
    data = json.dumps(dashboard, ensure_ascii=False, separators=(',', ':'))
    (REPORTS / 'index.html').write_text(template.read_text(encoding='utf-8').replace('__RESULTS__', data), encoding='utf-8')


def main():
    results = json.loads((PROCESSED / 'scoring_results.json').read_text(encoding='utf-8'))
    REPORTS.mkdir(exist_ok=True)
    quality_chart(results)
    shap_chart(results)
    threshold_dashboard(results)
    print(f'Графики и интерактивный отчёт сохранены в {REPORTS}')


if __name__ == '__main__':
    main()
