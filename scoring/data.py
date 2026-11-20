from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
SEED = 42
ID = 'SK_ID_CURR'
TARGET = 'TARGET'
SPLIT = 'SPLIT'


def load_data():
    sources = {
        'applications': RAW / 'application_train.csv',
        'bureau': RAW / 'bureau.csv',
        'previous': RAW / 'previous_application.csv',
        'payments': RAW / 'installments_payments.csv',
    }
    missing = [path.name for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Добавьте в {RAW} следующие файлы: {", ".join(missing)}')

    PROCESSED.mkdir(parents=True, exist_ok=True)
    sql = Path(__file__).with_name('features.sql').read_text(encoding='utf-8')
    with duckdb.connect(config={'threads': '2', 'memory_limit': '512MB'}) as connection:
        connection.execute('SET temp_directory = ?', [str(PROCESSED / 'duckdb_tmp')])
        connection.execute('SET enable_progress_bar = false')
        applications = connection.execute(sql, {key: str(path) for key, path in sources.items()}).df()

    split_path = PROCESSED / 'application_split.csv'
    if not split_path.is_file():
        create_split(sources['applications'], split_path)
    splits = pd.read_csv(split_path, usecols=[ID, SPLIT])
    data = applications.merge(splits, on=ID, how='outer', validate='one_to_one', indicator=True)
    if data[ID].isna().any() or not data['_merge'].eq('both').all():
        raise ValueError('В исходных заявках и файле разбиения должны совпадать все идентификаторы')
    if data[SPLIT].isna().any() or set(data[SPLIT]) != {'train', 'val', 'test'}:
        raise ValueError('Файл разбиения должен содержать метки train, val и test')
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError('TARGET должен содержать только 0 и 1')
    return data.drop(columns=[ID, '_merge'])


def create_split(source, destination):
    # Совпадает с разбиением в notebooks/03_validation_split.ipynb.
    labels = pd.read_csv(source, usecols=[ID, TARGET])
    train, remaining = train_test_split(
        labels, test_size=0.4, stratify=labels[TARGET], random_state=SEED,
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, stratify=remaining[TARGET], random_state=SEED,
    )
    splits = pd.concat([
        train[[ID]].assign(SPLIT='train'),
        validation[[ID]].assign(SPLIT='val'),
        test[[ID]].assign(SPLIT='test'),
    ])
    temporary = destination.with_suffix('.tmp')
    splits.to_csv(temporary, index=False)
    temporary.replace(destination)
