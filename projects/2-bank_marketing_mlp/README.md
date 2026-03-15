# bank_marketing_pytorch

## Goal

Реализовать первую нейросеть на PyTorch для табличной задачи бинарной классификации и понять полный цикл обучения модели вручную: от подготовки данных до train/validation loop и early stopping.

## Task Type

- binary classification
- tabular machine learning
- PyTorch fundamentals

## Dataset

Использован **Bank Marketing Dataset** (45,211 объектов, 16 признаков + целевая переменная `y`).

Задача: предсказать, подпишется ли клиент на срочный депозит (`yes` / `no`).

Особенности датасета:

- смешанные признаки: числовые и категориальные;
- пропусков нет;
- классы несбалансированы:
  - `no`: 39,922
  - `yes`: 5,289

## Approach

В проекте реализован полный baseline pipeline:

- анализ и предобработка данных;
- `StandardScaler` для числовых признаков;
- `OneHotEncoder` для категориальных признаков;
- `Dataset` и `DataLoader` на PyTorch;
- собственная `MLP` через `nn.Module`;
- `BCEWithLogitsLoss`;
- `Adam`;
- ручной `training loop` и `validation loop`;
- early stopping;
- эксперимент с `pos_weight` для борьбы с дисбалансом классов.

## Metrics

Оценивались:

- Accuracy
- Precision
- Recall
- F1-score

### Baseline MLP

Примерно:

- Accuracy: ~0.91
- Precision: ~0.61–0.67
- Recall: ~0.42–0.53
- F1: ~0.56–0.57

### MLP + weighted loss (`pos_weight`)

Примерно:

- Accuracy: ~0.82–0.87
- Precision: ~0.38–0.46
- Recall: ~0.84–0.91
- F1: ~0.53–0.59

## Main Results

Что удалось понять и реализовать:

- как табличные данные превращаются в тензоры;
- как работают `Dataset` и `DataLoader`;
- как устроены `forward`, `loss`, `backward`, `optimizer.step()`;
- почему `loss` и `F1` — не одно и то же;
- как дисбаланс классов меняет баланс между precision и recall;
- почему для табличных данных MLP может быть полезным baseline, но не всегда лучшим финальным решением.

## Main Takeaway

Проект выполнил свою учебную цель: дал практическое понимание того, как вручную обучается нейросеть в PyTorch.

Главный вывод: качество модели зависит не только от архитектуры, но и от данных, метрики, постановки задачи и способа работы с дисбалансом классов.

## Project Structure

- `notebooks/` — исследования и черновики
- `src/` — основной код
- `models/` — сохранённые веса модели
- `data/` — исходные и подготовленные данные
- `main.py` — запуск обучения

## How to run

1. Создать виртуальное окружение
2. Установить зависимости
3. Поместить `bank-full.csv` в `data/raw/`
4. Запустить:

```bash
python main.py
```
