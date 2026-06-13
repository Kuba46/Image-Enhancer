# Image Enhancer

Система улучшения изображений в браузере на базе ML-модели. Модель обучается на датасете MIT-Adobe FiveK и предсказывает параметры коррекции яркости, контраста и насыщенности. Инференс работает локально в Web Worker — изображение не покидает устройство пользователя.

Разработано в контексте VK Education.

---

## Структура проекта

```
Image-Enhancer/
├── train/                          # Python: обучение модели
│   ├── train.py                    # обучение MLP + экспорт весов
│   ├── evaluate.py                 # оценка качества модели
│   ├── requirements.txt            # зависимости Python
│   └── model/                      # создаётся после обучения
│       ├── tfjs/
│       │   ├── model.json          # архитектура + манифест весов
│       │   └── group1-shard1of1.bin
│       ├── scaler.json             # параметры нормализации признаков
│       └── best_keras.keras        # чекпоинт Keras
│
└── web/                            # браузерная часть
    ├── index.html
    ├── style.css
    ├── app.js                      # точка входа
    ├── enhancer.js                 # публичный API модуля
    ├── worker.js                   # Web Worker: инференс + обработка пикселей
    ├── model-loader.js             # загрузка весов модели
    ├── model/                      # сюда копируется обученная модель
    │   ├── model.json
    │   ├── group1-shard1of1.bin
    │   └── scaler.json
    └── ui/
        ├── dom.js                  # ссылки на DOM-элементы
        ├── dropzone.js             # drag & drop, валидация файла
        ├── progress.js             # прогресс-бар
        └── result.js               # отображение результата
```

---

## Быстрый старт

### Браузерная часть (без модели)

Без обученной модели страница работает в demo-режиме.

```bash
cd web
npx serve .
# открыть http://localhost:3000
```

ES-модули и Web Workers не работают через `file://` — нужен HTTP-сервер.

### Полный цикл с обученной моделью

```bash
# 1. Подготовить окружение
cd train
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Подготовить данные (см. раздел про датасет)

# 3. Обучить модель
python train.py

# 4. Скопировать модель в браузерную часть
cp model/tfjs/model.json           ../web/model/
cp model/tfjs/group1-shard1of1.bin ../web/model/
cp model/scaler.json               ../web/model/

# 5. Запустить
cd ../web && npx serve .
```

---

## Датасет MIT-Adobe FiveK

Используется Kaggle-версия датасета с уже конвертированными JPG:
https://www.kaggle.com/datasets/weipengzhang/adobe-fivek

Структура после скачивания:

```
adobe-fivek/
├── raw/   ← 5000 исходных изображений
├── a/     ← 5000 эталонов Expert A
├── b/     ← 5000 эталонов Expert B
├── c/     ← 5000 эталонов Expert C  ← используется для обучения
├── d/     ← 5000 эталонов Expert D
└── e/     ← 5000 эталонов Expert E
```

Имена файлов совпадают во всех папках. `train.py` использует только папку `c/` (Expert C) — стандартный бенчмарк для FiveK.

Скопировать в проект:

```bash
cp -r /путь/к/adobe-fivek/raw train/data/raw
cp -r /путь/к/adobe-fivek/c   train/data/c
```

---

## Обучение модели

### train.py

```bash
python train.py
```

Pipeline:
1. Загружает пары `data/raw/` + `data/c/` (5000 пар)
2. Кэширует признаки и статистики raw-изображений — читается один раз
3. Для каждой пары вычисляет дельты между исходником и эталоном Expert C: `brightness_delta`, `contrast_delta`, `saturation_delta` ∈ [-1, 1]
4. Для каждого исходника извлекает **35 признаков**:
   - гистограмма яркости (16 бинов)
   - статистики яркости: mean, std, median
   - доля тёмных (<25%) и светлых (>75%) пикселей
   - насыщенность HSV: mean, std, value
   - средние каналы RGB и цветовой баланс (R-B, R-G, G-B)
   - перцентили яркости: p5, p25, p75, p95
   - энтропия гистограммы
5. Обучает MLP: `Input(35) → Dense(128, relu) → BN → Dense(64, relu) → BN → Dense(32, relu) → Dense(3, tanh)`
6. Сохраняет лучший чекпоинт в `model/best_keras.keras`
7. Экспортирует веса в `model/tfjs/` напрямую через numpy — без `tensorflowjs`

Настройки в `train.py`:

```python
EXPERT_DIRS = [Path("data/c")]  # только Expert C
EPOCHS      = 100               # EarlyStopping остановит раньше
BATCH_SIZE  = 64
```

Ожидаемый вывод:

```
Исходников (raw): 5000
Кэширование признаков raw-изображений...
  c/: 5000 пар ✓  mean=[-0.07, +0.05, -0.06]  (brightness, contrast, saturation)
Всего пар: 5000
```

### evaluate.py

```bash
python evaluate.py --n 200
```

Оценивает модель на N изображениях из `data/raw/` против эталонов `data/c/`. Выводит MAE, корреляцию и std ошибки по каждому параметру, средний PSNR. Сохраняет scatter plot `evaluation_scatter.png`.

Ориентиры:

| Метрика | Хорошо | Отлично |
|---|---|---|
| MAE | < 0.12 | < 0.08 |
| PSNR | > 30 дБ | > 35 дБ |

Опции:
- `--n` — количество фото (default: 200)
- `--raw` — папка исходников (default: `data/raw`)
- `--expert` — папка эталонов (default: `data/c`)
- `--scaler` — файл нормализации (default: `model/scaler.json`)

---

## Браузерный API

```js
import { ImageEnhancer, TaskStatus } from './enhancer.js';

const enhancer = new ImageEnhancer();
await enhancer.init({
  workerUrl: './worker.js',
  modelUrl:  './model/model.json',
  scalerUrl: './model/scaler.json',
});

// Поставить задачу
const taskId = await enhancer.enqueue(file); // File | Blob | ImageData

// Подписаться на события
enhancer.addEventListener('taskchange', (e) => {
  const { taskId, status, progress, params } = e.detail;
  // status: 'queued' | 'processing' | 'done' | 'error' | 'cancelled'
  // params: { brightness, contrast, saturation } — при status === 'done'
});

// Опросить статус
const { status, progress } = enhancer.getStatus(taskId);

// Отменить
await enhancer.cancel(taskId);

// Получить результат
const blob = await enhancer.getResult(taskId); // Promise<Blob>
```

Параметры коррекции в диапазоне `[-1, 1]`:

| Параметр | -1 | 0 | +1 |
|---|---|---|---|
| brightness | затемнить | без изменений | осветлить |
| contrast | снизить | без изменений | повысить |
| saturation | обесцветить | без изменений | насытить |

---

## Технические характеристики

| Параметр | Значение |
|---|---|
| Максимальный размер изображения | 15 Мпк (автомасштаб) |
| Среднее время обработки | ~5 с |
| Максимальное время обработки | 30 с |
| Поддерживаемые форматы | JPG, PNG, HEIC, BMP |
| Блокировка UI | нет (Web Worker) |
| Размер кода | ~1.4 МБ |
| Размер модели | ~60 КБ |

### Совместимость браузеров

| Браузер | Минимальная версия |
|---|---|
| Chrome | 80+ |
| Edge | 80+ |
| Firefox | 79+ |
| Safari | 15+ |

---

## Устранение проблем

**Страница не работает при открытии через Finder**
Нужен HTTP-сервер: `cd web && npx serve .`

**Конфликт зависимостей tensorflowjs**
`tensorflowjs` не используется в проекте — удалить:
```bash
pip uninstall tensorflowjs tensorflow-decision-forests yggdrasil-decision-forests -y
```

**Чёрное изображение на выходе**
Веса модели загрузились некорректно. Убедиться что `model.json` содержит поле `layersMeta` только с Dense-слоями (без batch_normalization):
```js
fetch('./model/model.json').then(r=>r.json()).then(d=>console.log(d.layersMeta))
// Должно быть 4 объекта: dense, dense_1, dense_2, dense_3
```

**NaN в параметрах коррекции**
Несоответствие количества признаков между `train.py` и `worker.js`. Убедиться что оба файла используют 35 признаков (не 30).

**Модель не загружается в браузере**
Проверить наличие всех трёх файлов:
```bash
ls web/model/
# model.json  group1-shard1of1.bin  scaler.json
```

**Модель не улучшает игровые скриншоты**
Ожидаемо — модель обучена на фотографиях с камеры (MIT-Adobe FiveK). Для скриншотов с намеренно тёмной художественной гаммой коррекция будет минимальной или нулевой.

**HEIC не открывается в Chrome/Firefox**
`heic2any` загружается с CDN — нужен интернет. Для офлайн-работы:
```bash
cd web
curl -L https://cdn.jsdelivr.net/npm/heic2any@0.0.4/dist/heic2any.min.js -o heic2any.min.js
```