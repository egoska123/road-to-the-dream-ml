- цель — **гибридный junior ML engineer → с уклоном в AI / NLP / RAG**;
- формат обучения — **через проекты**, с коротким вводом и сразу в практику;
- тебе важно **часто менять фокус**, особенно в начале;
- математика — **только через практику и интуицию**, без сухой отдельной базы;
- приоритет — **сильное портфолио + шанс на стажировку/оффер**.

По сути, твой маршрут должен опираться на четыре слоя:  
сначала прикладной classic ML через `scikit-learn` и метрики, затем базовый deep learning через `PyTorch`, потом NLP/Transformers через экосистему Hugging Face, и поверх этого — прикладные AI-системы вроде RAG, плюс базовая инженерка: трекинг экспериментов, API, Docker, нормальный GitHub. Это хорошо совпадает с тем, как сами официальные материалы раскладывают стек: у scikit-learn отдельно выделены supervised learning, model selection, threshold tuning и metrics; PyTorch beginner tutorials ведут от данных и модели к обучению и сохранению; Hugging Face курс и docs строятся вокруг Transformers/Datasets/Tokenizers; Google Rules of ML делает акцент именно на практическом ML engineering, а не на “сначала бесконечная теория”. ([scikit-learn.org](https://scikit-learn.org/stable/getting_started.html?utm_source=chatgpt.com "Getting Started — scikit-learn 1.8.0 documentation"))

**Junior ML Engineer / Junior AI Engineer (Applied AI),**  
где ты умеешь:

- взять данные,
- обучить базовую модель,
- сравнить решения,
- понять метрики,
- собрать inference,
- завернуть всё в API,
- сделать 1–2 сильных проекта по NLP / RAG / рекомендациям.

---

# Главная логика roadmap

У тебя будет не схема “3 недели одна тема”, а такая:

- в первые 3–4 недели — **короткие спринты по 2–3 дня**;
- дальше — **1 неделя = 1 основной фокус**;
- каждые 2 недели — контрольная точка;
- каждые 3–4 недели — один проект посильнее;
- математика встраивается только в момент, когда она реально нужна:
  - метрики — когда считаешь precision/recall/F1;
  - градиентный спуск — когда обучаешь первую нейросеть;
  - эмбеддинги и cosine similarity — когда делаешь RAG и рекомендации.

---

# Что у тебя должно быть к концу roadmap

К концу 3 месяцев у тебя должна быть не “теория в голове”, а вот такой комплект:

1. **Уверенная база по classic ML**
   - табличные данные,
   - пайплайн обработки,
   - baseline,
   - cross-validation,
   - метрики,
   - feature engineering,
   - гиперпараметры.

2. **База по deep learning**
   - PyTorch tensors,
   - Dataset / DataLoader,
   - training loop,
   - loss,
   - optimizer,
   - overfitting / regularization,
   - сохранение и загрузка модели. ([docs.pytorch.org](https://docs.pytorch.org/tutorials/index.html?utm_source=chatgpt.com "Welcome to PyTorch Tutorials"))

3. **База по NLP / Transformers**
   - токенизация,
   - эмбеддинги,
   - inference готовой модели,
   - fine-tuning маленькой модели,
   - сравнение classical NLP vs Transformer. ([Hugging Face](https://huggingface.co/learn/llm-course/chapter1/1?utm_source=chatgpt.com "Introduction - Hugging Face LLM Course"))

4. **База по RAG**
   - чанкинг,
   - embeddings,
   - retrieval,
   - vector store,
   - basic evaluation,
   - почему плохой retrieval ломает хороший ответ. Официальные RAG-гайды как раз описывают RAG как связку retrieval + generation поверх внешних данных и vector stores / semantic search. ([OpenAI для разработчиков](https://developers.openai.com/api/docs/guides/retrieval/?utm_source=chatgpt.com "Retrieval | OpenAI API"))

5. **База по инженерке**
   - GitHub-репозитории с README,
   - FastAPI для inference,
   - Docker хотя бы для 1–2 проектов,
   - базовый experiment tracking через MLflow. MLflow Tracking как раз предназначен для логирования параметров, метрик, версий кода и артефактов. README на GitHub нужен, чтобы объяснить, зачем проект нужен и как его запускать. ([MLflow AI Platform](https://mlflow.org/docs/latest/ml/tracking/quickstart/?utm_source=chatgpt.com "MLflow Tracking Quickstart"))

6. **Портфолио**
   - 4–5 публичных реп,
   - 1 сильный flagship-проект,
   - GitHub-профиль с понятным описанием,
   - желательно Hugging Face / демо / скринкасты.

---

# Сроки честно

**За 1 месяц** ты можешь очень сильно стартануть и собрать уже вменяемую базу + 3–4 мини/средних проекта.  
Но **для стажировки/оффера** намного реалистичнее смотреть на **2.5–3 месяца**, особенно если нужен не просто “я что-то запускал”, а **нормальный GitHub и сильные прикладные проекты**.

То есть:

- **1 месяц** — aggressive launch;
- **3 месяца** — realistic junior-ready track.

---

# Твой roadmap

## Этап 0. Подготовка среды — 2 дня

Цель — чтобы дальше ты не сливал энергию на хаос.

Сделай:

- один основной GitHub-репозиторий `ml-roadmap-notes` или `ml-playground`;
- отдельную папку `projects/`;
- шаблон для каждого проекта:
  - `README.md`
  - `notebooks/`
  - `src/`
  - `requirements.txt`
  - `results/`
- установи:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `jupyter`
  - `pytorch`
  - `transformers`
  - `datasets`
  - `sentence-transformers`
  - `fastapi`
  - `uvicorn`
  - `mlflow`
  - `docker` локально, если ещё не готов.  
     Официальные туториалы FastAPI и Docker действительно удобны именно как быстрый старт, без лишнего ухода в инфраструктуру. ([fastapi.tiangolo.com](https://fastapi.tiangolo.com/tutorial/?utm_source=chatgpt.com "Tutorial - User Guide"))

---

## Месяц 1 — быстрый боевой вход

## Неделя 1. Classic ML → сразу потом первая нейросеть

### Дни 1–2: mini-project 1

**Проект:** простая классификация на табличных данных.  
Примеры:

- loan default,
- churn,
- spam/ham tabular,
- fraud-like classification.

Что делаешь:

- загрузка данных,
- быстрый EDA,
- train/valid/test split,
- baseline: Logistic Regression,
- потом RandomForest / GradientBoosting / CatBoost, если захочешь,
- метрики: accuracy, precision, recall, F1, ROC-AUC,
- confusion matrix.

Что должен понять:

- что такое baseline,
- почему accuracy может врать,
- чем отличаются precision и recall,
- как выглядит утечка данных.

**Мини-математика через практику:**  
матрица ошибок, вероятности классов, threshold.

### Дни 3–4: mini-project 2

**Тот же датасет**, но теперь:

- пайплайн почище,
- обработка пропусков,
- кодирование категорий,
- scaling,
- cross-validation,
- подбор гиперпараметров без фанатизма.

Что должен вынести:

- ML — это не “накидал модель”, а “собрал воспроизводимый pipeline”.  
   Это как раз соответствует практикам scikit-learn по model selection и evaluation. ([scikit-learn.org](https://scikit-learn.org/stable/getting_started.html?utm_source=chatgpt.com "Getting Started — scikit-learn 1.8.0 documentation"))

### Дни 5–6: mini-project 3

**Первая нейросеть на PyTorch**  
Не лезем сразу в сложное NLP. Берём:

- либо тот же табличный датасет,
- либо очень простой текстовый task.

Что делаешь:

- tensor,
- Dataset / DataLoader,
- простая `nn.Module`,
- loss,
- optimizer,
- training loop,
- validation loop,
- early stopping руками.

Что должен понять:

- forward,
- backward,
- что такое loss,
- почему модель переобучается,
- чем отличается “обучил sklearn” от “сам написал цикл обучения”.  
   Это полностью в духе beginner tutorials PyTorch. ([docs.pytorch.org](https://docs.pytorch.org/tutorials/index.html?utm_source=chatgpt.com "Welcome to PyTorch Tutorials"))

### День 7: контрольная точка 1

У тебя должно быть:

- 2 небольших репозитория или 1 репо с двумя версиями,
- README,
- скрин метрик,
- короткий вывод: что лучше и почему.

---

## Неделя 2. NLP: classic → transformers

### Дни 1–2: mini-project 4

**Classical NLP**  
Пример:

- классификация отзывов,
- токсичность,
- спам,
- тональность.

Что делаешь:

- чистка текста,
- TF-IDF,
- Logistic Regression / LinearSVC,
- сравнение метрик.

Что должен понять:

- тексты тоже можно решать без нейросетей,
- baseline в NLP обязателен.

### Дни 3–4: mini-project 5

**Transformers / Hugging Face**  
Берёшь ту же задачу, но теперь:

- токенизатор,
- готовая предобученная модель,
- inference,
- потом маленький fine-tuning.

Лучше брать маленькую модель уровня `distilbert` или похожую, чтобы не умереть по железу.

Что должен понять:

- что делает токенизатор,
- что такое attention в практическом смысле,
- почему embeddings полезнее “мешка слов”,
- когда Transformer реально выигрывает у TF-IDF, а когда нет.  
   Экосистема Hugging Face именно под это и заточена: Transformers, Datasets, Tokenizers, Accelerate, Hub. ([Hugging Face](https://huggingface.co/learn/llm-course/chapter1/1?utm_source=chatgpt.com "Introduction - Hugging Face LLM Course"))

### Дни 5–6: mini-project 6

**Сравнение classic NLP vs transformer**  
Это очень сильный формат для портфолио:

- одна и та же задача,
- два подхода,
- честное сравнение качества, скорости, сложности.

### День 7: контрольная точка 2

У тебя уже есть:

- classic ML проект,
- PyTorch mini DL проект,
- classic NLP проект,
- transformer NLP mini-project.

На этом этапе ты уже не “совсем новичок”, а человек, который реально **потрогал несколько парадигм**.

---

## Неделя 3. Вход в RAG

Официальные гайды по retrieval описывают основу RAG так: у тебя есть внешний корпус данных, ты строишь индекс/векторное представление, находишь релевантный контекст, и только потом генерируешь ответ. То есть RAG — это не “магия LLM”, а pipeline из подготовки данных, retrieval и generation. ([OpenAI для разработчиков](https://developers.openai.com/api/docs/guides/retrieval/?utm_source=chatgpt.com "Retrieval | OpenAI API"))

### Дни 1–2: mini-project 7

**Самый простой RAG**  
Например:

- FAQ-бот по документам,
- помощник по PDF/статьям,
- ассистент по правилам/инструкциям.

Что делаешь:

- загружаешь документы,
- режешь на chunks,
- строишь embeddings,
- делаешь retrieval,
- отправляешь найденный контекст в LLM.

### Дни 3–4: mini-project 8

**Делаешь RAG лучше**  
Добавляешь:

- разные chunk sizes,
- overlap,
- top-k,
- сравнение retrieval-стратегий,
- хранение metadata.

### Дни 5–6: mini-project 9

**Оценка RAG**  
Смотришь не только “ответ вроде норм”, а:

- когда retrieval плохой,
- когда контекст нерелевантный,
- какие типы вопросов проваливаются,
- где hallucination из-за плохого контекста.

### День 7: контрольная точка 3

У тебя уже должен быть **один работающий RAG-репозиторий**.

---

## Неделя 4. Рекомендации + первая продуктовая обвязка

Рекомендательные системы полезны тебе по двум причинам:

1. это востребованная прикладная область;
2. там естественно соединяются embeddings, retrieval и ranking.

Официальные материалы TensorFlow Recommenders как раз разделяют recommendation pipeline на **retrieval** и **ranking**: сначала отбираются кандидаты, потом они ранжируются. ([TensorFlow](https://www.tensorflow.org/recommenders?utm_source=chatgpt.com "TensorFlow Recommenders"))

### Дни 1–2: mini-project 10

**Классическая рекомендация**

- MovieLens или похожий датасет,
- item-item similarity,
- user-item matrix,
- ALS / matrix factorization / implicit feedback.

### Дни 3–4: mini-project 11

**Чуть глубже**  
Не обязательно строить монстра. Достаточно понять:

- retrieval,
- ranking,
- cold start,
- implicit feedback.

### Дни 5–6: mini-project 12

**Продуктовая упаковка одного проекта**  
Берёшь лучший проект из предыдущих недель и добавляешь:

- FastAPI endpoint,
- `/predict` или `/ask`,
- Dockerfile,
- инструкции в README,
- пример запроса.  
   FastAPI даёт пошаговый туториал по API, а Docker — быстрый путь к контейнеризации приложения. ([fastapi.tiangolo.com](https://fastapi.tiangolo.com/tutorial/?utm_source=chatgpt.com "Tutorial - User Guide"))

### День 7: итог месяца 1

К этому моменту у тебя должно быть примерно:

- 4–6 мини-проектов,
- 1 проект с API,
- 1 RAG,
- 1 NLP,
- 1 классический ML,
- базовое понимание, что тебе нравится больше.

---

# После месяца 1 — развилка

Вот здесь ты делаешь выбор.

## Ветка A — тебе больше заходит NLP / RAG / LLM

Тогда месяцы 2–3 строим вокруг:

- text classification / retrieval,
- embeddings,
- RAG,
- evaluation,
- AI backend,
- maybe lightweight fine-tuning.

## Ветка B — тебе зашли рекомендации

Тогда месяцы 2–3:

- collaborative filtering,
- ranking,
- retrieval,
- embeddings,
- hybrid recommender,
- API / serving.

## Ветка C — ты пока хочешь гибрид

Это для тебя сейчас самый правильный вариант.

Тогда:

- 50% времени — NLP/RAG,
- 25% — классический ML / risk scoring,
- 25% — recsys / serving / infra.

---

# Месяц 2 — превращаем хаотичную практику в системный навык

## Неделя 5. Сильный tabular ML проект

**Тема:** risk scoring / churn / default / anomaly-lite

Что добавляешь:

- хороший split,
- pipeline,
- feature engineering,
- error analysis,
- MLflow tracking.  
   MLflow Tracking специально предназначен для логирования параметров, метрик, версий и артефактов, поэтому его очень хорошо встраивать уже на этом этапе. ([MLflow AI Platform](https://mlflow.org/docs/latest/ml/tracking/quickstart/?utm_source=chatgpt.com "MLflow Tracking Quickstart"))

Результат:

- не просто ноутбук,
- а уже почти “рабочий ML-проект”.

## Неделя 6. Сильный NLP проект

**Тема:** классификация текстов / intent detection / moderation

Что добавляешь:

- сравнение 3 подходов,
- honest baseline,
- fine-tuning маленькой модели,
- inference script,
- README с примерами.

## Неделя 7. RAG v2

**Тема:** не “чатик по PDF”, а уже нормальный прикладной кейс

Например:

- assistant for scientific papers,
- policy assistant,
- Q&A по документации,
- knowledge assistant for domain docs.

Что добавляешь:

- лучшее индексирование,
- metadata filtering,
- нормальные eval examples,
- список failure cases.

## Неделя 8. Deployment + cleanup week

Что делаешь:

- один ML-проект → FastAPI,
- один RAG-проект → FastAPI,
- Dockerize хотя бы один из них,
- почистить репозитории,
- добавить нормальные README,
- pinned repos на GitHub.  
   GitHub docs отдельно подчёркивают, что README нужен, чтобы объяснить, чем проект полезен и как им пользоваться. ([GitHub Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes?utm_source=chatgpt.com "About the repository README file"))

---

# Месяц 3 — выход на уровень “это уже можно показывать работодателю”

## Неделя 9–10. Flagship project

Тут уже не мини-проект, а **1 сильный проект**.

Требования к нему:

- прикладная проблема,
- данные/документы не игрушечные,
- baseline или сравнение подходов,
- метрики или eval,
- API,
- README,
- демо или скринкаст,
- структура проекта в `src/`,
- понятный стек.

## Неделя 11. Симуляция работы в команде

Даже если ты один, делай так:

- issues в GitHub,
- TODO board,
- ветки `feature/...`,
- pull request самому себе,
- changelog,
- фикс багов,
- refactor.

Это сильно добавляет ощущения “не просто ноутбук, а инженерный проект”.

## Неделя 12. Упаковка под рынок

Что должно быть готово:

- GitHub profile,
- 4–5 лучших pinned repos,
- короткое резюме,
- шаблон рассказа о каждом проекте,
- ответы на вопросы:
  - почему такая метрика,
  - какой baseline,
  - где переобучение,
  - как валидировал,
  - почему такой retrieval,
  - что бы улучшил дальше.

---

# Какой именно набор проектов я бы тебе рекомендовал

Вот мой лучший набор под тебя.

## Обязательные 5 проектов

### 1. Tabular ML baseline project

Например:

- loan default / churn / fraud-lite / risk scoring.

Зачем:

- база ML,
- метрики,
- pipeline,
- interview-friendly.

### 2. Classical NLP vs Transformer

Одна задача, два подхода.

Зачем:

- показывает зрелость,
- видно, что ты умеешь сравнивать решения, а не просто “запустил модную модель”.

### 3. RAG assistant

Но не тупой “пдф-бот”, а с eval и failure analysis.

Зачем:

- это прямо в твою цель,
- это современно,
- это пересекается с AI engineer треком. ([OpenAI для разработчиков](https://developers.openai.com/api/docs/guides/retrieval/?utm_source=chatgpt.com "Retrieval | OpenAI API"))

### 4. Recommendation mini-system

Например:

- movie/music recommendation,
- hybrid recommender,
- content + collaborative logic.

Зачем:

- редкая и сильная штука для джуна,
- показывает понимание retrieval/ranking. ([TensorFlow](https://www.tensorflow.org/recommenders?utm_source=chatgpt.com "TensorFlow Recommenders"))

### 5. Flagship end-to-end project

Один большой проект с API и упаковкой.

---

# Лучшие варианты flagship-проекта для тебя

Вот что реально может смотреться сильно и не слишком стандартно.

## Вариант 1. Research Copilot / Paper Assistant

Что делает:

- принимает статьи, заметки, тезисы,
- строит retrieval,
- отвечает по источникам,
- помогает сравнивать papers,
- делает summary + grounded answers.

Сильные стороны:

- необычно,
- очень в твою сторону,
- хорошо смотрится как AI project.

## Вариант 2. Smart Support RAG

Что делает:

- отвечает только на основе базы знаний,
- показывает источники,
- умеет различать “нашёл контекст / не нашёл”.

Сильные стороны:

- ближе к вакансиям,
- легко объяснить бизнес-ценность.

## Вариант 3. Hybrid Recommender

Что делает:

- рекомендации по взаимодействиям + по текстовому описанию объектов,
- решает cold start лучше, чем чистая коллаборативка.

Сильные стороны:

- сильная инженерная задача,
- редкий проект для джуна.

## Вариант 4. Risk Scoring + Explainability API

Что делает:

- скоринг риска,
- объяснение предсказания,
- API для скоринга.

Сильные стороны:

- очень близко к бизнесу и классическому ML,
- хорошо для собеседований.

**Мой совет:**  
для тебя сейчас самый сильный выбор —  
**либо Research Copilot, либо Hybrid Recommender, либо Smart Support RAG.**

---

# Что тебе пока НЕ нужно

Чтобы не слить время, в ближайшие 2–3 месяца **не делай упор** на это:

- длинные курсы по чистой математике;
- глубокую статистику ради статистики;
- CNN/CV, если это не твой фокус;
- обучение больших моделей с нуля;
- сложный MLOps раньше времени;
- распределённое обучение;
- долгие соревнования ради leaderboard.

Тебе сейчас нужна не “энциклопедия”, а **быстрый рост через прикладные циклы**.

---

# Как учиться каждый день, чтобы не выгореть

Под тебя я бы поставил такой ритм:

**Формат дня**

- 30–45 мин — короткий ввод: что делаем и зачем;
- 3–4 часа — основная практика;
- 1–2 часа — доделка/эксперименты;
- 30 мин — записать выводы и обновить README.

**Формат спринта**

- первые 4 недели: менять фокус каждые 2–3 дня;
- потом: 1 неделя = 1 фокус;
- каждые 14 дней: ревизия, что оставляем в портфолио.

---

# Что считать успехом на каждом этапе

## Через 2 недели

Ты уже:

- умеешь обучить и оценить классическую модель;
- умеешь написать базовый training loop в PyTorch;
- делал classic NLP и transformer NLP.

## Через 1 месяц

Ты уже:

- не путаешь classic ML, DL, NLP и RAG;
- сделал несколько проектов;
- понимаешь, что тебе ближе;
- можешь показать GitHub.

## Через 2 месяца

Ты уже:

- умеешь делать проект не только в ноутбуке,
- умеешь заворачивать inference в API,
- умеешь трекать эксперименты,
- имеешь 2–3 сильных проекта.

## Через 3 месяца

Ты уже:

- выглядишь как junior ML/AI engineer с прикладным уклоном,
- можешь идти на стажировки,
- можешь внятно рассказывать про свои решения,
- имеешь flagship-project.

---

# Самое важное

Твой маршрут не должен быть “сначала полностью classic ML, потом через полгода нейросети”.  
Для тебя лучше работает схема:

**classic ML → DL → NLP → RAG → recsys → обратно к ML engineering.**

То есть ты всё время чередуешь подходы, но не хаотично, а так, чтобы каждый следующий блок опирался на предыдущий:

- baseline и метрики из classic ML;
- training loop и переобучение из DL;
- embeddings и transformers из NLP;
- retrieval и grounding из RAG;
- candidate retrieval/ranking из recsys;
- API/MLflow/Docker из engineering. ([scikit-learn.org](https://scikit-learn.org/stable/user_guide.html?utm_source=chatgpt.com "User Guide — scikit-learn 1.8.0 documentation"))

Дальше логичный следующий шаг — я могу на основе этого roadmap составить тебе **конкретный план на первые 14 дней по дням**, с проектами, датасетами, стеком и ожидаемым результатом каждого дня.
