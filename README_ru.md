# Все техники RAG: Простой практический подход ✨

## Переводы
- [English version](README.md) (оригинал)
- Другие языки:
[Deutsch](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=de) | [Español](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=es) | [Français](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=fr) | [日本語](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=ja) | [한국어](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=ko) | [Português](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=pt) | [中文](https://www.readme-i18n.com/FareedKhan-dev/all-rag-techniques?lang=zh)

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![Nebius AI](https://img.shields.io/badge/Nebius%20AI-API-brightgreen)](https://cloud.nebius.ai/services/llm-embedding) [![OpenAI](https://img.shields.io/badge/OpenAI-API-lightgrey)](https://openai.com/) [![Medium](https://img.shields.io/badge/Medium-Blog-black?logo=medium)](https://medium.com/@fareedkhandev/testing-every-rag-technique-to-find-the-best-094d166af27f)

Этот репозиторий предлагает понятный, практический подход к **Retrieval-Augmented Generation (RAG)**, разбивая сложные техники на простые для понимания реализации. Вместо использования фреймворков вроде `LangChain` или `FAISS`, здесь всё построено на знакомых Python-библиотеках: `openai`, `numpy`, `matplotlib` и некоторых других.

Цель проста: предоставить читаемый, модифицируемый и обучающий код. Фокусируясь на основах, этот проект помогает разобраться в RAG и понять, как он действительно работает.

## Обновления: 📢

- (12-Май-2025) Добавлен новый ноутбук по работе с большими данными с использованием графов знаний.
- (27-Апреля-2025) Добавлен ноутбук для поиска лучшей техники RAG для конкретного запроса (Simple RAG + Reranker + Query Rewrite).
- (20-Марта-2025) Добавлен новый ноутбук по RAG с обучением с подкреплением.
- (07-Марта-2025) Добавлено 20 техник RAG в репозиторий.

## 🚀 Что внутри?

Репозиторий содержит коллекцию Jupyter ноутбуков, каждый из которых посвящён конкретной технике RAG. Каждый ноутбук предоставляет:

- Краткое объяснение техники
- Пошаговую реализацию с нуля
- Чёткие примеры кода с комментариями
- Оценки и сравнения для демонстрации эффективности
- Визуализации результатов

Вот обзор покрываемых техник:

| Ноутбук                                      | Описание                                                                 |
| :-------------------------------------------- | :---------------------------------------------------------------------- |
| [1. Простая RAG](01_simple_rag.ipynb) | Базовая реализация RAG. Отличная отправная точка! |
| [2. Семантический чанкинг](02_semantic_chunking.ipynb) | Разделяет текст по семантической схожести для более осмысленных чанков. |
| [3. Выбор размера чанка](03_chunk_size_selector.ipynb) | Исследует влияние разных размеров чанков на эффективность поиска. |
| [4. RAG с расширенным контекстом](04_context_enriched_rag.ipynb) | Извлекает соседние чанки для предоставления большего контекста. |
| [5. Контекстуальные заголовки чанков](05_contextual_chunk_headers_rag.ipynb) | Добавляет описательные заголовки к каждому чанку перед эмбеддингом. |
| [6. RAG с аугментацией документов](06_doc_augmentation_rag.ipynb) | Генерирует вопросы из текстовых чанков для улучшения поиска. |
| [7. Трансформация запросов](07_query_transform.ipynb) | Переписывает, расширяет или декомпозирует запросы для улучшения поиска. Включает **Step-back Prompting** и **Sub-query Decomposition**. |
| [8. Переранжирование](08_reranker.ipynb) | Пересортировывает первоначальные результаты с использованием LLM для лучшей релевантности. |
| [9. RSE](09_rse.ipynb) | Извлечение релевантных сегментов: идентифицирует и восстанавливает непрерывные сегменты текста, сохраняя контекст. |
| [10. Контекстуальное сжатие](10_contextual_compression.ipynb) | Фильтрует и сжимает извлечённые чанки, максимизируя релевантную информацию. |
| [11. RAG с обратной связью](11_feedback_loop_rag.ipynb) | Включает обратную связь пользователя для обучения и улучшения системы RAG. |
| [12. Адаптивная RAG](12_adaptive_rag.ipynb) | Динамически выбирает лучшую стратегию поиска на основе типа запроса. |
| [13. Self RAG](13_self_rag.ipynb) | Динамически решает, когда и как извлекать данные, оценивает релевантность и полезность. |
| [14. Чанкинг по утверждениям](14_proposition_chunking.ipynb) | Разбивает документы на атомарные фактические утверждения для точного поиска. |
| [15. Мультимодельная RAG](15_multimodel_rag.ipynb) | Комбинирует текст и изображения для поиска, генерируя подписи к изображениям с помощью LLaVA. |
| [16. Fusion RAG](16_fusion_rag.ipynb) | Комбинирует векторный поиск с ключевыми словами (BM25) для улучшения результатов. |
| [17. Graph RAG](17_graph_rag.ipynb) | Организует знания в виде графа, позволяя перемещаться по связанным концепциям. |
| [18. Иерархическая RAG](18_hierarchy_rag.ipynb) | Строит иерархические индексы (суммаризации + детальные чанки) для эффективного поиска. |
| [19. HyDE RAG](19_HyDE_rag.ipynb) | Использует гипотетические эмбеддинги документов для улучшения семантического соответствия. |
| [20. CRAG](20_crag.ipynb) | Корректирующая RAG: динамически оценивает качество поиска и использует веб-поиск как запасной вариант. |
| [21. RAG с RL](21_rag_with_rl.ipynb) | Максимизирует награду модели RAG с использованием обучения с подкреплением. |
| [Поиск лучшей RAG](best_rag_finder.ipynb) | Находит лучшую технику RAG для заданного запроса. |
| [22. Большие данные с графами знаний](22_Big_data_with_KG.ipynb) | Работает с большими наборами данных с использованием графов знаний. |

## 🗂️ Структура репозитория

```
all-rag-techniques/
├── README.md                          <- Основной файл (англ.)
├── README_ru.md                       <- Этот файл (рус.)
├── 01_simple_rag.ipynb
├── 02_semantic_chunking.ipynb
├── 03_chunk_size_selector.ipynb
├── 04_context_enriched_rag.ipynb
├── 05_contextual_chunk_headers_rag.ipynb
├── 06_doc_augmentation_rag.ipynb
├── 07_query_transform.ipynb
├── 08_reranker.ipynb
├── 09_rse.ipynb
├── 10_contextual_compression.ipynb
├── 11_feedback_loop_rag.ipynb
├── 12_adaptive_rag.ipynb
├── 13_self_rag.ipynb
├── 14_proposition_chunking.ipynb
├── 15_multimodel_rag.ipynb
├── 16_fusion_rag.ipynb
├── 17_graph_rag.ipynb
├── 18_hierarchy_rag.ipynb
├── 19_HyDE_rag.ipynb
├── 20_crag.ipynb
├── 21_rag_with_rl.ipynb
├── 22_Big_data_with_KG.ipynb
├── best_rag_finder.ipynb
├── requirements.txt                   <- Python зависимости
└── data/
    └── val.json                       <- Пример валидационных данных
    └── AI_Information.pdf             <- Тестовый PDF документ
    └── attention_is_all_you_need.pdf  <- PDF для тестирования мультимодальной RAG
```

## 🛠️ Начало работы

1. **Клонируйте репозиторий:**
   ```bash
   git clone https://github.com/FareedKhan-dev/all-rag-techniques.git
   cd all-rag-techniques
   ```

2. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройте API ключ:**
   - Получите API ключ от [Nebius AI](https://studio.nebius.com/)
   - Установите ключ как переменную окружения:
     ```bash
     export OPENAI_API_KEY='ВАШ_КЛЮЧ_API'
     ```
     или для Windows:
     ```bash
     setx OPENAI_API_KEY "ВАШ_КЛЮЧ_API"
     ```
     или в Python скрипте/ноутбуке:
     ```python
     import os
     os.environ["OPENAI_API_KEY"] = "ВАШ_КЛЮЧ_API"
     ```

4. **Запустите ноутбуки:**
   Откройте любой Jupyter ноутбук (`.ipynb` файлы) используя Jupyter Notebook или JupyterLab. Каждый ноутбук самодостаточен и может быть запущен независимо.

   **Примечание:** Файл `data/AI_Information.pdf` предоставляет тестовый документ. Вы можете заменить его своим PDF. Файл `data/val.json` содержит примеры запросов и идеальных ответов для оценки. Файл `attention_is_all_you_need.pdf` используется для тестирования мультимодальной RAG.

## 💡 Основные концепции

- **Эмбеддинги:** Числовые представления текста, сохраняющие семантическое значение. Используется API эмбеддингов Nebius AI и модель `BAAI/bge-en-icl`.
- **Векторное хранилище:** Простая база данных для хранения и поиска эмбеддингов. Мы создаём собственный класс `SimpleVectorStore` используя NumPy.
- **Косинусная схожесть:** Мера схожести между векторами. Более высокие значения указывают на большую схожесть.
- **Чанкинг:** Разделение текста на меньшие, управляемые части. Мы исследуем различные стратегии чанкинга.
- **Поиск:** Процесс нахождения наиболее релевантных текстовых чанков для заданного запроса.
- **Генерация:** Использование Large Language Model (LLM) для создания ответа на основе извлечённого контекста и запроса пользователя. Используется модель `meta-llama/Llama-3.2-3B-Instruct` через API Nebius AI.
- **Оценка:** Анализ качества ответов системы RAG, часто путём сравнения с эталонным ответом или использования LLM для оценки релевантности.

## 🤝 Участие

Вклады приветствуются!