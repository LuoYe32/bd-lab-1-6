# Fashion-MNIST ML Pipeline (CI/CD + DVC)

Пример полного жизненного цикла разработки модели машинного обучения:  
подготовка данных, обучение модели, тестирование, контейнеризация и автоматизация CI/CD.

Проект демонстрирует **MLOps pipeline** с использованием:
- FastAPI (API для инференса модели)
- DVC (версионирование данных и воспроизводимость)
- Docker (контейнеризация)
- GitHub Actions (CI/CD)
- pytest (тестирование)

---

# Dataset

Используется датасет **Fashion-MNIST**:

https://www.kaggle.com/datasets/zalando-research/fashionmnist

Описание:

- 70 000 изображений одежды
- размер изображений: **28×28**
- grayscale
- **10 классов одежды**

Классы:
- 0 - T-shirt/top
- 1 - Trouser
- 2 - Pullover
- 3 - Dress
- 4 - Coat
- 5 - Sandal
- 6 - Shirt
- 7 - Sneaker
- 8 - Bag
- 9 - Ankle boot

Данные версионируются с помощью **DVC** и хранятся вне Git.

---

# CI/CD Pipeline

CI pipeline выполняет:

1. установку зависимостей
2. загрузку данных через DVC
3. воспроизводимое обучение модели
4. запуск тестов
5. сборку Docker образа
6. публикацию образа в DockerHub

CD pipeline:

1. запускает контейнер
2. выполняет функциональное тестирование API (`scenario.json`)

---

# API

После запуска доступен Swagger UI:

```

http://localhost:8000/docs

```

Эндпоинты:

| endpoint | описание |
|--------|--------|
| `/health` | проверка состояния сервиса |
| `/predict` | предсказание по массиву пикселей |
| `/predict/image` | предсказание по изображению |
| `/predict/random` | случайный тестовый пример |

---

# Установка и запуск

## Создание виртуального окружения

```bash
python -m venv venv
```

Активировать:

Linux / macOS

```bash
source venv/bin/activate
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Запуск через Docker

Сборка образа:

```bash
docker build -t fashion-mnist-api .
```

Запуск контейнера:

```bash
docker run -p 8000:8000 fashion-mnist-api
```

## Docker Compose

```bash
docker-compose up --build
```

# Docker Image

Docker образ публикуется в DockerHub:

```
https://hub.docker.com/r/<username>/bd-lab-1-6
```

# DevSecOps Metadata

Pipeline генерирует файл:

```
dev_sec_ops.yml
```

Он содержит:

- Docker image
- последние коммиты
- результаты тестов

