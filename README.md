# Uzbek NER: API

## Запуск
1. Склонировать репозиторий, например:
```
git clone https://github.com/uzbeki-vs-ner/API.git
```
2. Распаковать содержимое [архива](https://drive.google.com/file/d/1jSJrC2kycv5u8zpjhqdLApTcIXZzqhKE/view?usp=sharing) в `artifacts/case-solution/model`, чтобы получилось:
```
artifacts/case-solution/model/gliner_config.json
artifacts/case-solution/model/pytorch_model.bin
artifacts/case-solution/model/tokenizer.json
artifacts/case-solution/model/tokenizer_config.json
artifacts/case-solution/model/train_config.json
```
3. Собрать контейнер и запустить его (`--gpus all` опционально, если есть GPU):
```
docker build -t ner-uz-solution .
docker run --rm --gpus all -p 8000:8000 ner-uz-solution
```
4. Позвонить на `localhost:8000` согласно контракту

## Проверка совместимости

Только контракт:
```bash
python scripts/check_service.py --url http://localhost:8000
```

Контракт + посчитать метрики:
```bash
python scripts/evaluate_service.py \
  --url http://localhost:8000 \
  --gold data/dev.jsonl \
  --predictions artifacts/service/dev_predictions.jsonl \
  --output artifacts/service/dev_metrics.json
```
