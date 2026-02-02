.PHONY: build run stop clean logs help

# Имя образа
IMAGE_NAME := bert-classification-api
CONTAINER_NAME := bert-classification-container
PORT := 8000

# Сборка Docker образа
build:
	docker build -t $(IMAGE_NAME) .

# Запуск контейнера с volume
run:
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/lib:/app/lib \
		$(IMAGE_NAME)

# Остановка контейнера
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Просмотр логов
logs:
	docker logs -f $(CONTAINER_NAME)

# Очистка (остановка и удаление контейнера)
clean: stop
	docker rmi $(IMAGE_NAME) || true

# Перезапуск контейнера
restart: stop run

# Справка
help:
	@echo "Доступные команды:"
	@echo "  make build    - Собрать Docker образ"
	@echo "  make run      - Запустить контейнер с volume"
	@echo "  make stop     - Остановить контейнер"
	@echo "  make logs     - Показать логи контейнера"
	@echo "  make restart  - Перезапустить контейнер"
	@echo "  make clean    - Удалить контейнер и образ"
	@echo "  make help     - Показать эту справку"

