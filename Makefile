.PHONY: all install test train run build docker-run clean

all: test train

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:
	python train.py

run:
	streamlit run app.py

build:
	docker build -t ml-clustering-app:v1.0 .

docker-run:
	docker run -d --name ml-clustering -p 8501:8501 --restart unless-stopped ml-clustering-app:v1.0

clean:
	rm -rf __pycache__ src/__pycache__ .pytest_cache

