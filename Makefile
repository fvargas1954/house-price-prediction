install:
	pip install --upgrade pip
	pip install -r requirements.txt

format:
	black src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503,E402,F401,F541 || true

train:
	set PYTHONPATH=%CD% && python src/train.py

validate:
	set PYTHONPATH=%CD% && python src/validate.py

test:
	set PYTHONPATH=%CD% && pytest tests/ -v

clean:
	rmdir /s /q mlruns __pycache__ .pytest_cache 2>nul
	for /d /r %%i in (__pycache__) do @rmdir /s /q "%%i" 2>nul

all: install format lint train validate test