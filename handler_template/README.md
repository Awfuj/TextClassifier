#TEXT CLASSIFIER
Моя модель анализирует семантику текста на основе уже обученного ruBERT-tiny и обученного мной классификатора, который определеяет, текст нейтральный, негативный или позитивный

# Installation (uv)

uv venv
uv pip install -r requirements.txt
uv run python -m uvicorn handler_template.lib.main:app --reload
