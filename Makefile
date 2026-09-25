.RECIPEPREFIX = >
.PHONY: install test demo benchmark lock clean

install:
> pip install -r requirements.txt
> python -m venv .venv || true

test:
> pytest -q -W ignore::UserWarning

demo:
> python -m voiceforge.examples.run_demo

benchmark:
> python -m voiceforge.cli --out benchmark.json

lock:
> pip freeze > requirements.lock.txt

clean:
> rm -rf .venv benchmark.json .pytest_cache
