.PHONY: test bench cli clean lint

test:
	python -m unittest discover -s tests -v

bench:
	PYTHONPATH=. python examples/bench_numbers.py

examples:
	PYTHONPATH=. python examples/generate_model_image.py
	PYTHONPATH=. python examples/run_workflow.py

cli:
	python -m zhulu plan model_image --input product.png
	python -m zhulu plan video
	python -m zhulu schedule examples/configs/jobs.json
	python -m zhulu voc examples/configs/reviews.txt

lint:
	ruff check zhulu tests examples

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache build dist *.egg-info
