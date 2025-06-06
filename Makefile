.PHONY: test demo

test:
	python -m pytest tests/test_crossword.py -v

demo:
	python main.py demo 