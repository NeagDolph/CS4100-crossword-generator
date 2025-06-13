.PHONY: test demo

test-crossword:
	python -m pytest tests/test_crossword.py -v
test-csp:
	python -m pytest tests/test_csp.py -v
test: test-crossword test-csp

demo:
	python main.py demo 