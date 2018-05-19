#! /usr/bin/env make

all: lint test doc

clean:
	$(RM) ijssel.html doc/pydoc/ijssel.html
	find -name \*.pyc -delete
	find -name __pycache__ -type d -print0 | xargs -0 -r rmdir
	$(RM) -r .tox .pytest_cache


lint:
	pocketlint *.py
	find ijssel test -name \*.py -print0 | xargs -0 pocketlint

test:
	tox


doc:
	pydoc -w ijssel
	mv ijssel.html doc/pydoc/


.PHONY: doc lint test
