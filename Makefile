#! /usr/bin/env make

all: lint test doc

lint:
	pocketlint *.py
	find ijssel test -name \*.py -print0 | xargs -0 pocketlint

test:
	tox


doc:
	pydoc -w ijssel
	mv ijssel.html doc/pydoc/


.PHONY: doc lint test
