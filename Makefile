#! /usr/bin/env make

all: lint test

lint:
	pocketlint *.py
	find ijssel test -name \*.py -print0 | xargs -0 pocketlint

test:
	tox


.PHONY: lint test
