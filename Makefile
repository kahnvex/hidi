ci-test-deps:
	pip install -r requirements.testing.txt
	pyenv install -s 2.7.11
	pyenv install -s 3.5.3
	pyenv global 2.7.11 3.5.3

ci-test:
	tox -e py27 -e py35

full-test-deps:
	pip install -r requirements.testing.txt
	pyenv install -s 2.7.11
	pyenv install -s 3.4.6
	pyenv install -s 3.5.3
	pyenv install -s 3.6.0
	pyenv global 2.7.11 3.4.6 3.5.3 3.6.0

full-test:
	tox

full-test-all: full-test-deps full-test

docs-deps:
	pip install -r requirements.txt

docs: docs-deps
	sphinx-build docs/source docs/build

docs-dev: docs-deps
	sphinx-autobuild docs/source docs/build
