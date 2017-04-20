ci-deps:
	pip install -r requirements.testing.txt
	pyenv install -s 2.7.11
	pyenv install -s 3.5.3
	pyenv global 2.7.11 3.5.3

ci:
	tox -e py27 -e py35
