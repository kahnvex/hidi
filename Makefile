ci-deps:
	pip install -r requirements.testing.txt
	pyenv install -s 2.7.11
	pyenv install -s 3.4.6
	pyenv install -s 3.5.3
	pyenv install -s 3.6.0
	pyenv global 2.7.11 3.4.6 3.5.3 3.6.0

ci:
	tox
