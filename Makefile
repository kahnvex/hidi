ci-deps:
	pip install -r requirements.testing.txt
	pyenv install 2.7.11
	pyenv install 3.4.6
	pyenv install 3.5.3
	pyenv install 3.6.0
	pyenv global 2.7.11 3.4.6 3.5.3 3.6.0

ci:
	tox
