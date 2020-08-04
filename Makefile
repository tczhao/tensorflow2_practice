init-local:
	@echo "Setting up python virtualenv"
	pyenv install 3.6.5
	pyenv virtualenv 3.6.5 tensorflow2_3.6.5
	pyenv local tensorflow2_3.6.5
	pip install --upgrade pip
	pip install -r requirements.txt
