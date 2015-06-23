CONDA_HOME = $(HOME)/.conda/
ENV_NAME=BOWPopcorn
ENV_DIR = $(CONDA_HOME)/envs/$(ENV_NAME)
ENV_BIN_DIR = $(ENV_DIR)/bin

conda_requriements:
	conda env export >> requirements.txt

nltk_downloads:
	$(ENV_BIN_DIR)/python -m nltk.downloader stopwords
	$(ENV_BIN_DIR)/python -m nltk.downloader words
