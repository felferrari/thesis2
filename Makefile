init:
	pip install -r env/requeriments.txt
prepare:
	python prepare.py +site=$(SITE) preparation.generate.patches=True
train:
	python train.py +exp=$(EXP) +site=$(SITE)