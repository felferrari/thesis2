init:
	pip install -r env/requeriments.txt

prepare:
	python prepare.py 


train:
	python train.py $(EXP)