init:
	pip install -r env/requeriments.txt
prepare:
	python prepare.py +site=$(SITE) preparation.generate.patches=True
train:
	python train.py +exp=$(EXP) +site=$(SITE)
nohuptrain:
	nohup python train.py +exp=$(EXP) +site=$(SITE) >output.log &
nohupmultitrain:
	nohup python train.py -m +exp=$(EXPS) +site=$(SITE) >output.log &
copyfiles:
	