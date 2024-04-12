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
	rm -rf data
	mkdir data
	mkdir data/general
	mkdir data/original
	mkdir data/prepared
	mkdir data/prepared/train
	mkdir data/prepared/validation
	cp -r /mnt/storage/ferrari/thesis/$(SITE)/data/opt data/original
	cp -r /mnt/storage/ferrari/thesis/$(SITE)/data/sar data/original
	cp -r /mnt/storage/ferrari/thesis/$(SITE)/data/prodes data/original