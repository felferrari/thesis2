init:
	pip install -r env/requeriments.txt
prepare:
	python prepare.py +site=$(SITE) preparation.calculate.statistics=True preparation.generate.tiles=True preparation.generate.labels=True preparation.generate.prev_map=True preparation.generate.patches=True
train:
	python train.py +exp=$(EXP) +site=$(SITE)
copyfiles:
	rm -rf data
	mkdir data
	mkdir data/general
	mkdir data/original
	mkdir data/prepared
	mkdir data/prepared/train
	mkdir data/prepared/validation
	rsync --progress -r /mnt/storage/ferrari/thesis/$(SITE)/data/sar data/original
	rsync --progress -r /mnt/storage/ferrari/thesis/$(SITE)/data/opt data/original
	rsync --progress -r /mnt/storage/ferrari/thesis/$(SITE)/data/prodes data/original
	python prepare.py +site=$(SITE) preparation.calculate.statistics=True preparation.generate.tiles=True preparation.generate.labels=True preparation.generate.prev_map=True preparation.generate.patches=True
nohup-run-all:
	nohup ./run_all.sh &
push-mlruns:
	rsync --progress -r -u -t mlruns /mnt/storage/ferrari/thesis/
pull-mlruns:
	rm -rf mlruns
	rsync --progress -r -u -t /mnt/storage/ferrari/thesis/mlruns .
