python train.py +exp=exp_3 +site=s1 retrain_model=4
python predict.py +exp=exp_3 +site=s1

python train.py +exp=exp_1 +site=s1
python predict.py +exp=exp_1 +site=s1

python train.py +exp=exp_2 +site=s1
python predict.py +exp=exp_2 +site=s1
