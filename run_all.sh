make copyfiles SITE=s2
python train.py +exp=exp_462 +site=s2
python predict.py +exp=exp_462 +site=s2
python evaluate.py +exp=exp_462 +site=s2
python analize.py +exp=exp_462 +site=s2
python analize2.py +exp=exp_462 +site=s2

python train.py +exp=exp_463 +site=s2
python predict.py +exp=exp_463 +site=s2
python evaluate.py +exp=exp_463 +site=s2
python analize.py +exp=exp_463 +site=s2
python analize2.py +exp=exp_463 +site=s2

python train.py +exp=exp_461 +site=s2
python predict.py +exp=exp_461 +site=s2
python evaluate.py +exp=exp_461 +site=s2
python analize.py +exp=exp_461 +site=s2
python analize2.py +exp=exp_461 +site=s2
make copyfiles SITE=s1