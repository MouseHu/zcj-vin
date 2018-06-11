train:

python train-vin.py
python train-vindyn.py
python train-cnn.py

test:

python test-vin.py model/moving-model-vin.pkl
python test-vindyn.py model/moving-model-vindyn.pkl
python test-cnn.py model/moving-model-cnn.pkl


acknowledgement:
myvin.py is modified from https://github.com/transedward/pytorch-dqn