# Sequence-to-Sequence Learning with Latent Neural Grammars
Code for the paper:  
[Sequence-to-Sequence Learning with Latent Neural Grammars](https://arxiv.org/abs/2109.01135)
Yoon Kim  
arXiv Preprint

## Dependencies
The code was tested in `python 3.7` and `pytorch 1.5`. We also use a slightly modified version of the [Torch-Struct](https://github.com/harvardnlp/pytorch-struct) library, which is included in the repo and can be installed via:
```
cd pytorch-struct
python setup.py install
```

## Data
For convenience we include the datasets used in the paper in the `data/` folder. Please cite the original papers when using the data (i.e. [Lake and Baroni 2018](https://arxiv.org/abs/1711.00350) for SCAN/MT, and [Lyu et al. 2021](https://arxiv.org/abs/2104.05196) for StylePTB).

## Training 


### SCAN
To train the model on (for example) the `length` split:
```
python train_scan.py --train_file data/SCAN/tasks_train_length.txt --save_path scan-length.pt
```
For prediction and evaluation:
```
python predict_scan.py --data_file data/SCAN/tasks_test_length.txt --model_path scan-length.pt
```

### Style Transfer
To train on (for example) the `active-to-passive` task:
```
python train_styleptb.py --train_file data/StylePTB/ATP/train.tsv --dev_file data/StylePTB/ATP/valid.tsv --save_path styleptb-atp.pt
```
To predict:
```
python predict_styleptb.py --data_file data/StylePTB/ATP/test.tsv --model_path styleptb-atp.pt 
--out_file styleptb-atp-pred.txt
```
We use the [nlg-eval](https://github.com/Maluuba/nlg-eval) package to calculate the various metrics.

### Machine Translation
To train on MT:
```
python train_mt.py --train_file_src data/MT/train.en --train_file_tgt data/MT/train.fr 
--dev_file_src data/MT/dev.en --dev_file_tgt data/MT/dev.fr --save_path mt.pt
```
To predict on the daxy test set:
```
python predict_mt.py --data_file data/MT/test-daxy.en --model_path mt.pt --out_file mt-pred-daxy.txt
```
For the regular test set:
```
python predict_mt.py --data_file data/MT/test.en --model_path mt.pt --out_file mt-pred.txt
```

We use the [multi-bleu script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl) to calculate BLEU.

### Training Stability
We observed training to be unstable and the approach required several runs across different seeds to perform well. For reference we have posted logs of some example runs in the `logs/` folder.

## License
MIT
