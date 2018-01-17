# SQLNet

This repo provides an implementation of SQLNet and Seq2SQL neural networks for predicting SQL queries on [WikiSQL dataset](https://github.com/salesforce/WikiSQL). The paper is available at [here](https://arxiv.org/abs/1711.04436).

## Citation

> Xiaojun Xu, Chang Liu, Dawn Song. 2017. SQLNet: Generating Structured Queries from Natural Language Without Reinforcement Learning.

## Bibtex

```
@article{xu2017sqlnet,
  title={SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning},
  author={Xu, Xiaojun and Liu, Chang and Song, Dawn},
  journal={arXiv preprint arXiv:1711.04436},
  year={2017}
}
```

## Installation
The data is in `data.tar.bz2`. Unzip the code by running
```bash
tar -xjvf data.tar.bz2
```

The code is written using PyTorch in Python 2.7. Check [here](http://pytorch.org/) to install PyTorch. You can install other dependency by running 
```bash
pip install -r requirements.txt
```

## Downloading the glove embedding.
Download the pretrained glove embedding from [here](https://github.com/stanfordnlp/GloVe) using
```bash
bash download_glove.sh
```

## Extract the glove embedding for training.
Run the following command to process the pretrained glove embedding for training the word embedding:
```bash
python extract_vocab.py
```

## Train
The training script is `train.py`. To see the detailed parameters for running:
```bash
python train.py -h
```

Some typical usage are listed as below:

Train a SQLNet model with column attention:
```bash
python train.py --ca
```

Train a SQLNet model with column attention and trainable embedding (requires pretraining without training embedding, i.e., executing the command above):
```bash
python train.py --ca --train_emb
```

Pretrain a [Seq2SQL model](https://arxiv.org/abs/1709.00103) on the re-splitted dataset
```bash
python train.py --baseline --dataset 1
```

Train a Seq2SQL model with Reinforcement Learning after pretraining
```bash
python train.py --baseline --dataset 1 --rl
```

## Test
The script for evaluation on the dev split and test split. The parameters for evaluation is roughly the same as the one used for training. For example, the commands for evaluating the models from above commands are:

Test a trained SQLNet model with column attention
```bash
python test.py --ca
```

Test a trained SQLNet model with column attention and trainable embedding:
```bash
python test.py --ca --train_emb
```

Test a trained [Seq2SQL model](https://arxiv.org/abs/1709.00103) withour RL on the re-splitted dataset
```bash
python test.py --baseline --dataset 1
```

Test a trained Seq2SQL model with Reinforcement learning
```bash
python test.py --baseline --dataset 1 --rl
```

