# Exploring Neural Text Simplification--Pytorch Version
This is the reimplementation of the [NeuralTextSimplification](https://github.com/senisioi/NeuralTextSimplification) repository in Pytorch. The original repository is based on Lua Torch, which may not be able to be installed in some machines (at least in my machine), therefore I provide this pytorch version in case someone may need it. 

The algorithm behind this code is from this paper: [Nisioi, Sergiu, et al. "Exploring neural text simplification models." Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). 2017.](https://www.aclweb.org/anthology/P17-2014/) 

It is based on the standard LSTM based seq-to-seq translation model and OpenNMT is used as the code base. 

## How to use

1. OpenNMT dependency: You first need to install the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) tool:

```
pip install OpenNMT-py
```

2. Checkout this repository:

```
git clone https://github.com/jind11/NeuralTextSimplification-Pytorch
```

3. Make a directory named "models", download the pre-trained released models [NTS](https://drive.google.com/file/d/1oRRWXTQ-JXSTxJ944X1JI-PZyovmtMxU/view?usp=sharing) and save into it. If you want to train your own model based on your data, you can use this command (remember to change the directory of the data and model save path):

```
./train.sh
```

We also provide the EW-SEW dataset used to train the released pre-trained model in the "data" folder.

4. Run translate.sh to get the translation results for your dataL

```
mkdir results
./translate.sh
```

5. Run automatic evaluation metrics (nltk package is needed for this step):

```
./evaluate.sh
```

## Benchmark

Since this is a reimplementation of an existing repository, we would like to compare the performance to the original one for quality checking based on two automatic metrics: SARI and BLEU.

| Approach                           | Repository |  SARI |  BLEU |
|------------------------------------|------------|------:|------:|
| NTS default (beam 5, hypothesis 1) | Original   | 30.65 | 84.51 |
| NTS default (beam 5, hypothesis 1) | This one   | 29.90 | 93.67 |
| NTS SARI (beam 5, hypothesis 2)    | Original   | 37.25 | 80.69 |
| NTS SARI (beam 5, hypothesis 2)    | This one   | 38.63 | 87.19 |
| NTS BLEU (beam 12, hypothesis 1)   | Original   | 30.77 | 84.70 |
| NTS BLEU (beam 12, hypothesis 1)   | This one   | 29.78 | 93.71 |

From this table, we see that this reimplementation is comparable or even better than the original code. 

In the end, we put the automatic metrics results for all four hypotheses for beam search of 5:

| Hypothesis Number |  SARI |  BLEU |
|-------------------|------:|------:|
| 1                 | 29.90 | 93.67 |
| 2                 | 38.63 | 87.19 |
| 3                 | 38.65 | 84.67 |
| 4                 | 37.92 | 84.19 |
