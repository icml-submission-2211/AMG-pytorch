# Matrix Completion with Aggregated Multi-hop Graph

PyTorch based implementation of Aggregated Multi-hop Graph (AMG) for matrix completion.

This code covers the Flixster, Douban, YahooMusic, Movielens 100K Dataset.

After downloading [ml_1m](https://grouplens.org/datasets/movielens/) to the ```./data``` directory, you need to preprocess it by ```Preprocess.ipynb```.

## Requirements


  * Python 3.5
  * PyTorch (1.0)


## Usage

```bash
python train.py --datatype 'flixster'
python train.py --datatype 'douban'
python train.py --datatype 'yahoo_music'
python train.py --datatype 'ml_100k'
```
