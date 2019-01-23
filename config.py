import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # data
    datatype = 'ml_100k'#'yahoo_music'#'flixster'#'ml_1m'#'douban'#
    parser.add_argument('--mode', type=str, default="train",
                                  help='train / test')
    parser.add_argument('--model-path', type=str, default="./models")
    parser.add_argument('--data-path', type=str, default="./data")
    parser.add_argument('--data-type', type=str, default=datatype)
    parser.add_argument('--data-shuffle', type=bool, default=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--val-step', type=int, default=10)
    parser.add_argument('--test-epoch', type=int, default=50)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--seed', type=float, default=20)
    parser.add_argument('--neg-cnt', type=int, default=100)
    parser.add_argument('--at-k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')

    parser.add_argument('--accum', type=str, default='sum', help='sum / stack')
    parser.add_argument('--emb-dim', type=int, default=64)
    parser.add_argument('--nb', type=int, default=5)

    if datatype == 'yahoo_music':
        parser.add_argument('--hidden', default=[64,32])
        parser.add_argument('--train-path', type=str, default='./data/yahoo_music_0.pkl')
        parser.add_argument('--val-path', type=str, default='./data/yahoo_music_1.pkl')
        parser.add_argument('--test-path', type=str, default='./data/yahoo_music_2.pkl')
    elif datatype == 'flixster':
        parser.add_argument('--hidden', default=[32,16])
        parser.add_argument('--train-path', type=str, default='./data/flixster_0.pkl')
        parser.add_argument('--val-path', type=str, default='./data/flixster_1.pkl')
        parser.add_argument('--test-path', type=str, default='./data/flixster_2.pkl')
    elif datatype == 'douban':
        parser.add_argument('--hidden', default=[32,16])
        parser.add_argument('--train-path', type=str, default='./data/douban_0.pkl')
        parser.add_argument('--val-path', type=str, default='./data/douban_1.pkl')
        parser.add_argument('--test-path', type=str, default='./data/douban_2.pkl')
    elif datatype == 'ml_100k':
        parser.add_argument('--hidden', default=[64,32])
        parser.add_argument('--train-path', type=str, default='./data/rating_0.pkl')
        parser.add_argument('--val-path', type=str, default='./data/rating_1.pkl')
        parser.add_argument('--test-path', type=str, default='./data/rating_2.pkl')
    elif datatype == 'ml_1m':
        parser.add_argument('--hidden', default=[32,16])
        parser.add_argument('--train-path', type=str, default='./data/ml_1m_0.pkl')
        parser.add_argument('--val-path', type=str, default='./data/ml_1m_1.pkl')
        parser.add_argument('--test-path', type=str, default='./data/ml_1m_2.pkl')

    args = parser.parse_args()

    return args
