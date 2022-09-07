# Kevin Patel
# EE 509
# 6/5/2017

import numpy as np
import pandas as pd
from common import _assert_all_finite, _drop_nans_idx
from reservoir import Reservoir
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA

load_array = lambda path: np.load(path)
sharpe_ratio = lambda return_curve: float(np.mean(return_curve) / np.std(return_curve, ddof=0))
sortino_ratio = lambda return_curve: float(np.mean(return_curve) / np.std(return_curve[return_curve < 0], ddof=0))

def main():
    np.random.seed(0)
    print('*****************************************************************')

    # # Small reservoir save
    # small_res = Reservoir(size=100, connectivity=1, spectral_radius=1, standardize_input=False,
    #                       input_scaling=1, leaking_rate=.75, discard=0)
    # print(small_res)
    # print('\nsaving label data and computing small reservoir activations...', end='')
    # process_and_save('./act/' +'SPX-small-act.npy', reservoir=small_res)
    # print('done\n')
    # print('*****************************************************************')

    # # Big reservoir save
    # big_res = Reservoir(size=200, connectivity=1, spectral_radius=10, standardize_input=False,
    #                       input_scaling=1, leaking_rate=.90, discard=0)
    # print(big_res)
    # print('\nsaving label data and computing big reservoir activations...', end='')
    # process_and_save('./act/' +'SPX-big-act.npy', reservoir=big_res)
    # print('done\n')
    print('*****************************************************************')

    # load labels into memory
    print('load labels...', end='')
    all_dates = load_array('./lab/' + 'SPX-dates.npy')
    net_return = load_array('./lab/' + 'SPX-net_return.npy')
    # log_return = load_array('./lab/' + 'SPX-log_return.npy')
    # price_dir = load_array('./lab/' + 'SPX-price_dir.npy')
    lb_price_dir = load_array('./lab/' + 'SPX-lb_price_dir.npy')
    print('done\n')
    print('*****************************************************************')

    # load precomputed reservoir activations into memory
    print('load precomputed reservoir activations...', end='')
    small_activations = load_array('./act/' +'SPX-small-act.npy')
    big_activations = load_array('./act/' +'SPX-big-act.npy')
    print('done\n')
    print('*****************************************************************')

    # PCA
    # num_comp = 150
    # print('pca with', num_comp, 'components...', end='')
    # pca = PCA(n_components=num_comp)
    # pca_activations = pca.fit_transform(activations)
    # print('done\n')
    print('*****************************************************************')
    
    # Regression preparation
    print('Regression Readout')
    train_bound = int(.8 * small_activations.shape[0])
    to_drop = _drop_nans_idx(net_return)
    data = np.delete(np.copy(small_activations), to_drop, axis=0)
    target = np.delete(np.copy(net_return), to_drop, axis=0)
    dates = np.delete(np.copy(all_dates), to_drop, axis=0)
    _assert_all_finite(data)
    _assert_all_finite(target)

    # Regression
    print('Ridge Regression of net return:')
    reg = Ridge(alpha=1e-6, fit_intercept=False)
    reg.fit(data[:train_bound], target[:train_bound])
    reg_guesses = reg.predict(data[train_bound:])
    mse = mean_squared_error(target[train_bound:], reg_guesses)
    print('mean squared error: {0:.8f}'.format(mse))
    print()

    # Regression Graphing
    test_period = pd.to_datetime(dates, format='%Y-%m-%d').values[train_bound:]
    plt.scatter(test_period, target[train_bound:], color='red', marker='.', label='actual')
    plt.scatter(test_period, reg_guesses, color='blue', marker='.', label='ESN prediction')
    plt.xlabel('Time (hour)')
    plt.ylabel('Net Return')
    title = 'S&P 500 ESN Ridge Regression from' +str(test_period[0])[:10] +' to ' +str(test_period[-1])[:10]
    plt.title(title)
    plt.legend()
    plt.show()
    print('*****************************************************************')

    # Classification preparation
    print('Classification Readout')
    train_bound = int(.8 * big_activations.shape[0])
    to_drop = _drop_nans_idx(lb_price_dir)
    data = np.delete(np.copy(big_activations), to_drop, axis=0)
    label = np.delete(np.copy(lb_price_dir), to_drop, axis=0)
    dates = np.delete(np.copy(all_dates), to_drop, axis=0)
    pct = np.delete(np.copy(net_return), to_drop, axis=0)
    _assert_all_finite(data)
    _assert_all_finite(label)
    _assert_all_finite(pct)
    label[label == -1] = 0
    label = label.astype(np.int64)

    # Classification
    print('Logistic Classification of price direction:')
    clf = LogisticRegression(C=1.0, fit_intercept=True)
    clf.fit(data[:train_bound], label[:train_bound])
    clf_guesses = clf.predict(data[train_bound:])
    acc = accuracy_score(label[train_bound:], clf_guesses)
    print('accuracy: {0:.8f}'.format(acc))
    print('prior distribution of [down, up]:', np.bincount(label[train_bound:]) / label[train_bound:].shape[0])
    print()

    # Classification equity curve graphing
    test_period = pd.to_datetime(dates, format='%Y-%m-%d').values[train_bound:]
    long_only, short_only, long_short, long_hold = equity_curves(clf_guesses, label[train_bound:], pct[train_bound:])
    plt.plot(test_period, long_only[1:], color='green', label='long only')
    plt.plot(test_period, short_only[1:], color='red', label='short only')
    plt.plot(test_period, long_short[1:], color='blue', label='long short')
    plt.plot(test_period, long_hold[1:], color='black', label='long hold (benchmark)')
    plt.xlabel('Time (hour)')
    plt.ylabel('Cumulative Return')
    title = 'S&P 500 Logistic Classification Equity Curves from ' +str(test_period[0])[:10] +' to ' +str(test_period[-1])[:10]
    plt.title(title)
    plt.legend()
    plt.show()

    print('long only final value: {0:1.4f}'.format(long_only[-1]))
    print('short only final value: {0:1.4f}'.format(short_only[-1]))
    print('long short final value: {0:1.4f}'.format(long_short[-1]))
    print('long hold (benchmark) final value {0:1.4f}:'.format(long_hold[-1]))
    print()

    # print('long only sharpe ratio: {0:1.4f}'.format(sharpe_ratio(np.diff(long_only))))
    # print('short only sharpe ratio: {0:1.4f}'.format(sharpe_ratio(np.diff(short_only))))
    # print('long short sharpe ratio: {0:1.4f}'.format(sharpe_ratio(np.diff(long_short))))
    # print('long hold (benchmark) sharpe ratio {0:1.4f}:'.format(sharpe_ratio(np.diff(long_hold))))
    # print()

    print('long only sortino ratio: {0:1.4f}'.format(sortino_ratio(np.diff(long_only))))
    print('short only sortino ratio: {0:1.4f}'.format(sortino_ratio(np.diff(short_only))))
    print('long short sortino ratio: {0:1.4f}'.format(sortino_ratio(np.diff(long_short))))
    print('long hold (benchmark) sortino ratio: {0:1.4f}'.format(sortino_ratio(np.diff(long_hold))))
    print('*****************************************************************')

# compute activations, get labels, and save both to file
def process_and_save(act_path, reservoir=None):
    """
    :param act_path: path to dump reservoir activations
    :param res: prebuilt reservoir object (optional)
    """
    indexers = ['id', 'date']
    targets = ['net_return', 'log_return']
    labels = ['price_dir', 'long_bias_price_dir']

    # Get the data and prepare targets/labels
    data = pd.read_csv('./feat/' +'SPX-feat.csv') # S&P 500 based features
    dates = data['date'].as_matrix()
    net_return = data['net_return'].as_matrix()
    # log_return = data['log_return'].as_matrix()
    # price_dir = data['price_dir'].as_matrix()
    lb_price_dir = data['long_bias_price_dir'].as_matrix()

    # Save label arrays to file
    np.save('./lab/' + 'SPX-dates.npy', dates)
    np.save('./lab/' + 'SPX-net_return.npy', net_return)
    # np.save('./lab/' + 'SPX-log_return.npy', log_return)
    # np.save('./lab/' + 'SPX-price_dir.npy', price_dir)
    np.save('./lab/' + 'SPX-lb_price_dir.npy', lb_price_dir)

    # Prepare features
    to_drop = indexers + targets + labels
    feat_matrix = data.drop(to_drop, axis=1).as_matrix()

    # Init reservoir, compute activations, and save them to file
    if (reservoir is None):
        reservoir = Reservoir()
    np.save(act_path, reservoir.transform(feat_matrix))

def equity_curves(predictions, actual, pct):
    """
    :param predictions: ndarray of predicted price direction
    :param actual: ndarray of actual price direction
    :param pct: return statistic used
    """
    correct = predictions == actual
    long_only_curve = np.zeros(correct.shape[0] + 1)
    short_only_curve = np.zeros(correct.shape[0] + 1)
    long_short_curve = np.zeros(correct.shape[0] + 1)
    long_hold_curve = np.zeros(correct.shape[0] + 1)
    long_only = 1
    short_only = 1
    long_short = 1
    long_hold = 1
    long_only_curve[0] = long_only
    short_only_curve[0] = short_only
    long_short_curve[0] = long_short
    long_hold_curve[0] = long_hold

    for i in range(1, correct.shape[0] + 1):
        ret = abs(pct[i-1])
        long_prediction = predictions[i-1] == 1
        long_hold += pct[i-1]
        if (correct[i-1]):
            if (long_prediction):
                long_only += ret
            else:
                short_only += ret
            long_short += ret
        else:
            if (long_prediction):
                long_only -= ret
            else:
                short_only -= ret
            long_short -= ret
        long_only_curve[i] = long_only
        short_only_curve[i] = short_only
        long_short_curve[i] = long_short
        long_hold_curve[i] = long_hold

    return long_only_curve, short_only_curve, long_short_curve, long_hold_curve

if __name__ == "__main__":
    main()