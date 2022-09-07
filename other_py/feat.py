import datetime
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
#from altair import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import sklearn.metrics as skm

from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
    cldir = './data/CL/'
    rutdir = './data/RUT/'
    gldir = './data/GLD/'
    src = cldir
    labelName = 'Z_dailydir'
    pricestats=['Date', 'Z_dailypct']

    data.dropna(axis=0, inplace=True)
    #data.dropna(axis=0, inplace=True)
        
    #optional - remove useless sentiments

    #TODO - add weekend sentiment to monday

    #develop features
    data = getFeats(data, labelName, pricestats)
    feats = data.drop((col for col in data if col[0] != 'F'), axis=1)
    stats = data.drop((col for col in data if col not in pricestats), axis=1)    
    labels = data[labelName]
    #binning

    #cross validation
    train = 4
    folds = 5
    rounds = 5
    dq = collections.deque(idxsplit(feats.shape[0], folds=folds, shuffle=False))
    metrics = {'accuracy': np.zeros(rounds)}#, 'upPrecision': np.zeros(rounds),
    #  'downPrecision':np.zeros(rounds), 'upRecall': np.zeros(rounds),'downRecall':np.zeros(rounds),
    #  'upF1': np.zeros(rounds), 'downF1':np.zeros(rounds)}

    for i in range(rounds):
        dq.rotate(1)
        indices = list(dq)
        train_idx = np.sort(np.concatenate(indices[:train]))
        test_idx = np.sort(np.concatenate(indices[train:]))

        # train/test splits
        feats_train = feats.iloc[train_idx]
        feats_test = feats.iloc[test_idx]
        stats_train = stats.iloc[train_idx]
        stats_test = stats.iloc[test_idx]
        labels_train = labels.iloc[train_idx]
        labels_test = labels.iloc[test_idx]

        kn = KNeighborsClassifier(n_neighbors=600)
        kn = tree.DecisionTreeClassifier()
        kn.fit(feats_train, labels_train)
        predictions = kn.predict(feats_test)
        eCurves = equityCurves(predictions, labels_test, stats_test)
        # print(predictions)

        # plt.plot_date(dates, longOnly, mfc='green', label='long only')
        # plt.plot_date(dates, shortOnly, mfc='red', label='short only')
        # plt.plot_date(dates, longShort, mfc='blue', label='long short')
        plt.plot(eCurves['dates'], eCurves['longOnly'], color='green', label='long only')
        plt.plot(eCurves['dates'], eCurves['shortOnly'], color='red', label='short only')
        plt.plot(eCurves['dates'], eCurves['longShort'], color='blue', label='long short')

        #TODO: accuracy, up/down p&r, sharpe ratio, alpha, graphing
        metrics['accuracy'][i] = skm.accuracy_score(labels_test, predictions)
        # metrics['upPrecision'][i] = skm.precision_score(y_test, predictions, average='micro')
        # metrics['downPrecision'][i] = skm.precision_score(y_test, predictions, pos_label=-1)
        # metrics['upRecall'][i] = skm.recall_score(y_test, predictions, average='micro')
        # metrics['downRecall'][i] = skm.recall_score(y_test, predictions, pos_label=-1)
        # metrics['upF1'][i] = skm.f1_score(y_test, predictions, average='micro')
        # metrics['downF1'][i] = skm.f1_score(y_test, predictions, pos_label=-1)
    # for metric in sorted(metrics):
    #     print(metric +':', np.mean(metrics[metric]))
    plt.xlabel('day')
    plt.ylabel('cumulative return')
    plt.title('$1 invested daily equity curve')
    plt.legend()
    plt.show()
    print('accuracy:', np.mean(metrics['accuracy']))
    cnt = data[labelName].value_counts()
    totalCount = cnt.sum(axis=0)
    print('Guess Up:', cnt[1] / totalCount)
    print('Guess Down:', cnt[-1] / totalCount)
    return

def getFeats(df, labelName='Z_dailydir', pricestats=['Date', 'Z_dailypct']):
    feats = df.copy()
    def getMACrossover(series, slow, fast):
        slowMA = pd.rolling_mean(series, slow, 1).apply(lambda x: int(x * 100))
        fastMA = pd.rolling_mean(series, fast, 1).apply(lambda x: int(x * 100))
        return slowMA - fastMA

    #feats['F_dayOfWeek'] = feats['Y_day'].apply(lambda x: int(x))
    feats['F_isFriday'] = feats['Y_day'].apply(lambda x: int(x)==4)
    #feats['F_month'] = feats['ate'].apply(lambda x: int(x[5:7]))
    nSent = ['N_stress', 'N_relativeBuzz', 'N_gloom']
    sSent = ['S_sentiment', 'S_priceDirection', 'S_longShort', 'S_relativeBuzz']

    for sent in nSent:# + sSent:
        feats['F_' + sent] = getMACrossover(feats[sent], 7, 28)

    # pd28 = feats['N_priceDirection'].rolling(window=28)
    # pd7 = feats['N_priceDirection'].rolling(window=7)

    # ma14 = pd.rolling_mean(feats['N_priceDirection'], 28, 1).apply(lambda x: int(x * 100))
    # ma7 = pd.rolling_mean(feats['N_priceDirection'], 7, 1).apply(lambda x: int(x * 100))
    # feats['F_MA(7)-F_MA(28)_N_priceDirection'] = ma7 - ma14

    #Drop any column that is not a feature, the date, or the label
    dropList = [col for col in feats if col[0] != 'F' and col != 'Date' and
     col != labelName and col not in pricestats]
    feats.drop(dropList, axis=1, inplace=True)

    #drop all unlabelled feature vectors (such as weekends)
    feats.dropna(subset=[labelName], axis=0, inplace=True)
    return feats

def equityCurves(predictions, actual, stats):
    correct = predictions == actual.values
    dailypct = stats['Z_dailypct'].values
    dates = pd.to_datetime(stats['Date'], format='%Y-%m-%d').values
    eCurves = pd.DataFrame()
    longOnly = 1
    shortOnly = 1
    longShort = 1
    longOnlyCurve = np.zeros(len(correct))
    shortOnlyCurve = np.zeros(len(correct))
    longShortCurve = np.zeros(len(correct))
    for i in range(len(correct)):
        val = abs(dailypct[i])
        longPrediction = predictions[i] == 1
        if (correct[i]):
            if (longPrediction):
                longOnly += val
            else:
                shortOnly += val
            longShort += val
        else:
            if (longPrediction):
                longOnly -= val
            else:
                shortOnly -= val
            longShort -= val
        longOnlyCurve[i] = longOnly
        shortOnlyCurve[i] = shortOnly
        longShortCurve[i] = longShort
    eCurves['dates'] = dates
    eCurves['longOnly'] = longOnlyCurve
    eCurves['shortOnly'] = shortOnlyCurve
    eCurves['longShort'] = longShortCurve

    return eCurves#dates, longOnlyCurve, shortOnlyCurve, longShortCurve

def countHoles(df):
    for col in df:
        print(col +':', (df[col].isnull().values == True).sum())
    print()
    return

#Returns a list ndarrays, each of which hold the indices of each fold
#of a k-fold cross validation
def idxsplit(numExamples, folds=10, shuffle=True, seed=None):
    arr = np.linspace(0, numExamples, numExamples, False, False, np.int32)
    cutpoints = list(np.linspace(numExamples // folds, numExamples, folds-1, False, False))
    cutpoints = [int(round(x)) for x in cutpoints]
    if (seed):
        np.random.seed(seed)
    if (shuffle):
        np.random.shuffle(arr)
    return np.split(arr, cutpoints)

if __name__ == "__main__":
    main()
