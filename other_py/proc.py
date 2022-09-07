import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

def prepareDataFrame(path, prepend, dropIDX=['1', '0']):
    df = pd.read_csv(path)
    
    for idx in dropIDX:
        df.drop(df.columns[idx], axis = 1, inplace = True)
    df.drop('id', axis = 1, inplace = True)
    df.drop('windowTimestamp', axis = 1, inplace = True)
    df.drop('dataType', axis = 1, inplace = True)
    df.drop('systemVersion', axis = 1, inplace = True)
    df.drop('assetCode', axis = 1, inplace = True)
    try:
        df.drop('Asset', axis = 1, inplace = True)
    except:
        pass

    sentCols = [col_name for col_name in df.columns if (col_name != 'Date')]
    df.columns = ['Date'] + [prepend + col_name for col_name in sentCols]
    
    return df, sentCols

def mergeAll(srcpath, dropList = {}, labelcols=['dailypct', 'dailydir'], split=['2000-01-01', '2016-09-30'], shiftLabel=None):
    news = Path(srcpath + '/0_raw/news.csv')
    social = Path(srcpath + '/0_raw/social.csv')
    price = Path(srcpath + '/0_raw/price.csv')
    assert(news.is_file() and social.is_file() and price.is_file())

    newsData, sentCols = prepareDataFrame(srcpath + '/0_raw/news.csv', 'N_')
    socialData, sentCols = prepareDataFrame(srcpath + '/0_raw/social.csv', 'S_')
    priceData = pd.read_csv(srcpath + '/0_raw/price.csv')

    if 'news' in droplist:
        for colname in droplist['news']:
            if colname in newsData:
                newsData.drop(colname, axis=1, inplace=True)
    if 'social' in droplist:
        for colname in droplist['social']:
            if colname in socialData:
                socialData.drop(colname, axis=1, inplace=True)
    if 'price' in droplist:
        for colname in droplist['price']:
            if colname in priceData:
                priceData.drop(colname, axis=1, inplace=True)

    valid = lambda date: (split[0] <= date) & (date <= split[-1])
    newsData = newsData[valid(newsData.Date)]
    socialData = socialData[valid(socialData.Date)]
    priceData = priceData[valid(priceData.Date)]
    priceData.rename(columns=lambda x: 'X_' +x, inplace=True)
    if (shiftLabel):
        for col in labelcols:
            priceData['Z_' + col] = priceData['X_' + col].shift(shiftLabel) #label shifted
    else:
        for col in labelcols:
            priceData['Z_' + col] = priceData['X_' + col]
        for col in priceData:
            if (col != 'X_Date' and col[0] != 'Z'):
                priceData[col] = priceData[col].shift(1)
                #print(priceData)
        
    sent = pd.merge(newsData, socialData, on='Date', how='outer')
    total = pd.merge(sent, priceData, left_on='Date', right_on='X_Date', how='outer')
    total.dropna(axis=0, how='any', subset=['Date'], inplace=True)
    # prev = ''
    # for date in total['Date']:
    #     if (not isinstance(date, str)):
    #         print(date)
    #         print(prev)
    #     else:
    #         prev = date
    total['Y_day'] = [datetime.datetime.strptime(date, '%Y-%m-%d').weekday() for date in total['Date']]
    total.drop('X_Date', axis = 1, inplace = True)
    total.set_index('Date', inplace=True)
    total.sort_index(axis=1, inplace=True)
    total.sort_index(axis=0, inplace=True)
    total.to_csv(srcpath + '/1_merged/all_data.csv')
    if len(split) > 2:

        ffillSplit(srcpath, split)
    return

def ffillSplit(srcpath, split=['2000-01-01', '2015-12-31', '2016-01-01', '2016-10-31'], limit=None):
    assert(len(split) == 4)
    data = pd.read_csv(srcpath + '/1_merged/all_data.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    data = data.set_index('Date')
    data[:split[1]].to_csv(srcpath + '1_merged/in_sample.csv')
    data[split[2]:].to_csv(srcpath + '1_merged/out_sample.csv')
    for col in data:
        if (col[0] == 'N' or col[0] == 'S'):
            data[col].fillna(method='ffill', inplace=True, limit=limit)
    data.to_csv(srcpath + '1_merged/all_dataFF.csv')
    data[:split[1]].to_csv(srcpath + '1_merged/in_sampleFF.csv')
    data[split[2]:].to_csv(srcpath + '1_merged/out_sampleFF.csv')


cldir = './data/CL/'
rutdir = './data/RUT/'
gldir = './data/GLD/'
src = rutdir
droplist = {'price': ['ID', 'rownum']}
labelcols=['dailypct', 'dailydir']
split = ['2000-01-01', '2015-12-31', '2016-01-01', '2016-10-31']
mergeAll(src, droplist, labelcols, split)

# priceData = pd.read_csv(src + '0_raw/price.csv')
# priceData.set_index('Date', inplace=True)
# #priceData = priceData.iloc[::-1] #reverse
# # priceData.drop(priceData.columns[0], axis = 1, inplace = True)
# #priceData[priceData['Date'] >= '1999-12-31'].to_csv(src + '0_raw/all_price.csv')
# #priceData.drop(priceData.columns[0], axis = 1, inplace = True)
# #priceData['daily_pct'] = priceData['Settle'].pct_change()
# try:
#     priceData.drop(['dailypct', 'dailydir', 'onightpct', 'onightdir'], axis=1, inplace=True)
# except:
#     pass
# priceData['dailypct'] = (priceData['Close'] / priceData['Open']) - 1
# #priceData['onightpct'] = (priceData['Close'].shift(1) / priceData['Open']) - 1
# #TODO can add threshold for directional change here
# priceData['dailydir'] = priceData['dailypct'].map(np.sign)
# print(priceData)
# #priceData['onightdir'] = priceData['onight_pct'].map(np.sign)
# priceData.to_csv(src + '0_raw/price.csv')

# # print(priceData)