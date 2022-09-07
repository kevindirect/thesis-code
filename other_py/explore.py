import datetime
import pandas as pd
import numpy as np
import re
from matplotlib import cm, pyplot
from matplotlib.backends.backend_pdf import PdfPages

def main():
    cldir = './data/CL/'
    rutdir = './data/RUT/'
    gldir = './data/GLD/'
    # data = pd.read_csv(rutdir + '1_merged/in_sample.csv')
    # data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    # data = data.set_index('Date')

    # overallStats('./data/CL/1_merged/in_sample.csv', './data/CL/reports/basic/overall/stats_2000-2013.csv',
    #  './data/CL/reports/basic/overall/stats_2000-2013Graphs.pdf')
    #slicedStats('./data/CL/1_merged/in_sample.csv', monthly, './data/CL/reports/basic/sliced/')
    yearly = [('20'+str(i).zfill(2) +'-01-01','20'+str(i+1).zfill(2) +'-12-31') for i in range(16)]

    for src in [rutdir]:
        data = pd.read_csv(src + '/1_merged/in_sampleFF.csv')
        data = data.set_index('Date')
        dumpStats(src, data, 'FF')
        for i in range(16):
            sliced = data[yearly[15][0]:yearly[15][1]]
            dumpStats(src, sliced, str(yearly[15][0][:4] +'FF'), str('/' +yearly[15][0][:4] +'/'))

    return

def dumpStats(srcpath, data, suffix='', dr='', labelHists=['Z_dailypct']):
    for label in labelHists:
        seriesHist(data[label], 50, srcpath + 'reports/basic/overall/' +dr +label[2:] +'_hist' +suffix +'.pdf')
    labelledStats(data, False, False, srcpath + 'reports/basic/overall/' +dr +'labelled' +suffix +'.pdf')
    labelledStats(data, True, False, srcpath + 'reports/basic/overall/' +dr +'labelledNoNa' +suffix +'.pdf')
    labelledStats(data, True, True, srcpath + 'reports/basic/overall/' +dr +'labelledNoNaNoZero' +suffix +'.pdf')
    
    features = [series for series in data if series[0] == 'N' or series[0] == 'S']
    labels = [series for series in data if series[0] == 'Z']
    correlationMatrix(data, [], [], srcpath + 'reports/basic/overall/' +dr +'full_corr' +suffix +'.csv')
    correlationMatrix(data, features, labels, srcpath + 'reports/basic/overall/' +dr +'label_corr' +suffix +'.csv')
    return

def correlationMatrix(data, features, labels, outputDataFile):
    if (not features or not labels):
        #Full correlation matrix
        data.corr().to_csv(outputDataFile)
    else:
        corr = pd.DataFrame()
        for feat in features:
            row = {}
            row['A_name'] = feat
            row.update({label: data[feat].corr(data[label]) for label in labels})
            corr = corr.append(row, ignore_index=True)
        corr.to_csv(outputDataFile)
    return

def overallStats(inputCSV, outputDataFile, outputGraphFile):
    data = pd.read_csv(inputCSV)
    data = data.set_index('Date')
    copy = data.copy(deep=True)

    all_stats = {}
    for col in sorted(copy.columns):
        if (col != 'Y_day' and col != 'Z_dailydir'):
            fixed = copy[col].dropna()     #drop hole values
            all_stats[col] = fixed.describe(percentiles=[.1,.2,.3,.4,.5,.6,.7,.8,.9])
    statsDF = pd.DataFrame(all_stats).transpose()
    statsDF.to_csv(outputDataFile)

    #Plot
    pdf = PdfPages(outputGraphFile)
    for index, row in statsDF.iterrows():
        pyplot.title(index)
        hor = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
        series = [row['min'], row['10%'], row['20%'], row['30%'], row['40%'],
         row['50%'], row['60%'], row['70%'], row['80%'], row['90%']]
        pyplot.plot(hor, series)
        #add points to plot
        pdf.savefig()
        pyplot.cla()
    pdf.close()
    return

def slicedStats(inputCSV, slices, outputDataDir):
    data = pd.read_csv(inputCSV)
    data = data.set_index('Date')

    sliced = {}
    sliceList = []
    for slc in slices:
        start = slc[0]
        end = slc[1]
        label = str(start)+'_to_'+str(end)
        sliceList.append(label)
        sliced[label] = data[start:end] 

    #init
    allStatsSliced = {}
    for col in data:
        if (col[0] != 'Y' and col[0] != 'Z'):
            allStatsSliced[col] = pd.DataFrame()

    for slc in sliced:
        all_stats = {}
        dataSlice = sliced[slc]
        for col in dataSlice:
            if (col[0] != 'Y' and col[0] != 'Z'):
                fixed = dataSlice[col].dropna()     #drop hole values
                stats = fixed.describe().transpose()
                #print(stats)
                allStatsSliced[col] = allStatsSliced[col].append(stats)

    for col in allStatsSliced:
        sent = allStatsSliced[col]
        sent.to_csv(outputDataDir +col +'_sliced.csv')
        pdf = PdfPages(outputDataDir +col +'_sliced.pdf')
        relevant = ['50%', 'mean', 'std', 'min', 'max']
        for stat in relevant:
            pyplot.title(stat)
            pyplot.plot(range(len(sliceList)), sent[stat])
            pdf.savefig()
            pyplot.cla()
        pdf.close()
    return

def labelledStats(data, removeNAs, removeZeros, outputFile):
    try:
        data = data.set_index('Date')
    except:
        pass
    pdf = PdfPages(outputFile)
    colors = {1: "blue", 0: "white", -1: "red", -2: "black"}

    for col in data:
        if (col[0] != 'Z'):
            pyplot.title(col)
            feature = data[col]
            label = data['Z_dailydir'].fillna(-2)
            if (removeNAs):
                noNa = pd.DataFrame({'feature': feature, 'label': label})
                noNa = noNa[noNa.label != -2]   #remove all rows with missing value
                feature = noNa['feature']
                label = noNa['label']
            if (removeZeros):
                noZero = pd.DataFrame({'feature': feature, 'label': label})
                noZero = noZero[noZero.feature != 0] #remove all rows with '0' value
                feature = noZero['feature']
                label = noZero['label']
            dataSpace = np.linspace(0, len(feature)-1, num=len(feature))
            pyplot.scatter(dataSpace, feature, c=[colors[x] for x in label])
            #pyplot.ylim(-1, 1)         
            #pyplot.legend(loc='upper right')
            pdf.savefig()
            pyplot.cla()
    pdf.close()
    return

def seriesHist(series, bins, outputFile):
    pdf = PdfPages(outputFile)
    pyplot.title(series.name)
    pyplot.hist(series.dropna().values, bins=bins)
    pdf.savefig()
    #print(np.std(series.dropna().values))
    pyplot.cla()
    pdf.close()
    return

def infoPurityGraphs(filedir, files):
    numBins = 500
    for datasource in files:
        df = tn.load_csv(Process.g_tsetdir +filedir, datasource +'.csv')
        df.drop('Unnamed: 0', axis=1, inplace=True)     #Get rid of id col
        df.drop('date', axis=1, inplace=True)           #Get rid of date col
        pdf = PdfPages(Process.g_reportdir + 'purityhists/' +filedir +datasource +'.pdf')

        #Split into up, down, and sideways groups
        ups = df.loc[df['label'] == 1]
        downs = df.loc[df['label'] == -1]
        sideways = df.loc[df['label'] == 0]        

        for column in df:
            if (column != 'label'):
                pyplot.title(column)
                pyplot.hist(ups[column].values, bins=numBins, alpha=0.3, label='up')
                pyplot.hist(downs[column].values, bins=numBins, alpha=0.3, label='down')
                pyplot.hist(sideways[column].values, bins=numBins, alpha=0.3, label='sideways')
                pyplot.legend(loc='upper right')
                pdf.savefig()
                pyplot.cla()
        pdf.close()

#Get annual slices from 2000 to 2013
yearly = [('20'+str(i).zfill(2) +'-01-01','20'+str(i+1).zfill(2) +'-12-31') for i in range(13)]
monthly = []

for i in range(14):
    monthly.append(('20'+str(i).zfill(2) +'-01' +'-01',
        '20'+str(i).zfill(2) +'-01' +'-31'))
    
    if (i % 4 != 0):
        monthly.append(('20'+str(i).zfill(2) +'-02' +'-01',
            '20'+str(i).zfill(2) +'-02' +'-28'))
    else:
        monthly.append(('20'+str(i).zfill(2) +'-02' +'-01',
            '20'+str(i).zfill(2) +'-02' +'-29'))

    monthly.append(('20'+str(i).zfill(2) +'-03' +'-01',
        '20'+str(i).zfill(2) +'-03' +'-31'))

    monthly.append(('20'+str(i).zfill(2) +'-04' +'-01',
        '20'+str(i).zfill(2) +'-04' +'-30'))

    monthly.append(('20'+str(i).zfill(2) +'-05' +'-01',
        '20'+str(i).zfill(2) +'-05' +'-31'))

    monthly.append(('20'+str(i).zfill(2) +'-06' +'-01',
        '20'+str(i).zfill(2) +'-06' +'-30'))

    monthly.append(('20'+str(i).zfill(2) +'-07' +'-01',
        '20'+str(i).zfill(2) +'-07' +'-31'))

    monthly.append(('20'+str(i).zfill(2) +'-08' +'-01',
        '20'+str(i).zfill(2) +'-08' +'-31'))

    monthly.append(('20'+str(i).zfill(2) +'-09' +'-01',
        '20'+str(i).zfill(2) +'-09' +'-30'))

    monthly.append(('20'+str(i).zfill(2) +'-10' +'-01',
        '20'+str(i).zfill(2) +'-10' +'-31'))

    monthly.append(('20'+str(i).zfill(2) +'-11' +'-01',
        '20'+str(i).zfill(2) +'-11' +'-30'))

    monthly.append(('20'+str(i).zfill(2) +'-12' +'-01',
        '20'+str(i).zfill(2) +'-12' +'-31'))

if __name__ == "__main__":
    main()
