# Kevin Patel

import sys
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, vectorize

from common_util import search_df, get_subset, benchmark
from data.data_api import DataAPI
from recon.common import dum


def plot_dist(col_name):
	series = data[col_name].dropna().sort_values(inplace=False)
	mean = np.mean(series)
	median = np.median(series)
	sdev = np.std(series)
	fit = sc.stats.norm.pdf(series, mean, sdev)

	plt.figure(figsize=(4,4))
	plt.title(col_name)
	plt.xlabel('value')
	plt.ylabel('number of records')
	plt.grid(b=True, which='major', axis='y')
	textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(mean, median, sdev)

	# these are matplotlib.patch.Patch properties
	props = dict(boxstyle='round', facecolor='white', alpha=0.5)

	# place a text box in upper left in axes coords
	plt.axes().text(0.05, 0.95, textstr, transform=plt.axes().transAxes, fontsize=14,
		verticalalignment='top', bbox=props)

	plt.hist(series, bins=80, normed=True)
	plt.plot(series, fit,'k^')


def plot_heatmap(df):
	# plot correlation matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(df, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,9,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	plt.show()


#write plot_split_hists function
#plots histograms of the feature segmented by the label
def infoPurityGraphs(filedir, files):
	numBins = 500
	for datasource in files:
		df = tn.load_csv(Process.g_tsetdir +filedir, datasource +'.csv')
		df.drop('Unnamed: 0', axis=1, inplace=True)		#Get rid of id col
		df.drop('date', axis=1, inplace=True)			#Get rid of date col
		pdf = PdfPages(Process.g_reportdir + 'purityhists/' +filedir +datasource +'.pdf')

		#Split into up, down, and sideways groups
		ups = df.loc[df['label'] == 1]
		downs = df.loc[df['label'] == -1]
		sideways = df.loc[df['label'] == 0]

		for column in df:
			if (column != 'label'):
				plt.title(column)
				plt.hist(ups[column].values, bins=numBins, alpha=0.3, label='up')
				plt.hist(downs[column].values, bins=numBins, alpha=0.3, label='down')
				plt.hist(sideways[column].values, bins=numBins, alpha=0.3, label='sideways')
				plt.legend(loc='upper right')
				pdf.savefig()
				plt.cla()
		pdf.close()

def plot_feature(feature=None, label=None,
				remove_nans=True, remove_zeros=True,											#pre-processing options
				atomic_transform=lambda x:x, delta_transform=lambda x:x, combo='delta_atomic',	#processing options
				low_clip=-1.0, high_clip=1.0,
				show_class='all'):																#post-processing options

	assert(feature and label)
	#TODO - option for gradient coloring
	colors = {1: "blue", 0: "white", -1: "red", -2: "black"}
	output = pd.DataFrame({feature: data[feature], label: data[label], 'year': data['year']})

	#Drop rows with no label
	output.dropna(axis=0, subset=[label], inplace=True)

	#Pre processing
	output[feature].fillna(value=-2, axis=0, inplace=True,)
	if (remove_nans):
		output = output[output[feature] != -2]
	if (remove_zeros):
		output = output[output[feature] != 0]
		
	#TODO - upper and lower bound filtering (throw out middle)
		#past_period - number of rows behind the "big moves" to keep in
		#Need this for delta transforms
	
	if (combo == 'atomic'):
		output[combo] = atomic_transform(output[feature])
	elif (combo == 'delta'):
		output[combo] = delta_transform(output[feature])
	elif (combo == 'atomic_delta'):
		output[combo] = atomic_transform(delta_transform(output[feature]))
	elif (combo == 'delta_atomic'):
		output[combo] = delta_transform(atomic_transform(output[feature]))
	else:
		output[combo] = output[feature]
	low_clip = low_clip if low_clip > output[combo].min() else output[combo].min()
	high_clip = high_clip if high_clip < output[combo].max() else output[combo].max()
	output[combo] = np.clip(output[combo], low_clip, high_clip)

	#Post processing
	if (show_class != 'all'):
		if (show_class == 'up'):
			output = output[output[label] == 1]
		elif (show_class == 'down'):
			output = output[output[label] == -1]
		elif (show_class == 'sideways'):
			output = output[output[label] == 0]
	
	#TODO - add year specific removal option

	data_space = np.linspace(0, len(output[combo])-1, num=len(output[combo]))
	plt.figure(figsize=(25,10))
	#plt.figure().tight_layout()
	plt.title('Data Segmentation Graph')
	plt.xlabel('Observation')
	plt.ylabel('Feature Value')
	plt.xlim([-100, 4100])
	plt.grid(b=True, which='major', axis='y')
	#textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(mean, median, sdev)
	plt.scatt



def i_plot_feature(feature=None, label=None,
				remove_zeros=True,																#pre-processing options
				atomic_transform=lambda x:x, delta_transform=lambda x:x, combo='delta_atomic',  #processing options
				low_clip=-1.0, high_clip=1.0,
				show_class='all'):																#post-processing options

	assert(feature and label)
	output = pd.DataFrame({feature: data[feature], label: data[label], 'year': data['year']})
	output.dropna(axis=0, subset=[label], inplace=True) #Drop rows with no label
	output.dropna(axis=0, subset=[feature], inplace=True)
	output[label] = output[label].astype(int)
	
	if (remove_zeros):
		output = output[output[feature] != 0]

	#TODO - upper and lower bound filtering (throw out middle)
		#past_period - number of rows behind the "big moves" to keep in
		#Need this for delta transforms

	if (combo == 'atomic'):
		output[combo] = atomic_transform(output[feature])
	elif (combo == 'delta'):
		output[combo] = delta_transform(output[feature])
	elif (combo == 'atomic_delta'):
		output[combo] = atomic_transform(delta_transform(output[feature]))
	elif (combo == 'delta_atomic'):
		output[combo] = delta_transform(atomic_transform(output[feature]))
	else:
		output[combo] = output[feature]

	low_clip = low_clip if low_clip > output[combo].min() else output[combo].min()
	high_clip = high_clip if high_clip < output[combo].max() else output[combo].max()
	output[combo] = np.clip(output[combo], low_clip, high_clip)

	#Post processing
	if (show_class != 'all'):
		if (show_class == 'up'):
			output = output[output[label] > 1]
		elif (show_class == 'down'):
			output = output[output[label] < -1]
		elif (show_class == 'sideways'):
			output = output[output[label] == 0]
	
	#TODO - add year specific removal option

	data_space = np.linspace(0, len(output[combo])-1, num=len(output[combo]))
	plt.figure(figsize=(25,10))
	#plt.figure().tight_layout()
	plt.title('Data Segmentation Graph')
	plt.xlabel('Observation')
	plt.ylabel('Feature Value')
	plt.xlim([data_space.min()-100, data_space.max() + 100])
	plt.grid(b=True, which='major', axis='y')
	#textstr = '$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(mean, median, sdev)

	norm = mpl.colors.Normalize(vmin=output[label].min(), vmax=output[label].max()) 
	plt.scatter(data_space, output[combo], c=norm(output[label]), cmap=plt.cm.jet_r)
	plt.colorbar()
	plt.show()

	sent_means = []
	sent_vars = []
	combo_means = []
	combo_vars = []
	class_values = sorted(output[label].unique())
	for class_value in class_values:
		sent_means.append(output.loc[output[label] == class_value, feature].mean())
		combo_means.append(output.loc[output[label] == class_value, combo].mean())
		sent_vars.append(output.loc[output[label] == class_value, feature].var())
		combo_vars.append(output.loc[output[label] == class_value, combo].var())
	fig = plt.figure()
	fig, ax = plt.subplots(1, 2, figsize=(15, 5))
	fig.suptitle('stats segmented by class label')
	ax[0].set_title('mean')
	ax[0].set_ylabel('feature value')
	ax[0].plot(class_values, sent_means, 'ok', label='sent_means')
	ax[0].plot(class_values, combo_means, 'r^', label='combo_means')
	ax[0].legend()
	
	ax[1].set_title('variance')
	ax[1].set_xlabel('label value')
	ax[1].plot(class_values, sent_vars, 'ok', label='sent_vars')
	ax[1].plot(class_values, combo_vars, 'r^', label='combo_vars')
	ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	plt.show()
	return


def plot_ser(ser, title_str='plot', xlabel_str='xlab', ylabel_str='ylab'):
	plt.figure(figsize=((25, 10)))
	plt.title(title_str)
	plt.xlabel(xlabel_str)
	plt.ylabel(ylabel_str)
	plt.plot(ser.index, ser)
	plt.show()


def plot_day_ser(ser, date=None):
	if (date is None):
		date = str(np.random.choice(ser.index))[:10]
	plot_ser(ser[date], title_str=date, xlabel_str='hour', ylabel_str='value')


def plot_df(df, title_str='plot', xlabel_str='xlab', ylabel_str='ylab'):
	plt.figure(figsize=((25, 10)))
	plt.title(title_str)
	plt.xlabel(xlabel_str)
	plt.ylabel(ylabel_str)
	[plt.plot(df.index, df.loc[:, col_name], '^', label=str(col_name)) for col_name in df.columns]
	plt.legend(loc='upper left')
	plt.show()


def plot_day_df(df, date=None):
	if (date is None):
		date = str(np.random.choice(df.index))[:10]
	plot_df(df[date], title_str=date, xlabel_str='hour', ylabel_str='value')


#config options
# atomic_fn_options = OrderedDict({'nothing': lambda x:x, 'round': lambda a: np.around(a, 2), 'log10':np.log10, 'sine':np.sin, 'sinh':np.sinh})
# delta_fn_options = OrderedDict({'nothing': lambda x:x, 'log10':np.log10, 'sine':np.sin, 'sinh':np.sinh})
# feature_list = tuple(sorted([col for col in data.columns if ((col[0]=='N' or col[0]=='S') and 'buzz' not in col)]))

# interact(plot_feature,
# 			feature=feature_list,
# 			label=tuple(label_list),
# 			remove_zeros=OrderedDict({'remove': True, 'keep': False}),
# 			atomic_transform=atomic_fn_options,
# 			delta_transform=delta_fn_options,
# 			combo=('none','atomic','delta','atomic_delta','delta_atomic'),
# 			low_clip=(-1.0, 0.0, .001),
# 			high_clip=(.0, 1.0, .001),
# 			show_class=('all', 'up', 'down', 'sideways'));