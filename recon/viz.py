"""
Kevin Patel
"""
import sys
import os
import logging
from collections import Mapping

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
import scipy.stats

from common_util import benchmark, isnt
from recon.common import dum


""" ********** GLOBAL SETTINGS ********** """
font = {
	'family' : 'inconsolata',
	'weight' : 'normal',
	'size'   : 28
}

matplotlib.rc('font', **font)


""" ********** PANEL DATA VISUALIZATION ********** """
# Line Graphs
def plot_df_line(df, title='title', xlabel='xlab', ylabel='ylab', figsize=(25, 10),
	colors=None, linestyles=None):
	plt.figure(figsize=figsize)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		linestyle = linestyles[i % len(linestyles)] if (isinstance(linestyles, list)) else linestyles
		linewidth = None if (isnt(linestyle)) else 2
		plt.plot(df.index, df.loc[:, col_name], color=color, linewidth=linewidth, linestyle=linestyle, label=str(col_name))

	plt.legend(loc='upper left', fancybox=True, framealpha=0.75)

def plot_df_line_subplot(df, ax, title=None, xlabel=None, ylabel=None,
	colors=None, linestyles=None):
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid()

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		linestyle = linestyles[i % len(linestyles)] if (isinstance(linestyles, list)) else linestyles
		linewidth = None if (isnt(linestyle)) else 2
		ax.plot(df.index, df.loc[:, col_name], color=color, linewidth=linewidth, linestyle=linestyle, label=str(col_name))

	ax.legend(loc='upper left', fancybox=True, framealpha=0.75)

# Scatterplots
def plot_df_scatter(df, title='title', xlabel='xlab', ylabel='ylab', figsize=(25, 10),
	colors=None, alpha=None, markers='.'):
	plt.figure(figsize=figsize)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		marker = markers[i % len(markers)] if (isinstance(markers, list)) else markers
		plt.scatter(df.index, df.loc[:, col_name], color=color, alpha=alpha, marker=marker, label=str(col_name))

	plt.legend(loc='upper left', fancybox=True, framealpha=0.75)

def plot_df_scatter_subplot(df, ax, title=None, xlabel=None, ylabel=None,
	colors=None, alpha=None, markers='.'):
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.grid()

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		marker = markers[i % len(markers)] if (isinstance(markers, list)) else markers
		ax.scatter(df.index, df.loc[:, col_name], color=color, alpha=alpha, marker=marker, label=str(col_name))

	ax.legend(loc='upper left', fancybox=True, framealpha=0.75)

# Histograms
def plot_df_hist(df, title='title', xlabel='xlab', ylabel='frequency', figsize=(25, 10),
	colors=None, alpha=None, hist_bins=10):
	plt.figure(figsize=figsize)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		plt.hist(df.loc[:, col_name], bins=hist_bins, color=color, alpha=alpha, label=str(col_name))

	plt.legend(loc='upper left', fancybox=True, framealpha=0.75)

def plot_df_hist_subplot(df, ax, title=None, xlabel=None, ylabel=None,
	colors=None, alpha=None, hist_bins=10):
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.yaxis.grid()
    
	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		ax.hist(df.loc[:, col_name], bins=hist_bins, color=color, alpha=alpha, label=str(col_name))

	ax.legend(loc='upper left', fancybox=True, framealpha=0.75)

def plot_df_dist(df, col_name, fit_overlay=False):
	series = df[col_name].dropna().sort_values(inplace=False)
	plt.figure(figsize=(4,4))
	plt.title(col_name)
	plt.xlabel('value')
	plt.ylabel('number of records')
	plt.grid(b=True, which='major', axis='y')
	plt.hist(series, bins=80, normed=True)

	if (fit_overlay):
		size, mean, median, sdev = series.size, np.mean(series), np.median(series), np.std(series)
		textstr = '$n=%i$\n$\mu=%.2f$\n$\mathrm{median}=%.2f$\n$\sigma=%.2f$'%(size, mean, median, sdev)
		fit = scipy.stats.norm.pdf(series, mean, sdev)

		props = dict(boxstyle='round', facecolor='white', alpha=0.5)	# matplotlib.patch.Patch properties
		plt.axes().text(0.05, 0.95, textstr, transform=plt.axes().transAxes,
			fontsize=14, verticalalignment='top', bbox=props)	# place a text box in upper left
		plt.plot(series, fit,'k^')

# def plot_df_heatmap(df):
# 	# plot correlation matrix
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	cax = ax.matshow(df, vmin=-1, vmax=1)
# 	fig.colorbar(cax)
# 	ticks = np.arange(0,9,1)
# 	ax.set_xticks(ticks)
# 	ax.set_yticks(ticks)
# 	ax.set_xticklabels(names)
# 	ax.set_yticklabels(names)
# 	plt.show()

# Heatmaps
def plot_df_heatmap(df, figsize=(10,10), cmap='cividis', aspect='auto'):
	"""
	Plot a pd.DataFrame as heatmap with matplotlib.
	"""
	fig, ax = plt.subplots(figsize=figsize)
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.10)

	im = ax.imshow(df, cmap=cmap, aspect=aspect)
	fig.colorbar(im, cax=cax, orientation='vertical')

	ax.set_xticks(range(len(df.columns)))
	ax.set_yticks(range(len(df.index)))
	ax.set_xticklabels(df.columns)
	ax.set_yticklabels(df.index)
	return fig

def plot_dfs_heatmap(dfs, row_labels=None, col_labels=None, bars_for_all=False,
	sharex=False, sharey=True, figsize=(20,10), cmap='cividis', aspect='auto'):
	"""
	Given a 2D iterable of pd.DataFrames (rows, columns), creates plots a 2D grid of heatmaps with matplotlib.
	"""
	rows, cols = len(dfs), len(dfs[0])
	fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=sharex, sharey=sharey, squeeze=False, figsize=figsize)

	for row in range(rows):
		for col in range(cols):
			df = dfs[row][col]
			ax = axes[row][col]
			im = ax.imshow(df, cmap=cmap, aspect=aspect)
			ax.set_xticks(range(len(df.columns)))
			ax.set_yticks(range(len(df.index)))
			ax.set_xticklabels(df.columns)
			ax.set_yticklabels(df.index)
			if (row_labels and col == 0):
				ax.set_ylabel(row_labels[row], rotation=0, size='large')
			if (col_labels and row == 0):
				ax.set_title(col_labels[col])
			if (bars_for_all or col == cols-1):
				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='5%', pad=.2 if (bars_for_all) else .5)
				fig.colorbar(im, cax=cax, orientation='vertical')
	return fig


## XXX EVERYTHING PAST THIS POINT IS DEPRECATED

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

def ploty(y, x_start=0, x_step=1):
	y = np.array(y)
	x = np.arange(x_start, y.size, x_step)
	plt.plot(x, y)
	plt.show()

def plot_ser(ser, title='plot', xlabel='xlab', ylabel='ylab'):
	plt.figure(figsize=((25, 10)))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(ser.index, ser)
	plt.show()


def plot_day_ser(ser, date=None):
	if (date is None):
		date = str(np.random.choice(ser.index))[:10]
	plot_ser(ser[date], title=date, xlabel='hour', ylabel='value')


def plot_df(df, title='plot', xlabel='xlab', ylabel='ylab', marker='-'):
	plt.figure(figsize=((25, 10)))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	[plt.plot(df.index, df.loc[:, col_name], marker, label=str(col_name))
		for col_name in df.columns]
	plt.legend(loc='upper left')
	return plt


def plot_day_df(df, date=None):
	if (date is None):
		date = str(np.random.choice(df.index))[:10]
	plot_df(df[date], title=date, xlabel='hour', ylabel='value')


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


