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

# font = {
# 	'family': 'Serif',
# 	'weight': 'medium',
# 	'size': 28
# }

# font = {
# 	'family' : 'Sans',
# 	'weight' : 'medium',
# 	'size'   : 28
# }

matplotlib.rc('font', **font)

def dump_fig(fpath, fig=None):
	plt.savefig(fpath, bbox_inches="tight", transparent=True)
	plt.close(fig)


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
	colors=None, alpha=None, hist_bins=10, density=False):
	plt.figure(figsize=figsize)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		plt.hist(df.loc[:, col_name], bins=hist_bins, color=color, alpha=alpha,
			density=density, label=str(col_name))

	plt.legend(loc='upper left', fancybox=True, framealpha=0.75)

def plot_df_hist_subplot(df, ax, title=None, xlabel=None, ylabel=None,
	colors=None, alpha=None, hist_bins=10, density=False):
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.yaxis.grid()
    
	for i, col_name in enumerate(df.columns):
		color = colors[i % len(colors)] if (isinstance(colors, list)) else colors
		ax.hist(df.loc[:, col_name], bins=hist_bins, color=color, alpha=alpha,
			density=density, label=str(col_name))

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

