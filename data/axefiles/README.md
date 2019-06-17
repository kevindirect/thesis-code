```
#    _|_|    _|      _|  _|_|_|_|
#  _|    _|    _|  _|    _|
#  _|_|_|_|      _|      _|_|_|
#  _|    _|    _|  _|    _|
#  _|    _|  _|      _|  _|_|_|_|
# JSON based data access utility.
```
# Axe
This directory of json files is loaded by axe.
Axe is nothing but a set of user-editable json files and a short script to load them into a dictionary on runtime.
These json objects / dicts are used by the Data_API to load or lazily load the requested data.
Stuff downstream calls data_api methods and passes handles to axefiles in order to load specific data.

Yeah I should've just used Postgres..

# Conventions
The conventions within axe are important to prevent confusion and horror. This was never meant to relay quantities of data a single human
couldn't wrap their head around, but in order to keep that limit as large as possible conventions are important.

* Every directory represents an axefile or axefile group
* Every axefile directory contains one _root axefile_ and zero or more _view axefiles_
* All axefiles are either root axefiles, views on root axefiles, or children of those children (basically a tree structure)

## content
The fields of a dg/cs file are 'all' and 'subsets'. Both are required and nullable. The subsets of the cs (column subsetter) file must be the same as the 'dg' (df getter) if the are subsets.

## crunch/data/axefiles/ directory structure
Every directory in here (crunch/data/axefiles/) is flat, no nested directories. Each directory is named after the root axefile in that
directory. For example the 'ddiff' directory has a 'ddiff' axefile (technically two 'cs_ddiff.json' and 'dg_ddiff.json', but the cs_* and dg_* files are associated - they are going to be combined in a future update. Just think of the two as one for now). All the axefiles within the directory preprend the root axefile name and they are all simply views/filtrations on the original root axefile - no transforms or anything like that. If you want a transform you create a new root axefile (obviously make sure that data exists first). For example ddiff_ohlca contains only the Open High Low Close Average data from the ddiff root axefile. If you take a look you'll see that that is what it is. The dg_ and cs_ files exist for dataframe selecting and column selecting within each dataframe, respectively.

Other files (readmes and whatever) can exist. Just make sure to not have any json files you don't want loaded within axefiles

## the desc key and root axefile subsets
The 'desc' field uniquely identifies each dataframe past the 'raw' stage (it still does if you preprend the 'root' field to it). The desc key is used a lot for searching and filtering data so it is important. If the 'subsets' feature is used in a root axefile, the label for each subset must match the desc keys of the data within. This is usually simple if the subset only has one df in it. For cases where there are multiple dfs in a subset, it must match after any parentheses (and content within them) are removed.

Here is an example from the dffd root axefile (note that the desc field doesn't necesarrily have to be used, the convention just needs to hold for the underlying data):

```
{
	"dffd": {
		"all": {
			"stage": "mutate",
			"type": "dffd"
		},
		"subsets": {
			"pba_dohlca_dffd": {
				"desc": ["pba_dohlca_dffd(0.200000,0.010000)",
					"pba_dohlca_dffd(0.400000,0.010000)",
					"pba_dohlca_dffd(0.600000,0.010000)",
					"pba_dohlca_dffd(0.800000,0.010000)"]
			},
			"vol_dohlca_dffd": {
				"desc": ["vol_dohlca_dffd(0.200000,0.010000)",
					"vol_dohlca_dffd(0.400000,0.010000)",
					"vol_dohlca_dffd(0.600000,0.010000)",
					"vol_dohlca_dffd(0.800000,0.010000)"]
			},
			"trmi2_dffd": {
				"desc": ["trmi2_dffd(0.200000,0.010000)",
					"trmi2_dffd(0.400000,0.010000)",
					"trmi2_dffd(0.600000,0.010000)",
					"trmi2_dffd(0.800000,0.010000)"]
			},
			"trmi3_dffd": {
				"desc": ["trmi3_dffd(0.200000,0.010000)",
					"trmi3_dffd(0.400000,0.010000)",
					"trmi3_dffd(0.600000,0.010000)",
					"trmi3_dffd(0.800000,0.010000)"]
			}
		}
	}
}
```

The desc<->subset label convention is not enforced for view axefiles, but it is encouraged wherever possible. If the subset label does not match the desc field (after variants - parentheses and any content in them - are removed) in a view axefile, it will be used in place of the desc string for any transforms that consume it. This is a useful feature for splitting a dataframe into multiple smaller dataframes.

## axefile naming convention and codes
The names can look strange at first glance, but it was all thought through. Terseness is important because we may have long chains of sequential trasnforms on data. Other than the 'raw' root axefile, all axefiles are preprended by a letter indicating the frequency of the data. This is a time series centric project after all:

* 'd': daily
* 'h': hourly
* Others may be added in the future, including non-time based bars (like tick bars or whatever)

The root axefile does not adhere to the above because we want for it to be able to have different frequencies of data. But even so, when we actually use this data, we get it from another axefile. For example we use 'hohlca' for hourly ohlca from raw instead of using 'raw' itself - this makes things cleaner. Notice that I punked you because everything in ./hohlca/ is just a symlink to a file in ./raw/. The use of symlinks is encouraged whenever possible instead of making hard copies. And remember we can make root axefiles that are filtrations of other root axefiles but we cannot make sub-axefiles that are not filtrations. For new transforms, you must create a new root axefile.

A secondary frequency may be located after the first, oftentimes to indicate the aggregation frequency if relevant to that transform or data subset (so the 'hd' prefix is read 'hourly, daily'. Care was made to make names/codes terse but also be remember-able with mneumonics.

After the frequency info you may have some other qualifications on the bars themselves. For example 'x' means 'expanding'. Below is the full list of codes that describe what axefiles represent:

* 'diff': differenced
* 'dir': direction
* 'ffd': fixed window fracdiff
* 'ret': return
	- 'cret': clipped return
* 'ma': moving average
	- 'sma': simple moving average
	- 'ema: exponential moving average
* 'fb': first break
* 'val': value / score
* 'mx': min-max (as in min-max scaling)
* 'zn': z-score normalization
* 'log': natural log
* 'ohlca': open high low close avergage
* 'rm': row mask
* 'spread': spread (as in spread of two prices)

Using these we can decode our names to plain English. It's kind of like German - large words made by concanetating small ones.
For example 'dxfbcret' is:

'daily expanding first break clipped return'

See? That's the most complex one and it isn't so bad. This method makes it easy to remember what something is without lugging around giant strings of historical transforms. Generally when we add a new root axefile we want to write them in terms of existing codes or modifications therof. You can see in the above we prepended a 'c' to 'ret' to get 'cret': a new code. This is encourgaged, but stick to preprending and not appending to preserve the disambiguitiy among these codes. Ambiguities may happen as more codes are added. The intent is to make this human readable and use-able - not to to make some kind of suboptimal huffman coding system. If ambiugities are found, just add a README to the relevant root axefile directories.


