Directory where datasets from other stages are dumped to.
Furnishes a DataAPI static class to make dumping and loading dataframes simpler and more powerful.

<!-- TODO - method to filter rows in access_util.col_subsetter v2 -->


## Accessors / Access Utils
* Accessors are pairs of JSON files that make it easy to specify datasets
* The directory structure of the directory that houses them is flat
* Generally an accessor of the same name as the accessor category exists
* This root accessor contains all the data, sub-accessors are filtrations of this one

	*********** ROW FILTERING ***********

	*********** TIME INDEX BASED ***********

	Filter all rows before 2018:
		date_range = {
		    'id': ('lt', 2018)
		}
		df.loc[search_df(df, date_range)]

	Filter rows to date range from 2001-02-03 to 2004-05-06
		df.loc['2001-02-03':'2004-05-06']

	Group by four year periods:
		df.groupby(pd.Grouper(freq='4Y'))

	*********** VALUE BASED ***********

	Filter all col rows in range -.1 to .2 inclusive:
		val_range = {
		    'col': ('ine', -.1, .2)
		}
		df.loc[search_df(df, val_range)]

