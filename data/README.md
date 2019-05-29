Directory where datasets from other stages are dumped to.
Furnishes a DataAPI static class to make dumping and loading dataframes simpler and more powerful.

<!-- TODO - method to filter rows in access_util.col_subsetter v2 -->


## Accessors / Access Utils
* Accessors are pairs of JSON files that make it easy to specify datasets
* The directory structure of the directory that houses them is flat
* Generally an accessor of the same name as the accessor category exists
* This root accessor contains all the data, sub-accessors are filtrations of this one
