Raw data acquisition, cleaning/pruning, and joining

Note: 'trmi.json' must be in the parent (project) directory for 'get_trmi.py' to work


<!-- TODO - columns.json rename 'Ave. Price' to 'avg' instead of 'avgPrice' -->


Note: VIX has a significant amount (~3k rows) of premarket data (from 2AM to 8AM).
The row mask clips all rows not within 8AM to 4PM inclusive (local time).
