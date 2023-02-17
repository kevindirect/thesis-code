### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 664a9594-23f9-11ed-02ce-b9139e40c55f
begin
	using Pkg; Pkg.activate("..");
	using Dates
	using DataFrames
	using DateTimeDataFrames
	using Arrow
end

# ╔═╡ a7b97057-92ff-4346-b303-9a5ffe5389ea
md"""
## Roadmap
* split into train/val/test ✓
* ~~shift targets forward in time ✓~~
* dump as arrow files ✓
"""

# ╔═╡ 2438d9e5-2aa7-40db-829e-0eeb177b2b12
md"""
#### Settings
"""

# ╔═╡ 24721c0d-d09b-4370-bd69-4b7ecfbfdec0
begin
	const PATH_DATA = dirname(dirname(@__DIR__))
	const PATH_001_FRD_INDEX = "$PATH_DATA/001/frd/"
	const ASSETS = [:SPX, :RUT, :NDX, :DJI]
	const TDAY = Time(09, 30) => Time(16, 00)
	const TDAYLEN = Minute(TDAY[2] - TDAY[1])
	# const TARGETΔ = -1 # shift index backward in time by one slot
	const TARGETΔ = 0 # no shifting here
	const TRAIN_RATIO = .6

	function gettrades001(extpath; basepath=PATH_001_FRD_INDEX)
		DataFrame(Arrow.Table("$basepath$extpath.arrow"))
	end
end

# ╔═╡ 1815386b-b489-4ffd-a757-6bb343f76eb2
md"""
#### Train / Val / Test Splits
"""

# ╔═╡ 0e003f50-ba33-4d16-83f9-afb29d3e86fb
function getsplits(df::AbstractDataFrame, train_end, val_range, test_start)
	subset(df, :≤, train_end), subset(df, val_range), subset(df, :≥, test_start)
end

# ╔═╡ 9c67d0df-aa2e-468b-b0bc-d61598ed1fce
function getsplits_points(df::AbstractDataFrame, train_ratio=TRAIN_RATIO; index=:datetime)
	train_lastrow = nrow(df)*TRAIN_RATIO
	val_lastrow = nrow(df)*(TRAIN_RATIO+((1-TRAIN_RATIO)/2))

	train_end = Date(df[floor(Int, train_lastrow), index])
	val_range = Date(df[ceil(Int, train_lastrow), index]) => Date(df[floor(Int, val_lastrow), index])
	test_start = Date(df[ceil(Int, val_lastrow), index])

	#@show train_end, val_range, test_start
	train_end, val_range, test_start
end

# ╔═╡ 9b98aedb-8d1f-4712-9cee-264bd9661731
function getsplits(asset::Symbol, train_ratio=TRAIN_RATIO, target_shift=TARGETΔ; index=:datetime)
	feature_price = gettrades001("$asset/feature/price")
	feature_ivol = gettrades001("$asset/feature/ivol")
	feature_logprice = gettrades001("$asset/feature/logprice")
	feature_logivol = gettrades001("$asset/feature/logivol")
	feature_logchangeprice = gettrades001("$asset/feature/logchangeprice")
	feature_logchangeivol = gettrades001("$asset/feature/logchangeivol")
	target_price = gettrades001("$asset/target/price")

	train_end, val_range, test_start = getsplits_points(target_price, train_ratio; index=index)
	feature_price_split = getsplits(feature_price, train_end, val_range, test_start)
	feature_ivol_split = getsplits(feature_ivol, train_end, val_range, test_start)
	feature_logprice_split = getsplits(feature_logprice, train_end, val_range, test_start)
	feature_logivol_split = getsplits(feature_logivol, train_end, val_range, test_start)
	feature_logchangeprice_split = getsplits(feature_logchangeprice, train_end, val_range, test_start)
	feature_logchangeivol_split = getsplits(feature_logchangeivol, train_end, val_range, test_start)
	target_price_split = getsplits(target_shift==0 ? target_price : shift(target_price, target_shift), train_end, val_range, test_start)
	test_end = Date(target_price_split[3][end, index])

	(
		train = (
			feature = (
				price = feature_price_split[1],
				ivol = feature_ivol_split[1],
				logprice = feature_logprice_split[1],
				logivol = feature_logivol_split[1],
				logchangeprice = feature_logchangeprice_split[1],
				logchangeivol = feature_logchangeivol_split[1]
			),
			target = (
				price = target_price_split[1],
			)
		),
		val = (
			feature = (
				price = feature_price_split[2],
				ivol = feature_ivol_split[2],
				logprice = feature_logprice_split[2],
				logivol = feature_logivol_split[2],
				logchangeprice = feature_logchangeprice_split[2],
				logchangeivol = feature_logchangeivol_split[2]
			),
			target = (
				price = target_price_split[2],
			)
		),
		test = (
			feature = (
				price = subset(feature_price_split[3], :≤, test_end),
				ivol = subset(feature_ivol_split[3], :≤, test_end),
				logprice = subset(feature_logprice_split[3], :≤, test_end),
				logivol = subset(feature_logivol_split[3], :≤, test_end),
				logchangeprice = subset(feature_logchangeprice_split[3], :≤, test_end),
				logchangeivol = subset(feature_logchangeivol_split[3], :≤, test_end)
			),
			target = (
				price = target_price_split[3],
			)
		)
	)
end

# ╔═╡ c8f57a8e-cb60-4121-87c7-a771cb38eb10
md"""
#### Dump Splits
"""

# ╔═╡ 9f7fee8a-713a-4bfe-b496-08f395719144
for name in ASSETS
	dfs = getsplits(name);
	for split in [:train, :val, :test]
		destf = mkpath("../../002/frd/$name/$split/feature")
		destt = mkpath("../../002/frd/$name/$split/target")
		Arrow.write("$destf/price.arrow", dfs[split].feature.price)
		Arrow.write("$destf/ivol.arrow", dfs[split].feature.ivol)
		Arrow.write("$destf/logprice.arrow", dfs[split].feature.logprice)
		Arrow.write("$destf/logivol.arrow", dfs[split].feature.logivol)
		Arrow.write("$destf/logchangeprice.arrow", dfs[split].feature.logchangeprice)
		Arrow.write("$destf/logchangeivol.arrow", dfs[split].feature.logchangeivol)
		Arrow.write("$destt/price.arrow", dfs[split].target.price)
	end
end

# ╔═╡ Cell order:
# ╠═664a9594-23f9-11ed-02ce-b9139e40c55f
# ╟─a7b97057-92ff-4346-b303-9a5ffe5389ea
# ╟─2438d9e5-2aa7-40db-829e-0eeb177b2b12
# ╠═24721c0d-d09b-4370-bd69-4b7ecfbfdec0
# ╟─1815386b-b489-4ffd-a757-6bb343f76eb2
# ╠═0e003f50-ba33-4d16-83f9-afb29d3e86fb
# ╠═9c67d0df-aa2e-468b-b0bc-d61598ed1fce
# ╠═9b98aedb-8d1f-4712-9cee-264bd9661731
# ╟─c8f57a8e-cb60-4121-87c7-a771cb38eb10
# ╠═9f7fee8a-713a-4bfe-b496-08f395719144
