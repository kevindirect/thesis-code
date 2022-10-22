### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 664a9594-23f9-11ed-02ce-b9139e40c55f
begin
	using Pkg; Pkg.activate("..");
	using Dates
	using StatsBase
	using DataFrames
	using DateTimeDataFrames
	using Arrow
end

# ╔═╡ 6465ad4b-c0f6-439f-860c-3688436bca70
begin
	using Plots
	timenow = Time(now())
	if timenow > Time(ENV["TIMENIGHT"]) || timenow < Time(ENV["TIMEDAY"])
		theme(:orange)
	end
end

# ╔═╡ a7b97057-92ff-4346-b303-9a5ffe5389ea
md"""
## Roadmap
* Check Data
"""

# ╔═╡ 2438d9e5-2aa7-40db-829e-0eeb177b2b12
md"""
#### Settings
"""

# ╔═╡ 24721c0d-d09b-4370-bd69-4b7ecfbfdec0
begin
	const PATH_DATA = dirname(dirname(@__DIR__))
	const PATH_002_FRD_INDEX = "$PATH_DATA/002/frd/"
	const ASSETS = [:SPX, :RUT, :NDX, :DJI]
	const TDAY = Time(09, 30) => Time(16, 00)
	const TDAYLEN = Minute(TDAY[2] - TDAY[1])
	const TARGETΔ = -1 # shift index backward in time by one slot
	const TRAIN_RATIO = .6

	function gettrades(extpath; basepath=PATH_002_FRD_INDEX)
		DataFrame(Arrow.Table("$basepath$extpath.arrow"))
	end
end

# ╔═╡ 1815386b-b489-4ffd-a757-6bb343f76eb2
md"""
#### Train / Val / Test Splits
"""

# ╔═╡ 28cf753c-2a5e-445c-8ea6-6e5fd7abf7aa
begin
	for asset in ASSETS
		for split in ["train", "val", "test"]
			df1 = gettrades("$asset/$split/feature/price")
			df2 = gettrades("$asset/$split/feature/ivol")
			if size(df1, 1) != size(df2, 1)
				@show asset, split
				@show size(df1, 1)
				@show size(df2, 1)
			end
		end
	end
end

# ╔═╡ 880e9c84-818d-4ca3-b73a-4c618a9a1319
df = gettrades("SPX/train/feature/price")

# ╔═╡ 1b3d370b-135c-4ced-82e3-05ffb4e41924
theme(:orange)

# ╔═╡ 4239c9ac-0e04-4469-bf53-9b962a00f68f
begin
	l = log.(dropmissing(df)[!, :close])
	plot(l)
	hline!([mean(l)])
end

# ╔═╡ 0ce8c66c-7357-46fa-b845-095bbd949c61
begin
	day = Date(2010, 03, 3)
	dmdf = dropmissing(subset(df, :∈, day))
	fn = log
	z = OHLC[(fn(r[1]), fn(r[2]), fn(r[3]), fn(r[4])) for r in eachrow(dmdf[:, [:open, :high, :low, :close]])]
	#z = OHLC[(dmdf[!, :open], dmdf[!, :high], dmdf[!, :low], dmdf[!, :close])]
	#typeof(z)
	ohlc(z)
	title!(string(day))
end

# ╔═╡ 156fc870-bb0e-4917-804b-e8c39f107704


# ╔═╡ 67b7bea7-4a9d-4615-9be6-64a3eab2cc77
begin
	meanprices = cumsum(randn(100))
	y = OHLC[(p+rand(),p+1,p-1,p+rand()) for p in meanprices]
	#typeof(y)
	ohlc(y)
end

# ╔═╡ a2352238-6456-4121-a015-50ab0e847f0d
begin
	#plot((l .- mean(l)) ./ std(l))
end

# ╔═╡ Cell order:
# ╠═664a9594-23f9-11ed-02ce-b9139e40c55f
# ╠═6465ad4b-c0f6-439f-860c-3688436bca70
# ╟─a7b97057-92ff-4346-b303-9a5ffe5389ea
# ╟─2438d9e5-2aa7-40db-829e-0eeb177b2b12
# ╠═24721c0d-d09b-4370-bd69-4b7ecfbfdec0
# ╟─1815386b-b489-4ffd-a757-6bb343f76eb2
# ╠═28cf753c-2a5e-445c-8ea6-6e5fd7abf7aa
# ╠═880e9c84-818d-4ca3-b73a-4c618a9a1319
# ╠═1b3d370b-135c-4ced-82e3-05ffb4e41924
# ╠═4239c9ac-0e04-4469-bf53-9b962a00f68f
# ╠═0ce8c66c-7357-46fa-b845-095bbd949c61
# ╠═156fc870-bb0e-4917-804b-e8c39f107704
# ╠═67b7bea7-4a9d-4615-9be6-64a3eab2cc77
# ╠═a2352238-6456-4121-a015-50ab0e847f0d
