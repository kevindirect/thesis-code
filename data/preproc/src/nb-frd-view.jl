### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 664a9594-23f9-11ed-02ce-b9139e40c55f
begin
	using Pkg
	Pkg.activate("..")
	using Dates
	using DataFrames
	using DateTimeDataFrames
	using Arrow
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

# ╔═╡ c8f57a8e-cb60-4121-87c7-a771cb38eb10


# ╔═╡ 6caf5c52-8f18-416a-b8a3-6893fef0d69a
setdiff

# ╔═╡ Cell order:
# ╠═664a9594-23f9-11ed-02ce-b9139e40c55f
# ╟─a7b97057-92ff-4346-b303-9a5ffe5389ea
# ╟─2438d9e5-2aa7-40db-829e-0eeb177b2b12
# ╠═24721c0d-d09b-4370-bd69-4b7ecfbfdec0
# ╟─1815386b-b489-4ffd-a757-6bb343f76eb2
# ╠═28cf753c-2a5e-445c-8ea6-6e5fd7abf7aa
# ╟─c8f57a8e-cb60-4121-87c7-a771cb38eb10
# ╠═6caf5c52-8f18-416a-b8a3-6893fef0d69a
