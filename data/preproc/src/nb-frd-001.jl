### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 664a9594-23f9-11ed-02ce-b9139e40c55f
begin
	using Pkg; Pkg.activate("..");
	using CSV
	using Dates
	using DataFrames
	using DateTimeDataFrames
	using Arrow
	using StatsBase
end

# ╔═╡ a7b97057-92ff-4346-b303-9a5ffe5389ea
md"""
## Roadmap
* load raw data ✓

### base preproc ✓
* start with first common day ✓
* remove duplicates off the end ✓
* drop days below threshold nrows ✓
* intersect features on common days ✓
* sort data ✓

### target preproc ✓
* produce returns ✓
* produce rvol targets ✓
* dump as arrow files ✓

### feature preproc
* ffill from first valid bar to last valid bar, for each trading day ✓
* expand index to full trading day range (9:30 to 16:00), propagate Missing ✓
* replace Missing with 0.0 ✓
* dump as arrow files ✓
"""

# ╔═╡ 2438d9e5-2aa7-40db-829e-0eeb177b2b12
md"""
### base preproc
"""

# ╔═╡ 24721c0d-d09b-4370-bd69-4b7ecfbfdec0
begin
	const FREQ = "1min"
	const PATH_DATA = dirname(dirname(@__DIR__))
	const PATH_000_FRD_INDEX = "$PATH_DATA/000/frd/usindex_$(FREQ)_jolhn8"
	const EXT = ".txt"
	const FMT = "yyyy-mm-dd HH:MM:SS"
	const ID = :datetime
	const OHLC = [:open, :high, :low, :close]
	const PRICES = [:SPX, :RUT, :NDX, :DJI]
	const IVOLS = [:VIX, :RVX, :VXN, :VXD]
	const TDAY = Time(09, 30) => Time(16, 00)
	const TDAYLEN = Minute(TDAY[2] - TDAY[1])

	function gettrades000(ticker::Symbol, drop=[:volume];
		path=PATH_000_FRD_INDEX, freq=FREQ, ext=EXT, fmt=FMT)
		CSV.read("$(path)/$(ticker)_$FREQ$ext",
			DataFrame, dateformat=FMT, header=[ID, OHLC...], drop=drop)
	end
end

# ╔═╡ ab2f8bba-17af-4fa0-9a6a-95cdf1b2ff74
# ╠═╡ disabled = true
#=╠═╡
begin
	assets_test = Dict(
		name=>subset(gettrades000(name), t₀, t₁) for name in ASSETS
	);
	ivs_test = Dict(
		name=>subset(gettrades000(name), t₀, t₁) for name in IVS
	);
	first_asset = (df->first(df.datetime)).(values(assets_test))
	first_iv = (df->first(df.datetime)).(values(ivs_test))
	first_date = maximum(vcat(first_asset, first_iv))
	common_start_date = Date(first_date)+Day(1)
	@show string(common_start_date)
end # common_start_date = Date(2007,05,01)
  ╠═╡ =#

# ╔═╡ 3cd72af1-90ef-4517-97f6-0b9ade0be869
"""
Remove trailing duplicates
"""
function filter_tail_dups(df::AbstractDataFrame, τ::Dates.Period=Day(1))
	flt = combine(groupby(df, τ),
		sdf-> (lu = lastunique(sdf)) == last(sdf) ? sdf : subset(sdf, 1, lu))
	select!(flt, Not(1))
end

# ╔═╡ ed49330f-8e59-451d-91f5-a76592715381
"""
Filter out `τ` groups where nrow is below `thresh`
"""
function filter_partials(df::AbstractDataFrame, thresh=ceil(TDAYLEN.value/4), τ=Day(1))
	#daylens = combine(groupby(df, Day(1)), nrow)
	#daylens[daylens.nrow .< thresh, :]
	#subset(groupby(df, τ), )
	#combine(filter(g->nrow(g) > thresh, groupby(df, Day(1))), names(df))
	nr = transform(groupby(df, τ), nrow; ungroup=true)
	select!(subset(nr, :nrow=>ByRow(≥(thresh))), names(df))
end

# ╔═╡ c36fed06-832c-4523-8fc9-e4672aef813a
function preproc_base(df::AbstractDataFrame, d₁::Date, τ=Day(1))
	@assert issorted(df, ID)
	time_filtered = subset(subset(df, :≥, d₁), τ, TDAY)
	filter_partials(filter_tail_dups(time_filtered))
end

# ╔═╡ 147c8b33-02df-4716-bff6-1e7a58d196a2
function select_common(df::AbstractDataFrame, common, τ=Day(1))
	gb = groupby(df, τ)
	fkeys = filter(key -> Date(key.bar) in common, keys(gb))
	select(combine(gb[fkeys], All()), Not(:bar))
end

# ╔═╡ 0c25717d-3ee4-4425-bc91-6ef7db74f815
function preproc_intersect(dfs::AbstractDataFrame...; index::Symbol=:datetime)
	dates = [unique(Date.(df[:, index])) for df in dfs]
	commondates = Base.intersect(dates...)
	(sort!(select_common(df, commondates), index) for df in dfs)
end

# ╔═╡ 91ff9704-adc6-44fa-b3cd-27319c076c95
function commonindex(dfs::AbstractDataFrame...; index::Symbol=:datetime)
	sort!(Base.intersect((df[!, index] for df in dfs)...))
end

# ╔═╡ f32cd057-127c-4995-b0a5-72dd85da247d
function assert_base(dfs::AbstractDataFrame...; τ=Day(1))
	@assert allequal([size(groupby(df, τ)) for df in dfs])
end

# ╔═╡ ec32cfae-a6c6-4460-b177-07dc1471038a
begin
	d₀ = Date(2007, 05, 01)
	prices, ivols = Dict(), Dict()
	for (price, ivol) in zip(PRICES, IVOLS)
		price_df = preproc_base(gettrades000(price), d₀)
		ivol_df = preproc_base(gettrades000(ivol), d₀)
		price_df, ivol_df = preproc_intersect(price_df, ivol_df)
		assert_base(price_df, ivol_df)
		prices[price] = price_df
		ivols[price] = ivol_df
	end
	# prices = Dict(
	# 	name=>preproc_base(gettrades000(name), d₀) for name in PRICES
	# );
	# ivols = Dict(
	# 	PRICES[i]=>preproc_base(gettrades000(name), d₀) for (i, name) in enumerate(IVOLS)
	# );
end;

# ╔═╡ c8f57a8e-cb60-4121-87c7-a771cb38eb10
md"""
#### target preproc
"""

# ╔═╡ da1e6fd9-5f99-4241-9d39-710c3f8fbb54
begin
	@inline pctchange(pₜ::T, pₜ₊₁::T) where T<:Real = pₜ₊₁ / pₜ - 1
	@inline pctchange(pₜ::AbstractVector{T}) where T<:Real = diff(pₜ) ./ pₜ[begin:end-1]
	@inline pctchange(pₜ::AbstractVector{T}, pₜ₊₁::AbstractVector{T}) where T<:Real = pctchange.(pₜ, pₜ₊₁)
end

# ╔═╡ 2810cb0b-48ca-4797-9fb8-4a5c190133a7
begin
	@inline logchange(pₜ::T, pₜ₊₁::T) where T<:Real = log(pₜ₊₁)-log(pₜ)
	@inline logchange(pₜ::AbstractVector{T}) where T<:Real = diff(log.(pₜ))
	@inline logchange(pₜ::AbstractVector{T}, pₜ₊₁::AbstractVector{T}) where T<:Real = logchange.(pₜ, pₜ₊₁)
end

# ╔═╡ bf4eb598-f890-43d5-a19d-4f9bd4ee7951
"""
Realized Volatility definition from Müller's book (Dacorogna, p. 41)
The 'nth root generalization' of the root mean square (RMS).

Generally `n` is set to 2 provide a good balance of weighting the tails
of the distribution and is generally below the tail index of the distribution
for high frequency data (Darocgna, p.44).
"""
function rvol(pₜ::AbstractVector{T}, n::Integer=2) where T<:Real
	rₜ = logchange(pₜ)
	rₜ = n==1 ? abs.(rₜ) : rₜ
	mean(rₜ.^n)^(1/n)
end

# ╔═╡ f0d22042-4fbe-4a22-81c0-0e511936f09d
function preproc_targets(df::AbstractDataFrame, τ::Dates.Period=Day(1))
	select(groupby(df, τ),
		:bar=>ID,

		:close => (c -> pctchange(first(c), last(c))) => :ret_daily_R,
		:close => (c -> logchange(first(c), last(c))) => :ret_daily_r,

		[:high, :low] => ((h, l) -> logchange(minimum(l), maximum(h))) => :rvol_daily_hl,
		:close => (c -> abs(logchange(first(c), last(c)))) => :rvol_daily_r_abs,
		:close => (c -> logchange(first(c), last(c))^2) => :rvol_daily_r²,
		:close => rvol => :rvol_minutely_r_rms,
		:close => (c -> std(logchange(c))) => :rvol_minutely_r_std,
		:close => (c -> var(logchange(c))) => :rvol_minutely_r_var,
		:close => (c -> sum(logchange(c).^2)) => :rvol_minutely_r²_sum,
	; keepkeys=false) |> unique |> disallowmissing!
end

# ╔═╡ 9f7fee8a-713a-4bfe-b496-08f395719144
begin
	for (name, df) in prices
		dest1 = mkpath("../../001/frd/$name/target")
		Arrow.write("$dest1/price.arrow", preproc_targets(df))
	end
end;

# ╔═╡ 789c68cb-c2c2-4076-bfe4-b7d14e32547f
md"""
#### feature preproc
"""

# ╔═╡ f18aab18-59e3-45c7-a01d-1a4bfa202959
notagg(df::AbstractDataFrame) = select(df, Not(DateTimeDataFrames.AGG_DT))

# ╔═╡ 552318bc-c1cc-4e22-98aa-9b78c495dad7
function expandtday(df::AbstractDataFrame, τ::Dates.Period=Minute(1))
	expandin = expand(notagg(df), τ) # expand and fill within valid bars
	expandout = expandindex(expandin, τ, TDAY) # expand index and propagate missing
end

# ╔═╡ 0f61c4c6-44a6-444f-b897-c5c32127d608
function assert_features(df::AbstractDataFrame, τ::Dates.Period=Day(1))
	lasttimes = Time.(select(combine(groupby(df, τ), last; ungroup=true), ID)) |> unique
	@assert nrow(lasttimes) == 1 && lasttimes[1, 1] == TDAY[2]
	firsttimes = Time.(select(combine(groupby(df, τ), first; ungroup=true), ID)) |> unique
	@assert nrow(firsttimes) == 1 && firsttimes[1, 1] == TDAY[1]
end

# ╔═╡ d680e43d-636b-40e1-9e00-4d90b69f6772
function preproc_features(df::AbstractDataFrame, τ::Dates.Period=Day(1))
	pf = notagg(combine(groupby(df, τ), expandtday))
	pf = coalesce.(pf, 0.0)
	disallowmissing!(pf)
	assert_features(pf)
	pf
end

# ╔═╡ 737b2890-0a96-4ab5-9099-54a6aea72b48
begin
	for (name, df) in prices
		dest2 = mkpath("../../001/frd/$name/feature")
		Arrow.write("$dest2/price.arrow", preproc_features(df))
	end
end;

# ╔═╡ 75a7ec76-8396-4636-a0cd-7a4d67b2555f
begin
	for (name, df) in ivols
		dest3 = mkpath("../../001/frd/$name/feature")
		Arrow.write("$dest3/ivol.arrow", preproc_features(df))
	end
end;

# ╔═╡ Cell order:
# ╠═664a9594-23f9-11ed-02ce-b9139e40c55f
# ╟─a7b97057-92ff-4346-b303-9a5ffe5389ea
# ╟─2438d9e5-2aa7-40db-829e-0eeb177b2b12
# ╠═24721c0d-d09b-4370-bd69-4b7ecfbfdec0
# ╟─ab2f8bba-17af-4fa0-9a6a-95cdf1b2ff74
# ╠═3cd72af1-90ef-4517-97f6-0b9ade0be869
# ╠═ed49330f-8e59-451d-91f5-a76592715381
# ╠═c36fed06-832c-4523-8fc9-e4672aef813a
# ╠═147c8b33-02df-4716-bff6-1e7a58d196a2
# ╠═0c25717d-3ee4-4425-bc91-6ef7db74f815
# ╠═91ff9704-adc6-44fa-b3cd-27319c076c95
# ╠═f32cd057-127c-4995-b0a5-72dd85da247d
# ╠═ec32cfae-a6c6-4460-b177-07dc1471038a
# ╟─c8f57a8e-cb60-4121-87c7-a771cb38eb10
# ╠═da1e6fd9-5f99-4241-9d39-710c3f8fbb54
# ╠═2810cb0b-48ca-4797-9fb8-4a5c190133a7
# ╠═bf4eb598-f890-43d5-a19d-4f9bd4ee7951
# ╠═f0d22042-4fbe-4a22-81c0-0e511936f09d
# ╠═9f7fee8a-713a-4bfe-b496-08f395719144
# ╟─789c68cb-c2c2-4076-bfe4-b7d14e32547f
# ╠═f18aab18-59e3-45c7-a01d-1a4bfa202959
# ╠═552318bc-c1cc-4e22-98aa-9b78c495dad7
# ╠═d680e43d-636b-40e1-9e00-4d90b69f6772
# ╠═0f61c4c6-44a6-444f-b897-c5c32127d608
# ╠═737b2890-0a96-4ab5-9099-54a6aea72b48
# ╠═75a7ec76-8396-4636-a0cd-7a4d67b2555f
