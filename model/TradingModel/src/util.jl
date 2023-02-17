
function target(asset, split)
	path = joinpath(DATA_DIR, PROC_NAME, VENDOR_NAME, asset, split, "target", "price.arrow")
	df = (path |> Arrow.Table |> DataFrame)[:, [:datetime, :R_1day, :r_1day]]
	df[!, :datetime] = Date.(df[:, :datetime])
	rename!(df, :datetime=>:date)
end

function featurepaths(asset, split, abl, feat)
	expdir = joinpath(EXP_DIR, "manual", "anp")
	glob(joinpath("$PROC_NAME-$asset-*-$feat", abl, ["*" for _ in 1:4]..., "$(split)_pred.csv"), expdir)
end

function loadfeature(path)
	sel = [:it, :pred_mean, :pred_std]
	df = (path |> CSV.File |> DataFrame)[:, sel]
	name = split(split(path, "/")[9], "-")
	asset = name[2]
	rv = split(name[3], "_")[end]
	pfx = join([asset, rv], "_")
	rename!(df, sel .=> [:date, Symbol(pfx, "_", sel[2]), Symbol(pfx, "_", sel[3])])
end

function feature(asset, split, abl, feat="logchangeprice")
	innerjoin(loadfeature.(featurepaths(asset, split, abl, feat))...; on=:date)
end

"""
sign with threshold around (-τ, τ)
"""
@inline sign(x::Number, τ::Number) = ifelse(x >= τ, 1.0, ifelse(x <= -τ, -1.0, 0.0))

"""
sign with threshold around (τₗ, τₕ)
"""
@inline sign(x::Number, τₗ::Number, τₕ::Number) = ifelse(x >= τₕ, 1.0, ifelse(x <= τₗ, -1.0, 0.0))

function getsplit(asset, asset_f, split, abl)
	featdf = feature(replace(asset_f, "all"=>"*"), split, replace(abl, "all"=>"*"))
	comm = innerjoin(featdf, target(asset, split); on=:date)
	idx = comm[:, :date]
	feat = comm[:, occursin.("_pred_", names(comm))] |> Matrix
	tgt = comm[:, :R_1day]
	lbl = sign.(tgt)
	names(featdf)[2:end], idx, feat, tgt, lbl
end

"""
Add a "flat" (no trade) signal column if it doesn't exist
"""
function addflat(ŷ::AbstractMatrix)
	ncol = size(ŷ, 2)
	if ncol == 2
		hcat(ŷ[:, 1], zeros(size(ŷ, 1)), ŷ[:, 2])
	elseif ncol == 3
		ŷ
	else
		throw("wrong number of columns: must be |down|up| or |down|flat|up|")
	end
end

"""
Trading signal
The magnitude of the signal indicates the betsize, 0.0 indicates flat / no bet.
The sign of the signal indicates direction.
* long-short signals are in [-1.0, 1.0]
* long-flat signals are in [0.0, 1.0]
* short-flat signals are in [-1.0, 0.0]
"""
function signal(ŷ::AbstractMatrix; betsize=:score, long=true, short=true)
	@assert size(ŷ, 2) == 3 "ŷ must be three column matrix of [down flat up] scores"
	act = [a[2]-2 for a in argmax(ŷ; dims=2)]
	if !long
		act[act.==1] .= 0
	end
	if !short
		act[act.==-1] .= 0
	end
	if betsize == :score
		bet = maximum(ŷ; dims=2)
		act = act .* bet
	end
	act[:]
end

pnl(R::AbstractVector) = cumsum(R)
pnl(R::AbstractVector, sig::AbstractVector) = cumsum(R .* sig)
sharpe(R::AbstractVector) = mean(R) / std(R)
sharpe(R::AbstractVector, periods::Real) = sharpe(R) * sqrt(periods)

"""
Convert long->flat, short->flat as needed based on trade permissions.
"""
function signls(R::AbstractVector; long=true, short=true)
	sigR = sign.(R)
	if !long
		sigR[sigR.==1] .= 0
	end
	if !short
		sigR[sigR.==-1] .= 0
	end
	sigR
end

"""
Buy & Hold with no overnight risk
"""
function backtest(R::AbstractVector; periods=PERIODS)
	pnl = cumsum(R)
	eqcomp = exp.(cumsum(log1p.(R)))
	nyears = length(eqcomp) / periods
	cagr = exp(log(eqcomp[end])/nyears)-1
	(
		finpnl = pnl[end],
		maxpnl = maximum(pnl),
		minpnl = minimum(pnl),
		finpnlcomp = eqcomp[end]-1,
		# using signls instead of sign won't change the result here, but for consistency's sake:
		accuracy = mean(signls(R; long=true, short=false) .== 1),
		cagr = cagr,
		sharpe = sharpe(R, periods),
		ret = R,
		pnl = pnl,
		pnlcomp = eqcomp .- 1
	)
end

function backtest(R::AbstractVector, sig::AbstractVector; periods=PERIODS, long=true, short=true)
	ret = R .* sig
	pnl = cumsum(ret)
	eqcomp = exp.(cumsum(log1p.(ret)))
	nyears = length(eqcomp) / periods
	cagr = exp(log(eqcomp[end])/nyears)-1
	(
		finpnl = pnl[end],
		maxpnl = maximum(pnl),
		minpnl = minimum(pnl),
		finpnlcomp = eqcomp[end]-1,
		accuracy = mean(signls(R; long=long, short=short) .== sign.(sig)),
		cagr = cagr,
		sharpe = sharpe(ret, periods),
		ret = ret,
		pnl = pnl,
		pnlcomp = eqcomp .- 1
	)
end

function backtest(R::AbstractVector, ŷ::AbstractMatrix)
	(
		lsf = (
			full = backtest(R, signal(ŷ; betsize=:full)),
			score = backtest(R, signal(ŷ; betsize=:score))
		),
		lf = (
			full = backtest(R, signal(ŷ; betsize=:full, short=false); short=false),
			score = backtest(R, signal(ŷ; betsize=:score, short=false); short=false)
		),
		sf = (
			full = backtest(R, signal(ŷ; betsize=:full, long=false); long=false),
			score = backtest(R, signal(ŷ; betsize=:score, long=false); long=false)
		)
	)
end

"""
Get index of the median value in a list of backtests
"""
function getbacktest(backs::AbstractVector, traderules::Symbol, betsize::Symbol, metric::Symbol; choose::Function=StatsBase.median)
	vals = [back[traderules][betsize][metric] for back in backs]
	findfirst(==(choose(vals)), vals)
end

function plotpnls(idx::AbstractVector, backs::AbstractVector, bench, betsize::Symbol; pnltype::Symbol=:pnl, titlepfx=nothing, colors=theme_palette(:default), metric::Symbol=:cagr, alpha=.01, altpaths=true)
	plot(idx, bench[pnltype], label="b&h", color=colors[1], line=(2, :solid))
	for (i, tr, lbl, style) in zip(2:5, [:lsf, :lf, :sf], ["long-short", "long-flat", "short-flat"], [:dashdot, :dash, :dot])
		med = getbacktest(backs, tr, betsize, metric)
		altpaths && plot!(idx, [b[tr][betsize][pnltype] for b in backs], color=colors[i],
			label=false, alpha = alpha)
		plot!(idx, backs[med][tr][betsize][pnltype], color=colors[i], line=(2, style),
			label=lbl, legend=true)
	end
	pnltypestr = pnltype == :pnlcomp ? "compounded" : "non-compounded"
	plot!(title="$titlepfx $pnltypestr profit & loss", xlabel="date", ylabel="net profit")
end

function plotpnls(idx::AbstractVector, backs::AbstractVector, back, bench, betsize::Symbol; pnltype::Symbol=:pnl,  titlepfx=nothing, colors=theme_palette(:default), alpha=.01, altpaths=true)
	plot(idx, bench[pnltype], label="buy&hold", color=colors[1], line=(2, :solid))
	for (i, tr, lbl, style) in zip(2:5, [:lsf, :lf, :sf], ["long-short", "long-flat", "short-flat"], [:dashdot, :dash, :dot])
		altpaths && plot!(idx, [b[tr][betsize][pnltype] for b in backs], color=colors[i],
			label=false, alpha = alpha)
		plot!(idx, back[tr][betsize][pnltype], color=colors[i], line=(2, style),
			label=lbl, legend=true)
	end
	pnltypestr = pnltype == :pnlcomp ? "compounded" : "non-compounded"
	plot!(title="$titlepfx $pnltypestr profit & loss", xlabel="date", ylabel="net profit")
end

"""
Extract metrics from backtest as DataFrame
"""
function metrics(back, asset, asset_f, abl; metrics=METRICS)
	metdf = DataFrame((met, asset, asset_f, abl,
		back.lsf.full[met], back.lsf.score[met],
		back.lf.full[met], back.lf.score[met],
		back.sf.full[met], back.sf.score[met])
		for met in metrics
	)
	rename!(metdf, [:metric, :asset, :assetf, :ablation,
		:lsf_full, :lsf_score,
		:lf_full, :lf_score,
		:sf_full, :sf_score]
	)
end

"""
Metrics of benchmark
"""
function metrics(back, asset; metrics=METRICS)
	metdf = DataFrame((met, asset, back[met]) for met in metrics)
	rename!(metdf, [:metric, :asset, :bh])
end

function getalpha(n::Integer)
	n < 200 && return .05
	n < 500 && return .03
	return .01
end

"""
Train `n` EvoTree models, dump the serialized median valued model, the metrics for
that model, and strawbroom pnl plots that include all models and b&h benchmark.
"""
function evodump(config, asset, asset_f, abl; n::Integer=NRUNS, runtest=false, metric=:cagr)
	fnames, train_idx, train_feat, train_R, train_dir = getsplit(asset, asset_f, "train", abl)
	_, val_idx, val_feat, val_R, val_dir = getsplit(asset, asset_f, "val", abl)

	models = []
	for i in 1:n
		push!(models,
			fit_evotree(config;
				x_train=train_feat, y_train=train_dir, fnames=fnames,
				# x_eval=val_feat, y_eval=val_dir
			)
		)
	end

	alpha = getalpha(n)
	path = joinpath(TM_DIR, "manual", asset, asset_f, abl) |> mkpath

	train_ys = [mdl(train_feat) |> addflat for mdl in models]
	train_backs = backtest.([train_R], train_ys)
	modidx = getbacktest(train_backs, :lsf, :score, metric)
	EvoTrees.save(models[modidx], joinpath(path, "model.bson"))

	evodumpmetrics(train_idx, train_R, train_feat, models, modidx, train_ys, train_backs;
		path=path, asset=asset, asset_f=asset_f, abl=abl, split="train", metric=metric, alpha=alpha
	)
	evodumpmetrics(train_idx, train_R, train_feat, models, train_ys, train_backs;
		path=path, asset=asset, asset_f=asset_f, abl=abl, split="train", metric=metric, alpha=alpha
	)

	val_ys = [mdl(val_feat) |> addflat for mdl in models]
	val_backs = backtest.([val_R], val_ys)
	evodumpmetrics(val_idx, val_R, val_feat, models, modidx, val_ys, val_backs;
		path=path, asset=asset, asset_f=asset_f, abl=abl, split="val", metric=metric, alpha=alpha
	)
	evodumpmetrics(val_idx, val_R, val_feat, models, val_ys, val_backs;
		path=path, asset=asset, asset_f=asset_f, abl=abl, split="val", metric=metric, alpha=alpha
	)

	if runtest
		_, test_idx, test_feat, test_R, test_dir = getsplit(asset, asset_f, "test", abl)
		test_ys = [mdl(test_feat) |> addflat for mdl in models]
		test_backs = backtest.([test_R], test_ys)
		evodumpmetrics(test_idx, test_R, test_feat, models, modidx, test_ys, test_backs;
			path=path, asset=asset, asset_f=asset_f, abl=abl, split="test", metric=metric, alpha=alpha
		)
		evodumpmetrics(test_idx, test_R, test_feat, models, test_ys, test_backs;
			path=path, asset=asset, asset_f=asset_f, abl=abl, split="test", metric=metric, alpha=alpha
		)
	end
end

"""
Dump metrics and plots for each model split.
"""
function evodumpmetrics(idx, R, feat, models, modidx::Integer, ys=nothing, backs=nothing;
	path, asset, asset_f, abl, split, metric, alpha)
	isnothing(ys) && (ys = [mdl(feat) |> addflat for mdl in models])
	isnothing(backs) && (backs = backtest.([R], ys))
	back = backs[modidx]
	bench = backtest(R)
	benchfile = joinpath(TM_DIR, "manual", asset, "bench_$(split).csv")
	!isfile(benchfile) && CSV.write(benchfile, metrics(bench, asset))

	titlepfx = "$asset $abl $split" |> lowercase
	for pnltype in [:pnl, :pnlcomp], betsize in [:full, :score]
		savefig(
			plotpnls(idx, backs, back, bench, betsize; pnltype=pnltype, titlepfx=titlepfx*" $betsize", alpha=alpha),
			joinpath(path, "plot_$(split)_$(pnltype)_$(betsize).png")
		)
	end
	CSV.write(joinpath(path, "metrics_$(split).csv"), metrics(back, asset, asset_f, abl))
end

"""
Dump metrics and plots for each model split, use median specific to the split.
"""
function evodumpmetrics(idx, R, feat, models, ys=nothing, backs=nothing;
	path, asset, asset_f, abl, split, metric, alpha)
	isnothing(ys) && (ys = [mdl(feat) |> addflat for mdl in models])
	isnothing(backs) && (backs = backtest.([R], ys))
	back = backs[getbacktest(backs, :lsf, :score, metric)]
	bench = backtest(R)
	benchfile = joinpath(TM_DIR, "manual", asset, "bench_$(split).csv")
	!isfile(benchfile) && CSV.write(benchfile, metrics(bench, asset))

	titlepfx = "$asset $abl $split (med)" |> lowercase
	for pnltype in [:pnl, :pnlcomp], betsize in [:full, :score]
		savefig(
			plotpnls(idx, backs, back, bench, betsize; pnltype=pnltype, titlepfx=titlepfx*" $betsize", alpha=alpha),
			joinpath(path, "plot_$(split)_$(pnltype)_$(betsize)_med.png")
		)
	end
	CSV.write(joinpath(path, "metrics_$(split)_med.csv"), metrics(back, asset, asset_f, abl))
end
