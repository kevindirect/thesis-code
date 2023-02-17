#!/usr/bin/env julia

using CSV
using EvoTrees
using Glob
using Logging
using TOML
include("TradingModel.jl")

Logging.disable_logging(Logging.Info)

runtest = "final" in ARGS
defconfig = EvoTreeClassifier()

function globcatwrite(fname, path)
	files = CSV.File.(glob("*/*/$fname", path))
	CSV.write("$path/$fname", reduce(vcat, files))
end

for asset in TradingModel.ASSETS
	for abl in TradingModel.ABLATIONS
		@info "$asset $abl"
		TradingModel.evodump(defconfig, asset, "all", abl; runtest=runtest)
	end

	fdir = joinpath(TradingModel.TM_DIR, "manual", asset)
	globcatwrite("metrics_train.csv", fdir)
	globcatwrite("metrics_train_med.csv", fdir)
	globcatwrite("metrics_val.csv", fdir)
	globcatwrite("metrics_val_med.csv", fdir)
	if runtest
		globcatwrite("metrics_test.csv", fdir)
		globcatwrite("metrics_test_med.csv", fdir)
	end
end

