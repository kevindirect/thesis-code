module TradingModel

import Base.sign
using Arrow
using CSV
using Dates
using DataFrames
using EvoTrees
using Glob
using Plots
using StatsBase

const ROOT_DIR = dirname(dirname(dirname(@__DIR__)))
const MODEL_DIR = joinpath(ROOT_DIR, "model")
const DATA_DIR = joinpath(ROOT_DIR, "data")
const PROC_NAME = "002"
const VENDOR_NAME = "frd"
const EXP_DIR = joinpath(MODEL_DIR, "exp-$PROC_NAME-$VENDOR_NAME")
const TM_DIR = joinpath(MODEL_DIR, "tm-$PROC_NAME-$VENDOR_NAME")

const PERIODS = 252
const NRUNS = 301
const ASSETS = ("SPX", "RUT", "NDX", "DJI")
const ABLATIONS = ("base", "cnp", "lnp", "np")
const METRICS = (:finpnlcomp, :finpnl, :minpnl, :accuracy, :cagr, :sharpe)

include("util.jl")

end
