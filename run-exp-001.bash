#!/bin/bash
# Run expo (optuna) train/val
# forecast, all assets, np

MS="np"
PM="001";
FIXED="--param=$PM --smodel=anp --models=$MS"
YPFX="rvol_1day_r_1min_"

for X in "logchangeprice" "logchangeprice,logchangeivol"; do
  for Y in $YPFX"meanad" $YPFX"std" $YPFX"var" $YPFX"mad"; do
    python3 -m model.expo --xdata=$X --ydata=$Y $FIXED;
  done;
done;
