#!/bin/bash
# Run expm (manual) train/val/test
# forecast, all assets, all ablations

PM="002";
FIXED="--final --smodel=anp"
YPFX="rvol_1day_r_1min_"

for A in "DJI" "NDX" "RUT" "SPX"; do
  for X in "logchangeprice" "logchangeprice,logchangeivol"; do
    for Y in $YPFX"meanad" $YPFX"std" $YPFX"var" $YPFX"mad"; do
      P="$PM-$A-$Y-$X"
      python3 -m model.expm --param=$P --assets=$A --xdata=$X --ydata=$Y $FIXED
    done;
  done;
done;
