# Kevin Patel

echo 1000 > /proc/self/oom_score_adj;
nice -n 12 "$@";

