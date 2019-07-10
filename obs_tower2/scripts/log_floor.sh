#!/bin/bash
#
# Get the average floor from a log file over the last 2K
# episodes.
#
# Requires the agg command: https://github.com/unixpickle/agg.
#

cat $1 | grep floor= | cut -f 3 -d = | cut -f 1 -d ' ' | tail -n 2000 | agg mean
