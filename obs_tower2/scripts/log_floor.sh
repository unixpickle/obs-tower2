#!/bin/bash

cat $1 | grep floor= | cut -f 3 -d = | cut -f 1 -d ' ' | tail -n 2000 | agg mean
