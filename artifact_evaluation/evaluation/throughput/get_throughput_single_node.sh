#!/bin/bash

# This script: 1) runs a single BERT multistream experiment with cfreq=1;
#             2) then runs the collector/plotter step to produce figures for
#                cfreqs [0,1,10,25,50,100].

# 1) run only the BERT multistream method for cfreq=1 (will write bert/log_bert_multistream_1.txt)
python3.9 run_throughput_model.py transformer --methods multistream --cfreqs "1"

# 2) collect and plot using cfreqs 0,1,10,25,50,100 from existing logs (skip running experiments)
python3.9 run_throughput_model.py transformer --skip-run --cfreqs "0,1,10,25,50,100"

# Notes:
# - First command runs the experiment and will produce the single log for cfreq=1.
# - Second command reads logs for the requested cfreqs and produces fig8_bert.csv and fig8_bert.png.