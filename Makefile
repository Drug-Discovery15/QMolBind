# EXP=smoke|full selects the experiment config for baselines/train-q/experiments (baselines, train-q default: full)
PY ?= python
T = $(PY) scripts/tasks.py

.PHONY: env data test smoke baselines train-q experiments report app
env:         ; $(T) env
data:        ; $(T) data
test:        ; $(T) test
smoke:       ; $(T) smoke
baselines:   ; $(T) baselines
train-q:     ; $(T) train-q
experiments: ; $(T) experiments
report:      ; $(T) report
app:         ; $(T) app
