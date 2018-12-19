#!/bin/bash

CURDIR=$(dirname $0)

export PYTHONPATH=$CURDIR/..:$PYTHONPATH
./plotIOCs.py
