#!/bin/bash
ncu="/usr/local/cuda/bin/ncu"
python="$(which python)"
sudo $ncu --set full -o profile_gemm -f $python ./call.py
