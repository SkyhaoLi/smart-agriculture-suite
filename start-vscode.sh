#!/bin/bash
# Start VS Code without conda environment for PlatformIO compatibility
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_PROM_MODIFIER
unset CONDA_SHLVL
code "/home/litianhao/桌面/智润/smart-agriculture-suite"
