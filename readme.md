# Hello!

This is the Yellow EO Toolbox. The following script and setup will be created as a docker image.

# Ubuntu setup
  * sudo apt-get update
  * sudo apt-get upgrade
  * sudo apt full-upgrade
  * sudo apt autoremove

# Packages
  ## Anaconda
  * Get the newest link @ https://www.anaconda.com/distribution/ 
  * cd /tmp
  * curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
  * sudo bash Anaconda3-2019.10-Linux-x86_64.sh
  * source ~/.bashrc
  * cd ~
  * sudo chown user -R ./*
  * sudo chown user .conda/environments.txt
  * conda update conda
  * conda update --all
  * conda create --name yellow --clone root
  * conda activate yellow
  * conda install -c conda-forge gdal
  * conda update --all

  ## Git
  * sudo apt-get install git
  * cd ~
  * git clone https://github.com/casperfibaek/yellow.git

  ## Orfeo-toolbox Run
  * cd ~
  * curl -O https://www.orfeo-toolbox.org/packages/OTB-7.0.0-Linux64.run

