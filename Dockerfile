FROM continuumio/anaconda:latest

RUN conda install -y theano tabulate numpy scipy mkl keras=1.1.1 matplotlib zope.interface=4.4.2 gfortran_linux-64
RUN source activate
RUN unset LDFLAGS
pip install https://github.com/WISDEM/AirfoilPreppy/tarball/master#egg=airfoilprep.py
pip install git+https://github.com/emerrf/CCBlade.git@packaging-refactored#egg=ccblade
pip install git+https://github.com/emerrf/gym-wind-turbine.git
