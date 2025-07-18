# Biosig Package

Biosig is a toolbox for processing biomedical signals
like ECG, EEG etc. It can read about 50 different
data formats including EDF, GDF, BDF, CFS, Heka, ABF,
HL7aECG, SCP (EN1064), BrainVision, and many more.
The heavy lifting is done in "libbbiosig"
the biosig library implemented in C/C++. The
Python module of Biosig provides an interface between
libbiosig and Python. Currently, two functions are
provided, one for reading the data samples, and one
for extracting the meta information, like event
table, scaling factors, sampling rates, etc.


More information is available at
- [Biosig project homepage](https://biosig.sourceforge.io)
- [Biosig wiki page](https://sourceforge.net/p/biosig/wiki/Home/)

# Installation:
## GNU/Linux (Debian, Ubuntu, and family)
```
  sudo apt install python3-biosig
```
or if you want to install in your own venv
```
  sudo apt install libbiosig-dev  # make sure that libbiosig and its prerequisites are in place
  pip install biosig
```

## MacOSX/Homebrew
```
  brew install biosig             # make sure libbiosig is in place.
  pip install biosig
```

## MS-Windows (Python >=3.9,<=3.12)
```
  pip install biosig
```

## Other *NIX (Linux, Unix) OS

- install libbiosig from source (`configure && make && make install`)
- `pip install biosig`

# Usage
```
  import biosig
  import json
```
## read header/metainformation
```
  HDR=json.loads(biosig.header(FILENAME))
```
## read data
```
  DATA=biosig.data(FILENAME)
```

## A more elaborate example is this [demo](https://sourceforge.net/p/biosig/code/ci/master/tree/biosig4c++/python/demo2.py "demo2.py")

showing
- how to load data
- displaying the traces with matplotlib
- splitting the traces into sweeps



# History
In the past there were several attempts
of providing Python support. A first attempt
using pure python should to be very slow and a
lot of implementation effort, only some early
version of GDF has been supported.
Later, Swig was used for providing a python binding.
This was reasonable flexible, and the libbiosig
implementation could be mostly used. Disadvantages
were the effort to maintain swig.i interface file,
which is currently most likely broken. Known issues
are string support, lack of Python3 support. Also this
attempt is considered deprecated.

The third attempt is using "module extensions".
Currently, two functions, one for reading the header
information (in JSON format) and one for reading the
data samples is provided.
