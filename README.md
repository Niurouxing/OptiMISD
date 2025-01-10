# How to build
Requiring C++23 features, GCC > 13.0 or Clang > 19.0 is recommended for building the project.
And check your cmake version for supporting C++23.

```bash
$ git clone https://github.com/Niurouxing/OptiMISD.git
$ cd OptiMISD
$ mkdir build
$ cd build
$ cmake ..
$ make
```

# How to run

The antenna and QAM modulation parameters are set in the `main.cpp` file. 

`main.cpp` is used for latency evaluation.

`main.py` provides the parallel execution for BER evaluation.

Before run the `main.py`, build the cmake project first.

