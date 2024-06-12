

build:
	hipcc -std=c++17 arrowhead.cpp kernels.cpp

build-g:
	hipcc -std=c++17 -DDIAGNOSTIC arrowhead.cpp kernels.cpp