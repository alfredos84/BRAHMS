# ── BRAHMS CPU Engine Makefile ─────────────────────────────────────────────────
#
# Usage:
#   make                            # default grid
#   make NX=64 NY=64 NZ=100 NT=256
#   make NX=8 NY=8 NZ=20 NT=8      # quick test
#   make clean
#
# Dependencies: g++ ≥ 9, OpenMP, FFTW3 single-precision, HDF5, nlohmann/json
# ──────────────────────────────────────────────────────────────────────────────

NX  ?= 128
NY  ?= 128
NZ  ?= 100
NT  ?= 512

# ── Paths ─────────────────────────────────────────────────────────────────────
# FFTW3 — try common conda install locations (any user), then pkg-config,
# then the standard system paths (Debian/Ubuntu multiarch layout).
CONDA_FFTW := $(shell find $(HOME)/anaconda3/pkgs $(HOME)/miniconda3/pkgs \
                /opt/conda/pkgs -maxdepth 1 -type d -name "fftw-*" \
                2>/dev/null | sort -r | head -1)
PKGCONFIG_FFTW_LIB := $(shell pkg-config --variable=libdir fftw3f 2>/dev/null)
MULTIARCH := $(shell dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null)

ifneq ($(CONDA_FFTW),)
  FFTW3_INC := $(CONDA_FFTW)/include
  FFTW3_LIB := $(CONDA_FFTW)/lib
else ifneq ($(PKGCONFIG_FFTW_LIB),)
  FFTW3_INC := $(shell pkg-config --variable=includedir fftw3f)
  FFTW3_LIB := $(PKGCONFIG_FFTW_LIB)
else ifneq ($(MULTIARCH),)
  FFTW3_INC := /usr/include
  FFTW3_LIB := /usr/lib/$(MULTIARCH)
else
  FFTW3_INC := /usr/include
  FFTW3_LIB := /usr/lib
endif

HDF5_INC  ?= /usr/include/hdf5/serial
HDF5_LIB  ?= /usr/lib/x86_64-linux-gnu/hdf5/serial
JSON_INC  ?= /usr/include

# ── Compiler ──────────────────────────────────────────────────────────────────
CXX = g++

GRID_FLAGS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DNT=$(NT)

CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp \
           $(GRID_FLAGS) \
           -I$(FFTW3_INC) \
           -I$(HDF5_INC) \
           -I$(JSON_INC)

LDFLAGS = -fopenmp \
          -L$(FFTW3_LIB) -lfftw3f_omp -lfftw3f \
          -Wl,-rpath,$(FFTW3_LIB) \
          -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 \
          -lm

TARGET = twm_cpu
SRCS   = twm_cpu.cpp

.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(SRCS) headers/*.hpp
	@echo "  [CXX] Compiling BRAHMS CPU engine..."
	@echo "        Grid: NX=$(NX)  NY=$(NY)  NZ=$(NZ)  NT=$(NT)"
	@echo "        FFTW3: $(FFTW3_LIB)"
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "  [OK]  ./$(TARGET) ready"

clean:
	rm -f $(TARGET) *.o

info:
	@echo "NX=$(NX) NY=$(NY) NZ=$(NZ) NT=$(NT)"
	@echo "FFTW3_INC=$(FFTW3_INC)"
	@$(CXX) --version | head -1
