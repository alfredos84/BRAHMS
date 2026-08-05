# ── TWM Engine Makefile ────────────────────────────────────────────────────────
#
# Usage:
#   make                          # default grid
#   make NX=256 NY=256 NZ=100 NT=1024
#   make clean
#
# Grid dimensions are compile-time constants (-D flags).
# The GUI triggers a rebuild whenever dimensions change.
#
# Dependencies:
#   CUDA ≥ 11.0, HDF5, nlohmann/json (header-only, in ./third_party/json/)
# ──────────────────────────────────────────────────────────────────────────────

# ── Grid dimensions (overrideable from command line) ──────────────────────────
NX   ?= 128
NY   ?= 128
NZ   ?= 100
NT   ?= 512
BLKX ?= 16
BLKY ?= 16
BLKT ?= 16

# ── Compute capability (override for your GPU, e.g. sm_86 for RTX 3090) ──────
ARCH ?= sm_75

# ── Platform detection ────────────────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    WINDOWS := 1
endif

# ── Paths ─────────────────────────────────────────────────────────────────────
ifdef WINDOWS
CUDA_HOME  ?= $(subst \,/,$(CUDA_PATH))
# vcpkg build (MSVC runtime) -- NOT the HDF Group installer build, which
# links against Intel's libmmd.dll (not present on a plain Windows box).
# Installed in-repo by install/setup_windows.ps1 so this path works on any
# machine without edits.
HDF5_HOME  ?= vcpkg_installed/x64-windows
HDF5_INC   ?= $(HDF5_HOME)/include
HDF5_LIB   ?= $(HDF5_HOME)/lib
JSON_INC   ?= third_party
else
CUDA_HOME  ?= /usr/local/cuda
HDF5_INC   ?= /usr/include/hdf5/serial
HDF5_LIB   ?= /usr/lib/x86_64-linux-gnu/hdf5/serial
JSON_INC   ?= /usr/include             # nlohmann/json system install
endif

# ── Compiler ──────────────────────────────────────────────────────────────────
ifdef WINDOWS
NVCC = "$(CUDA_HOME)/bin/nvcc"
else
NVCC = $(CUDA_HOME)/bin/nvcc
endif

# ── Flags ─────────────────────────────────────────────────────────────────────
GRID_FLAGS = -DNX=$(NX) -DNY=$(NY) -DNZ=$(NZ) -DNT=$(NT) \
             -DBLKX=$(BLKX) -DBLKY=$(BLKY) -DBLKT=$(BLKT)

ifdef WINDOWS
NVCC_FLAGS = -std=c++17 -O3 --use_fast_math \
             -Xcompiler "/Zc:preprocessor" \
             -D_USE_MATH_DEFINES -DH5_BUILT_AS_DYNAMIC_LIB \
             -arch=$(ARCH) \
             $(GRID_FLAGS) \
             -I"$(CUDA_HOME)/include" \
             -I"$(HDF5_INC)" \
             -I"$(JSON_INC)"

LDFLAGS = -L"$(CUDA_HOME)/lib/x64" -lcufft -lcurand \
           -L"$(HDF5_LIB)" -lhdf5_cpp -lhdf5

TARGET = twm.exe
else
NVCC_FLAGS = -std=c++17 -O3 --use_fast_math \
             -arch=$(ARCH) \
             $(GRID_FLAGS) \
             -I$(CUDA_HOME)/include \
             -I$(HDF5_INC) \
             -I$(JSON_INC)

LDFLAGS = -L$(CUDA_HOME)/lib64 -lcufft -lcurand \
           -L$(HDF5_LIB) -lhdf5_cpp -lhdf5 \
           -lstdc++ -lm

TARGET = twm
endif

# ── Target ────────────────────────────────────────────────────────────────────
SRCS = twm.cu

.PHONY: all clean info

all: $(TARGET)

$(TARGET): $(SRCS) headers/*.cuh
	@echo "  [NVCC] Compiling TWM engine..."
	@echo "         Grid: NX=$(NX)  NY=$(NY)  NZ=$(NZ)  NT=$(NT)"
	@echo "         Arch: $(ARCH)"
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)
	@echo "  [OK]   ./$(TARGET) ready"

clean:
ifdef WINDOWS
	-del /Q $(TARGET) *.obj *.exp *.lib 2>NUL
else
	rm -f $(TARGET) *.o
endif

info:
	@echo "NX=$(NX) NY=$(NY) NZ=$(NZ) NT=$(NT) ARCH=$(ARCH)"
	@$(NVCC) --version

# ── Convenience targets ───────────────────────────────────────────────────────
# Small CW test run (plane-wave SHG, 1D effectively)
test_shg:
	make NX=8 NY=8 NZ=200 NT=1 ARCH=$(ARCH)

# Full 3D pulsed run
build_3d:
	make NX=128 NY=128 NZ=200 NT=512 ARCH=$(ARCH)
