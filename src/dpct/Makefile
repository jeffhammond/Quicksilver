SHELL = /bin/bash

CXX       = dpcpp
CXXFLAGS  = -std=c++14
CXXFLAGS += -O3
CXXFLAGS += -ferror-limit=1
CPPFLAGS += -DHAVE_SYCL -DHAVE_CUDA
LDFLAGS   =

EXT = .cc

Quicksilver_EXE=qs

# clear all suffixes
.SUFFIXES:
# list only those that we use
.SUFFIXES: ${EXT} .o

.PHONY: DEFAULT clean distclean depend

SOURCES= \
    CollisionEvent${EXT} \
    CoralBenchmark${EXT} \
    CycleTracking${EXT} \
    DecompositionObject${EXT} \
    DirectionCosine${EXT} \
    EnergySpectrum${EXT} \
    GlobalFccGrid${EXT} \
    GridAssignmentObject${EXT} \
    InputBlock${EXT} \
    MCT${EXT} \
    MC_Adjacent_Facet${EXT} \
    MC_Base_Particle${EXT} \
    MC_Domain${EXT} \
    MC_Facet_Crossing_Event${EXT} \
    MC_Fast_Timer${EXT} \
    MC_Load_Particle${EXT} \
    MC_Location${EXT} \
    MC_Particle_Buffer${EXT} \
    MC_RNG_State${EXT} \
    MC_Segment_Outcome${EXT} \
    MC_SourceNow${EXT} \
    MacroscopicCrossSection${EXT} \
    MeshPartition${EXT} \
    MonteCarlo${EXT} \
    MpiCommObject${EXT} \
    NuclearData${EXT} \
    Parameters${EXT} \
    ParticleVault${EXT} \
    ParticleVaultContainer${EXT} \
    PopulationControl${EXT} \
    SendQueue${EXT} \
    SharedMemoryCommObject${EXT} \
    Tallies${EXT} \
    cmdLineParser${EXT} \
    cudaFunctions${EXT} \
    initMC${EXT} \
    main${EXT} \
    parseUtils${EXT} \
    utils${EXT} \
    utilsMpi${EXT}

CC_OBJECTS=$(SOURCES:${EXT}=.o)

DEFAULT: ${Quicksilver_EXE}

%.o: %${EXT}
	${CXX} ${CPPFLAGS} ${CXXFLAGS} -c $< -o $@

${Quicksilver_EXE}: ${CC_OBJECTS}
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o ${Quicksilver_EXE} ${CC_OBJECTS}

clean:
	rm -f *.o

distclean: clean
	rm -f ${Quicksilver_EXE}
