#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <string>
#include "parameters.h"
#include "utils_core.h"
#include "primitives.hpp"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "benchmark_helper.hpp"
#include "bfs.hpp"

typedef BfsBase BfsOnCPU;

int main(int argc, char** argv)
{
	int SCALE = 16;
	if (argc >= 2) SCALE = atoi(argv[1]);
	setup_globals(argc, argv, SCALE, 16);
    SET_AFFINITY;
    EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
																 (int64_t(1) << SCALE) * 16 / mpi.size_2d, getenv("TMPFILE"));
    generate_graph_spec2010(&edge_list, SCALE, 16);
    BfsOnCPU* benchmark = new BfsOnCPU();
    benchmark->construct(&edge_list);

    MPI_Finalize();
    exit(0);
}
