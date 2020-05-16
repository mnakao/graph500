#include <mpi.h>
#include <math.h>
#include "parameters.h"
#include "utils_core.h"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "bfs.hpp"

typedef BfsBase BfsOnCPU;

template <typename EdgeList>
void generate_graph(EdgeList* edge_list, const GraphGenerator<typename EdgeList::edge_type>* generator)
{
  typedef typename EdgeList::edge_type EdgeType;
    EdgeType* edge_buffer = static_cast<EdgeType*>
	  (cache_aligned_xmalloc(EdgeList::CHUNK_SIZE*sizeof(EdgeType)));
    edge_list->beginWrite();
    const int64_t num_global_edges = generator->num_global_edges();
    const int64_t num_global_chunks = (num_global_edges + EdgeList::CHUNK_SIZE - 1) / EdgeList::CHUNK_SIZE;
    const int64_t num_iterations = (num_global_chunks + mpi.size_2d - 1) / mpi.size_2d;

#pragma omp parallel
    for(int64_t i = 0; i < num_iterations; ++i) {
	  const int64_t start_edge = std::min((mpi.size_2d*i + mpi.rank_2d) * EdgeList::CHUNK_SIZE, num_global_edges);
	  const int64_t end_edge = std::min(start_edge + EdgeList::CHUNK_SIZE, num_global_edges);
	  generator->generateRange(edge_buffer, start_edge, end_edge);
#pragma omp master
	  edge_list->write(edge_buffer, end_edge - start_edge);
    }
    edge_list->endWrite();
    free(edge_buffer);
}

template <typename EdgeList>
void generate_graph_spec2010(EdgeList* edge_list, int scale, int edge_factor, int max_weight = 0)
{
  RmatGraphGenerator<typename EdgeList::edge_type, 5700, 1900> generator(scale, edge_factor, 255,
																		 PRM::USERSEED1, PRM::USERSEED2, InitialEdgeType::NONE);
  generate_graph(edge_list, &generator);
}

int main(int argc, char** argv)
{
	int SCALE = 16;
	if (argc >= 2) SCALE = atoi(argv[1]);
	setup_globals(argc, argv, SCALE, 16);
    EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
																 (int64_t(1) << SCALE) * 16 / mpi.size_2d, getenv("TMPFILE"));
    generate_graph_spec2010(&edge_list, SCALE, 16);
    BfsOnCPU* benchmark = new BfsOnCPU();
    benchmark->construct(&edge_list);

    MPI_Finalize();
	return 0;
}
