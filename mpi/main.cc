/*
 * main.cc
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

// C includes
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// C++ includes
#include <string>

#include "parameters.h"
#include "utils_core.h"
#include "primitives.hpp"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "validate.hpp"
#include "benchmark_helper.hpp"
#include "bfs.hpp"
#include "bfs_cpu.hpp"
#if CUDA_ENABLED
#include "bfs_gpu.hpp"
#endif

/*
	template <typename T>
	void dump_data(const char* name, T* data, int length) {
		print_with_prefix("Dump %s: %d", name, length);
		FILE* file = fopen(name, "w+b");
		fwrite(data, sizeof(T), length, file);
		fclose(file);
	}

	template <typename E>
	void dump_edge_list(const char* name, E& edge_list) {
		FILE* file = fopen(name, "w+b");
		int64_t total = 0;
		int num_loops = edge_list.beginRead(false);
		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			typename E::edge_type* edges;
			const int length = edge_list.read(&edges);
			fwrite(edges, sizeof(typename E::edge_type), length, file);
			total += length;
		}
		edge_list.endRead();
		fclose(file);
		print_with_prefix("Dump %s: %d", name, total);
	}
*/

void graph500_bfs(int SCALE, int edgefactor)
{
	using namespace PRM;
	SET_AFFINITY;

	double bfs_times[NUM_BFS_ROOTS], validate_times[NUM_BFS_ROOTS], edge_counts[NUM_BFS_ROOTS];
	LogFileFormat log = {0};
	int root_start = read_log_file(&log, SCALE, edgefactor, bfs_times, validate_times, edge_counts);
	if(mpi.isMaster() && root_start != 0)
		print_with_prefix("Resume from %d th run", root_start);

	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
//	EdgeListStorage<UnweightedPackedEdge, 512*1024> edge_list(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, getenv("TMPFILE"));

	BfsOnCPU::printInformation();

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	generate_graph_spec2010(&edge_list, SCALE, edgefactor);
	generation_time = MPI_Wtime() - generation_time;

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create BFS instance and the *COMMUNICATION THREAD*.
	BfsOnCPU* benchmark = new BfsOnCPU();
	double construction_time = MPI_Wtime();
	benchmark->construct(&edge_list);
	construction_time = MPI_Wtime() - construction_time;

#if VALIDATION_LEVEL > 0
	if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
	double redistribution_time = MPI_Wtime();
	redistribute_edge_2d(&edge_list);
	redistribution_time = MPI_Wtime() - redistribution_time;
#else
	double redistribution_time = 0;
	edge_list.clear();
#endif

	//dump_edge_list("dump_main_edge_list", edge_list);

	int64_t bfs_roots[NUM_BFS_ROOTS];
	int num_bfs_roots = NUM_BFS_ROOTS;
	find_roots(benchmark->graph_, bfs_roots, num_bfs_roots);
	const int64_t max_used_vertex = find_max_used_vertex(benchmark->graph_);
	const int64_t nlocalverts = benchmark->graph_.pred_size();

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
	}
#endif

	bool result_ok = true;

	if(root_start == 0)
		init_log(SCALE, edgefactor, generation_time, construction_time, redistribution_time, &log);

	benchmark->prepare_bfs();
// narashi
		double time_left = PRE_EXEC_TIME;
        for(int c = root_start; time_left > 0.0; ++c) {
                if(mpi.isMaster())  print_with_prefix("========== Pre Running BFS %d ==========", c);
                MPI_Barrier(mpi.comm_2d);
                double bfs_time = MPI_Wtime();
                benchmark->run_bfs(bfs_roots[c % num_bfs_roots], pred);
                bfs_time = MPI_Wtime() - bfs_time;
                if(mpi.isMaster()) {
                        print_with_prefix("Time for BFS %d is %f", c, bfs_time);
                        time_left -= bfs_time;
                }
               MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
        }
/////////////////////
	for(int i = root_start; i < num_bfs_roots; ++i) {
	//for(int i = 0; i < num_bfs_roots; ++i) {
		VERVOSE(print_max_memory_usage());

		if(mpi.isMaster())  print_with_prefix("========== Running BFS %d ==========", i);
#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
		PROF(profiling::g_pis.reset());
		double cur_bfs_time = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[i], pred);
		bfs_times[i] = cur_bfs_time = MPI_Wtime() - cur_bfs_time;
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for BFS %d is %f", i, cur_bfs_time);
		}
		benchmark->get_pred(pred);

		if(mpi.isMaster()) {
			print_with_prefix("Validating BFS %d", i);
		}

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
#if VALIDATION_LEVEL >= 2
		result_ok = validate_bfs_result(
					&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
#elif VALIDATION_LEVEL == 1
		if(i == 0) {
			result_ok = validate_bfs_result(
						&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
		}
		else {
			edge_visit_count = pf_nedge[SCALE];
		}
#else
		edge_visit_count = pf_nedge[SCALE];
#endif
		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for BFS %d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %"PRId64"", edge_visit_count);
			print_with_prefix("TEPS for BFS %d is %g", i, edge_visit_count / bfs_times[i]);
		}

		if(result_ok == false) {
			break;
		}

		update_log_file(&log, bfs_times[i], validate_times[i], edge_visit_count);
	}
	benchmark->end_bfs();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
	  fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
	  fprintf(stdout, "graph_generation:               %g\n", generation_time);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  fprintf(stdout, "redistribution_time:            %g\n", redistribution_time);
	  print_bfs_result(num_bfs_roots, bfs_times, validate_times, edge_counts, result_ok);
	}

	delete benchmark;

	free(pred);
}

template <typename EdgeList>
void writeGraphToFile(int SCALE, EdgeList* edge_list) {
	using namespace PRM;
	int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);
	detail::GraphConstructor2DCSR<EdgeList> constructor;
	constructor.writeToFile(SCALE, edge_list, log_local_verts_unit);
}

template <typename GraphType>
void readGraphToFile(int SCALE, int nLocalEdges, GraphType& g) {
	using namespace PRM;
	typedef EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> EdgeList;
	int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);
	detail::GraphConstructor2DCSR<EdgeList> constructor;
	constructor.readFromFile(SCALE, nLocalEdges, log_local_verts_unit, g);
}

void gen_graph(int SCALE, int edgefactor)
{
	using namespace PRM;
	SET_AFFINITY;

	if(mpi.isMaster()) print_with_prefix("**** Special Mode: Graph Generation ****");

	const char* TMPFILE = getenv("TMPFILE");
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
//	EdgeListStorage<UnweightedPackedEdge, 512*1024> edge_list(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, TMPFILE, MPI_MODE_CREATE);

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	generate_graph_spec2010(&edge_list, SCALE, edgefactor);
	generation_time = MPI_Wtime() - generation_time;

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	double construction_time = MPI_Wtime();
	writeGraphToFile(SCALE, &edge_list);
	construction_time = MPI_Wtime() - construction_time;

	if(TMPFILE != NULL) {
		if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
		double redistribution_time = MPI_Wtime();
		redistribute_edge_2d(&edge_list);
		redistribution_time = MPI_Wtime() - redistribution_time;
	}

	//dump_edge_list("dump_gen_edge_list", edge_list);
}

void read_graph_and_bfs(int SCALE, int edgefactor)
{
	using namespace PRM;
	SET_AFFINITY;

	if(mpi.isMaster()) print_with_prefix("**** Special Mode: Read Graph ****");

	double bfs_times[NUM_BFS_ROOTS], validate_times[NUM_BFS_ROOTS], edge_counts[NUM_BFS_ROOTS];

	int64_t nLocalEdges = (int64_t(1) << SCALE) * edgefactor / mpi.size_2d;
#if VALIDATION_LEVEL > 0
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(0, getenv("TMPFILE"), 0);
#endif

	//dump_edge_list("dump_read_edge_list", edge_list);

	BfsOnCPU::printInformation();

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create BFS instance and the *COMMUNICATION THREAD*.
	BfsOnCPU* benchmark = new BfsOnCPU();
	double construction_time = MPI_Wtime();
	readGraphToFile(SCALE, nLocalEdges, benchmark->graph_);
	construction_time = MPI_Wtime() - construction_time;

	int64_t bfs_roots[NUM_BFS_ROOTS];
	int num_bfs_roots = NUM_BFS_ROOTS;
	find_roots(benchmark->graph_, bfs_roots, num_bfs_roots);
	const int64_t max_used_vertex = find_max_used_vertex(benchmark->graph_);
	const int64_t nlocalverts = benchmark->graph_.pred_size();

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
	}
#endif

	bool result_ok = true;

	benchmark->prepare_bfs();
// narashi
		double time_left = PRE_EXEC_TIME;
        for(int c = 0; time_left > 0.0; ++c) {
                if(mpi.isMaster())  print_with_prefix("========== Pre Running BFS %d ==========", c);
                MPI_Barrier(mpi.comm_2d);
                double bfs_time = MPI_Wtime();
                benchmark->run_bfs(bfs_roots[c % num_bfs_roots], pred);
                bfs_time = MPI_Wtime() - bfs_time;
                if(mpi.isMaster()) {
                        print_with_prefix("Time for BFS %d is %f", c, bfs_time);
                        time_left -= bfs_time;
                }
               MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
        }
/////////////////////
	for(int i = 0; i < num_bfs_roots; ++i) {
		VERVOSE(print_max_memory_usage());

		if(mpi.isMaster())  print_with_prefix("========== Running BFS %d ==========", i);
#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
		PROF(profiling::g_pis.reset());
		double cur_bfs_time = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[i], pred);
		bfs_times[i] = cur_bfs_time = MPI_Wtime() - cur_bfs_time;
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for BFS %d is %f", i, cur_bfs_time);
		}
		benchmark->get_pred(pred);

		if(mpi.isMaster()) {
			print_with_prefix("Validating BFS %d", i);
		}

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
#if VALIDATION_LEVEL >= 2
		result_ok = validate_bfs_result(
					&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
#elif VALIDATION_LEVEL == 1
		if(i == 0) {
			result_ok = validate_bfs_result(
						&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
		}
		else {
			edge_visit_count = pf_nedge[SCALE];
		}
#else
		edge_visit_count = pf_nedge[SCALE];
#endif
		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for BFS %d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %"PRId64"", edge_visit_count);
			print_with_prefix("TEPS for BFS %d is %g", i, edge_visit_count / bfs_times[i]);
		}

		if(result_ok == false) {
			break;
		}
	}
	benchmark->end_bfs();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
	  fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  print_bfs_result(num_bfs_roots, bfs_times, validate_times, edge_counts, result_ok);
	}

	delete benchmark;

	free(pred);
}

#if 0
void test02(int SCALE, int edgefactor)
{
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
			(INT64_C(1) << SCALE) * edgefactor / mpi.size, getenv("TMPFILE"));
	RmatGraphGenerator<UnweightedPackedEdge> graph_generator(
//	RandomGraphGenerator<UnweightedPackedEdge> graph_generator(
				SCALE, edgefactor, 2, 3, InitialEdgeType::NONE);
	Graph2DCSR<Pack40bit, uint32_t> graph;

	double generation_time = MPI_Wtime();
	generate_graph(&edge_list, &graph_generator);
	generation_time = MPI_Wtime() - generation_time;

	double construction_time = MPI_Wtime();
	construct_graph(&edge_list, true, graph);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) {
		print_with_prefix("TEST02");
		fprintf(stdout, "SCALE:                          %d\n", SCALE);
		fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
		fprintf(stdout, "graph_generation:               %g\n", generation_time);
		fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size);
		fprintf(stdout, "construction_time:              %g\n", construction_time);
	}
}
#endif

int main(int argc, char** argv)
{
	// Parse arguments.
	int mode = 0;
	int SCALE = 16;
	int edgefactor = 16; // nedges / nvertices, i.e., 2*avg. degree
	if (argc >= 2) SCALE = atoi(argv[1]);
	if (argc >= 3) edgefactor = atoi(argv[2]);

	if(SCALE == 0 && argc >= 3) {
		std::string arg1(argv[1]);
		if(arg1 == "gen") {
			mode = 1;
		}
		else if(arg1 == "read") {
			mode = 2;
		}
		if (argc >= 3) SCALE = atoi(argv[2]);
		if (argc >= 4) edgefactor = atoi(argv[3]);
	}

	if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
		fprintf(IMD_OUT, "Usage: %s SCALE edgefactor\n"
				"SCALE = log_2(# vertices) [integer, required]\n"
				"edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n"
				"(Random number seed are in main.c)\n",
				argv[0]);
		return 0;
	}

	setup_globals(argc, argv, SCALE, edgefactor);

	if(mode == 1) {
		gen_graph(SCALE, edgefactor);
	}
	else if(mode == 2) {
		read_graph_and_bfs(SCALE, edgefactor);
	}
	else {
		graph500_bfs(SCALE, edgefactor);
	}

	cleanup_globals();
	return 0;
}


