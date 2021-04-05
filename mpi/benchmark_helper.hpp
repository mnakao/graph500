/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef BENCHMARK_HELPER_HPP_
#define BENCHMARK_HELPER_HPP_

#include "logfile.h"

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
	double logging_time = MPI_Wtime();
	if(mpi.isMaster()) {
		double global_data_size = (double)num_global_edges * 16.0 / 1000000000.0;
		double local_data_size = global_data_size / mpi.size_2d;
		print_with_prefix("Graph data size: %f GB ( %f GB per process )", global_data_size, local_data_size);
		print_with_prefix("Using storage: %s", edge_list->data_is_in_file() ? "yes" : "no");
		if(edge_list->data_is_in_file()) {
			print_with_prefix("Filepath: %s 1 2 ...", edge_list->get_filepath());
		}
		print_with_prefix("Communication chunk size: %d", EdgeList::CHUNK_SIZE);
		print_with_prefix("Generating graph: Total number of iterations: %" PRId64 "", num_iterations);
	}

#pragma omp parallel
	for(int64_t i = 0; i < num_iterations; ++i) {
		SET_OMP_AFFINITY;
		const int64_t start_edge = std::min((mpi.size_2d*i + mpi.rank_2d) * EdgeList::CHUNK_SIZE, num_global_edges);
		const int64_t end_edge = std::min(start_edge + EdgeList::CHUNK_SIZE, num_global_edges);
		generator->generateRange(edge_buffer, start_edge, end_edge);
#if defined(__INTEL_COMPILER)
#pragma omp barrier
#endif
		// we need to synchronize before this code.
		// There is the implicit barrier on the end of for loops.
#pragma omp master
		{
			edge_list->write(edge_buffer, end_edge - start_edge);

			if(mpi.isMaster()) {
				print_with_prefix("Time for iteration %" PRId64 " is %f ", i, MPI_Wtime() - logging_time);
				logging_time = MPI_Wtime();
			}
		}
#pragma omp barrier

	}

	edge_list->endWrite();
	free(edge_buffer);
	if(mpi.isMaster()) print_with_prefix("Finished generating.");
}

template <typename EdgeList>
void generate_graph_spec2010(EdgeList* edge_list, int scale, int edge_factor, int max_weight = 0)
{
	RmatGraphGenerator<typename EdgeList::edge_type, 5700, 1900> generator(scale, edge_factor, 255,
			PRM::USERSEED1, PRM::USERSEED2, InitialEdgeType::NONE);
	generate_graph(edge_list, &generator);
}

template <typename EdgeList>
void generate_graph_spec2012(EdgeList* edge_list, int scale, int edge_factor, int max_weight)
{
	RmatGraphGenerator<typename EdgeList::edge_type, 5500, 100> generator(scale, edge_factor, max_weight,
			PRM::USERSEED1, PRM::USERSEED2, InitialEdgeType::BINARY_TREE);
	generate_graph(edge_list, &generator);
}

// using SFINAE
// function #1
template <typename EdgeList>
void redistribute_edge_2d(EdgeList* edge_list, typename EdgeList::edge_type::has_weight dummy = 0)
{
	typedef typename EdgeList::edge_type EdgeType;
	ScatterContext scatter(mpi.comm_2d);
	EdgeType* edges_to_send = static_cast<EdgeType*>(
			xMPI_Alloc_mem(EdgeList::CHUNK_SIZE * sizeof(EdgeType)));
	int num_loops = edge_list->beginRead(true);
	edge_list->beginWrite();

	if(mpi.isMaster()) print_with_prefix("%d iterations.", num_loops);

	for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
		EdgeType* edge_data;
		const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				(counts[edge_owner(v0,v1)])++;
			} // #pragma omp for schedule(static)

#pragma omp master
			{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
			;
			int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				const int weight = edge_data[i].weight_;
				//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
				edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0,v1,weight);
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		if(mpi.isMaster()) print_with_prefix("Scatter edges.");

		EdgeType* recv_edges = scatter.scatter(edges_to_send);
		const int64_t num_recv_edges = scatter.get_recv_count();
		edge_list->write(recv_edges, num_recv_edges);
		scatter.free(recv_edges);

		if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
	}
	if(mpi.isMaster()) print_with_prefix("Finished.");
	edge_list->endWrite();
	edge_list->endRead();
	MPI_Free_mem(edges_to_send);
}

// function #2
template <typename EdgeList>
void redistribute_edge_2d(EdgeList* edge_list, typename EdgeList::edge_type::no_weight dummy = 0)
{
	typedef typename EdgeList::edge_type EdgeType;
	ScatterContext scatter(mpi.comm_2d);
	EdgeType* edges_to_send = static_cast<EdgeType*>(
			xMPI_Alloc_mem(EdgeList::CHUNK_SIZE * sizeof(EdgeType)));
	int num_loops = edge_list->beginRead(true);
	edge_list->beginWrite();

	for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
		EdgeType* edge_data;
		const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
		{
			int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				(counts[edge_owner(v0,v1)])++;
			} // #pragma omp for schedule(static)

#pragma omp master
			{ scatter.sum(); } // #pragma omp master
#pragma omp barrier
			;
			int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
			for(int i = 0; i < edge_data_length; ++i) {
				const int64_t v0 = edge_data[i].v0();
				const int64_t v1 = edge_data[i].v1();
				//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
				edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0,v1);
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		EdgeType* recv_edges = scatter.scatter(edges_to_send);
		const int64_t num_recv_edges = scatter.get_recv_count();
#ifndef NDEBUG
		for(int64_t i = 0; i < num_recv_edges; ++i) {
			const int64_t v0 = recv_edges[i].v0();
			const int64_t v1 = recv_edges[i].v1();
			assert (vertex_owner_r(v0) == mpi.rank_2dr);
			assert (vertex_owner_c(v1) == mpi.rank_2dc);
		}
#undef VERTEX_OWNER_R
#undef VERTEX_OWNER_C
#endif
		edge_list->write(recv_edges, num_recv_edges);
		scatter.free(recv_edges);

		if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
	}
	edge_list->endWrite();
	edge_list->endRead();
	MPI_Free_mem(edges_to_send);
}

template <typename GraphType>
void decode_edge(GraphType& g, int64_t e0, int64_t e1, int64_t& v0, int64_t& v1, int& weight)
{
	const int log_size_r = get_msb_index(mpi.size_2dr);
	const int log_size = get_msb_index(mpi.size_2d);
	const int mask_packing_edge_lists = ((1 << g.log_packing_edge_lists()) - 1);
	const int log_weight_bits = g.log_packing_edge_lists_;

	const int packing_edge_lists = g.log_packing_edge_lists();
	const int log_local_verts = g.log_local_verts();
	const int64_t v0_high_mask = ((INT64_C(1) << (log_local_verts - packing_edge_lists)) - 1);

	const int rank_c = mpi.rank_2dc;
	const int rank_r = mpi.rank_2dr;

	int v0_r = e0 >> (log_local_verts - packing_edge_lists);
	int64_t v0_high = e0 & v0_high_mask;
	int64_t v0_middle = e1 & mask_packing_edge_lists;
	v0 = (((v0_high << packing_edge_lists) | v0_middle) << log_size) | ((rank_c << log_size_r) | v0_r);

	int64_t v1_and_weight = e1 >> packing_edge_lists;
	weight = v1_and_weight & ((1 << log_weight_bits) - 1);
	int64_t v1_high = v1_and_weight >> log_weight_bits;
	v1 = (v1_high << log_size_r) | rank_r;
}

template <typename GraphType>
void find_roots(GraphType& g, int64_t* bfs_roots, int& num_bfs_roots)
{
	using namespace PRM;
	/* Find roots and max used vertex */
	int64_t counter = 0;
	const int64_t nglobalverts = int64_t(1) << g.log_orig_global_verts_;
	int bfs_root_idx;
	for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
		int64_t root;
		while (1) {
			double d[2];
			make_random_numbers(2, USERSEED1, USERSEED2, counter, d);
			root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
			counter += 2;
			if (counter > 2 * nglobalverts) break;
			int is_duplicate = 0;
			int i;
			for (i = 0; i < bfs_root_idx; ++i) {
				if (root == bfs_roots[i]) {
					is_duplicate = 1;
					break;
				}
			}
			if (is_duplicate) continue; /* Everyone takes the same path here */
			int root_ok = (int)g.has_edge(root);
			int send_root_ok = root_ok;
			MPI_Allreduce(&send_root_ok, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
			if (root_ok) break;
		}
		bfs_roots[bfs_root_idx] = root;
	}
	num_bfs_roots = bfs_root_idx;
}

template <typename GraphType>
int64_t find_max_used_vertex(GraphType& g)
{
	int64_t max_used_vertex = 0;
	const int64_t nlocal = g.pred_size();
	for (int64_t i = nlocal; (i > 0) && (max_used_vertex == 0); --i) {
		int64_t local = i - 1;
		for(int64_t j = mpi.size_2dr; (j > 0) && (max_used_vertex == 0); --j) {
			int64_t r = j - 1;
			int64_t v0 = local * mpi.size_2d + mpi.rank_2dc * mpi.size_2dr + r;
			if (g.has_edge(v0)) {
				max_used_vertex = v0;
			}
		}
	}
	int64_t send_max_used_vertex = max_used_vertex;
	MPI_Allreduce(&send_max_used_vertex, &max_used_vertex, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
	return max_used_vertex;
}
#endif /* BENCHMARK_HELPER_HPP_ */
