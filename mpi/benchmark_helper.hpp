#ifndef BENCHMARK_HELPER_HPP_
#define BENCHMARK_HELPER_HPP_

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
		SET_OMP_AFFINITY;
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

#endif /* BENCHMARK_HELPER_HPP_ */
