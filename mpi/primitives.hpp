#ifndef PRIMITIVES_HPP_
#define PRIMITIVES_HPP_

struct UnweightedEdge;
template <> struct MpiTypeOf<UnweightedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedEdge {
	int64_t v0_;
	int64_t v1_;
	typedef int no_weight;
	int64_t v0() const { return v0_; }
	int64_t v1() const { return v1_; }
	void set(int64_t v0, int64_t v1) { v0_ = v0; v1_ = v1; }

	static void initialize()
	{
		int block_length[] = {1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v0_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedEdge*>(NULL)->v1_)) };
		MPI_Type_create_hindexed(2, block_length, displs, MPI_INT64_T, &MpiTypeOf<UnweightedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedEdge>::type);
	}
};

struct UnweightedPackedEdge;
template <> struct MpiTypeOf<UnweightedPackedEdge> { static MPI_Datatype type; };
MPI_Datatype MpiTypeOf<UnweightedPackedEdge>::type = MPI_DATATYPE_NULL;

struct UnweightedPackedEdge {
	uint32_t v0_low_;
	uint32_t v1_low_;
	uint32_t high_;
	typedef int no_weight;
	int64_t v0() const { return (v0_low_ | (static_cast<int64_t>(high_ & 0xFFFF) << 32)); }
	int64_t v1() const { return (v1_low_ | (static_cast<int64_t>(high_ >> 16) << 32)); }
	void set(int64_t v0, int64_t v1) {
		v0_low_ = static_cast<uint32_t>(v0);
		v1_low_ = static_cast<uint32_t>(v1);
		high_ = ((v0 >> 32) & 0xFFFF) | ((v1 >> 16) & 0xFFFF0000U);
	}

	static void initialize()
	{
		int block_length[] = {1, 1, 1};
		MPI_Aint displs[] = {
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v0_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->v1_low_)),
				reinterpret_cast<MPI_Aint>(&(static_cast<UnweightedPackedEdge*>(NULL)->high_)) };
		MPI_Type_create_hindexed(3, block_length, displs, MPI_UINT32_T, &MpiTypeOf<UnweightedPackedEdge>::type);
		MPI_Type_commit(&MpiTypeOf<UnweightedPackedEdge>::type);
	}
};

#endif /* PRIMITIVES_HPP_ */
