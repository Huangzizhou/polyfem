#pragma once

#include <polyfem/mesh/Mesh.hpp>
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/basis/ElementBases.hpp>
#include <polyfem/utils/Types.hpp>

#include <Eigen/Core>

namespace polyfem::mesh
{
	enum class CollisionProxyTessellation
	{
		REGULAR,  ///< @brief Regular tessellation of the mesh
		IRREGULAR ///< @brief Irregular tessellation of the mesh (requires POLYFEM_WITH_TRIANGLE)
	};

	void build_collision_proxy(
		const mesh::Mesh &mesh,
		const int n_bases,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const std::vector<mesh::LocalBoundary> &total_local_boundary,
		const double max_edge_length,
		Eigen::MatrixXd &proxy_vertices,
		Eigen::MatrixXi &proxy_faces,
		std::vector<Eigen::Triplet<double>> &displacement_map,
		const CollisionProxyTessellation tessellation = CollisionProxyTessellation::REGULAR);
} // namespace polyfem::mesh