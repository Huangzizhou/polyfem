#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>

namespace polyfem
{
	class OBJWriter
	{
	public:
		static bool save(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &e, const Eigen::MatrixXi &f);
		static bool save(const std::string &path, const Eigen::MatrixXd &v, const Eigen::MatrixXi &f) { return save(path, v, Eigen::MatrixXi(), f); }
	};

} // namespace polyfem
