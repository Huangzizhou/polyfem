#include "OptimizationProblem.hpp"
#include "Optimizations.hpp"
#include "TopologyOptimizationProblem.hpp"
#include "LBFGSSolver.hpp"
#include "LBFGSBSolver.hpp"
#include "BFGSSolver.hpp"
#include "MMASolver.hpp"
#include "GradientDescentSolver.hpp"
#include <polyfem/utils/SplineParam.hpp>

#include <map>

namespace polyfem
{
	double cross2(double x, double y)
	{
		x = abs(x);
		y = abs(y);
		if (x > y)
			std::swap(x, y);

		if (x < 0.1)
			return 0.05;
		return 0.95;
	}

	double cross3(double x, double y, double z)
	{
		x = abs(x);
		y = abs(y);
		z = abs(z);
		if (x > y)
			std::swap(x, y);
		if (y > z)
			std::swap(y, z);
		if (x > y)
			std::swap(x, y);

		if (y < 0.2)
			return 0.001;
		return 1;
	}

	template <typename ProblemType>
	std::shared_ptr<cppoptlib::NonlinearSolver<ProblemType>> make_nl_solver(const json &solver_params)
	{
		const std::string name = solver_params.contains("solver") ? solver_params["solver"].template get<std::string>() : "lbfgs";
		if (name == "GradientDescent" || name == "gradientdescent" || name == "gradient")
		{
			return std::make_shared<cppoptlib::GradientDescentSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgs" || name == "LBFGS" || name == "L-BFGS")
		{
			return std::make_shared<cppoptlib::LBFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "bfgs" || name == "BFGS" || name == "BFGS")
		{
			return std::make_shared<cppoptlib::BFGSSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "lbfgsb" || name == "LBFGSB" || name == "L-BFGS-B")
		{
			return std::make_shared<cppoptlib::LBFGSBSolver<ProblemType>>(
				solver_params);
		}
		else if (name == "mma" || name == "MMA")
		{
			return std::make_shared<cppoptlib::MMASolver<ProblemType>>(
				solver_params);
		}
		else
		{
			throw std::invalid_argument(fmt::format("invalid nonlinear solver type: {}", name));
		}
	}

	double matrix_dot(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) { return (A.array() * B.array()).sum(); }

	void topology_optimization(State &state, const std::shared_ptr<CompositeFunctional> j)
	{
		const auto &opt_params = state.args["optimization"];
		const auto &opt_nl_params = state.args["solver"]["optimization_nonlinear"];

		std::shared_ptr<TopologyOptimizationProblem> top_opt = std::make_shared<TopologyOptimizationProblem>(state, j);
		std::shared_ptr<cppoptlib::NonlinearSolver<TopologyOptimizationProblem>> nlsolver = make_nl_solver<TopologyOptimizationProblem>(opt_nl_params);
		nlsolver->setLineSearch(opt_nl_params["line_search"]["method"]);

		Eigen::MatrixXd density_mat = state.assembler.lame_params().density_mat_;
		if (density_mat.size() != state.bases.size())
			density_mat.setZero(state.bases.size(), 1);
		for (const auto &param : opt_params["parameters"])
		{
			if (param["type"] == "topology")
			{
				if (param.contains("initial"))
					density_mat.setConstant(param["initial"]);
				else
				{
					Eigen::MatrixXd barycenters;
					if (state.mesh->is_volume())
					{
						state.mesh->cell_barycenters(barycenters);
						for (int e = 0; e < state.bases.size(); e++)
						{
							density_mat(e) = cross3(barycenters(e,0), barycenters(e,1), barycenters(e,2));
						}
					}
					else
					{
						state.mesh->face_barycenters(barycenters);
						for (int e = 0; e < state.bases.size(); e++)
						{
							density_mat(e) = cross2(barycenters(e,0), barycenters(e,1));
						}
					}
					// density_mat.setOnes();
				}
				
				if (param.contains("power"))
					state.assembler.update_lame_params_density(top_opt->apply_filter(density_mat), param["power"]);
				else
					state.assembler.update_lame_params_density(top_opt->apply_filter(density_mat));
				break;
			}
		}

		Eigen::VectorXd x = density_mat;
		nlsolver->minimize(*top_opt, x);

		json solver_info;
		nlsolver->getInfo(solver_info);
		std::cout << solver_info << std::endl;
	}
} // namespace polyfem