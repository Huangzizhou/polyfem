#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polysolve/LinearSolver.hpp>

#include <polyfem/Logger.hpp>

#include <memory>

namespace polyfem
{

class TransientNavierStokesSolver
{
public:
	TransientNavierStokesSolver(const json &solver_param, const json &problem_params, const std::string &solver_type, const std::string &precond_type);

	void minimize(const State &state, const bool is_full, const double alpha, const double dt, const Eigen::VectorXd &prev_sol, const Eigen::VectorXd &last_sol,
				  const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
				  const StiffnessMatrix &velocity_mass,
				  const Eigen::MatrixXd &rhs, Eigen::VectorXd &x);
	void getInfo(json &params)
	{
		params = solver_info;
	}

	int error_code() const { return 0; }

	const json solver_param;
	const std::string solver_type;
	const std::string precond_type;

	json solver_info;
	json problem_params;

	json internal_solver = json::array();

	double stokes_matrix_time;
	double stokes_solve_time;

	StiffnessMatrix nl_matrix, stoke_stiffness;
	StiffnessMatrix velocity_mass;
	std::unique_ptr<polysolve::LinearSolver> solver;

	bool
	has_nans(const polyfem::StiffnessMatrix &hessian);
};
} // namespace polyfem
