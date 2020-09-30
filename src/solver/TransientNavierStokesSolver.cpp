#include <polyfem/TransientNavierStokesSolver.hpp>

#include <polyfem/MatrixUtils.hpp>
#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/AssemblerUtils.hpp>

#include <polyfem/Logger.hpp>

#include <igl/Timer.h>

#include <unsupported/Eigen/SparseExtra>

#include <cmath>

namespace polyfem
{
	using namespace polysolve;

TransientNavierStokesSolver::TransientNavierStokesSolver(const json &solver_param, const json &problem_params, const std::string &solver_type, const std::string &precond_type)
	: solver_param(solver_param), problem_params(problem_params), solver_type(solver_type), precond_type(precond_type)
{
}

void TransientNavierStokesSolver::minimize(
	const State &state, const bool is_full, const double alpha, const double dt, const Eigen::VectorXd &prev_sol, const Eigen::VectorXd &last_sol,
	const StiffnessMatrix &velocity_stiffness, const StiffnessMatrix &mixed_stiffness, const StiffnessMatrix &pressure_stiffness,
	const StiffnessMatrix &velocity_mass1,
	const Eigen::MatrixXd &rhs, Eigen::VectorXd &x)
{
	auto &assembler = AssemblerUtils::instance();
	assembler.clear_cache();
	const auto &gbases = state.iso_parametric() ? state.bases : state.geom_bases;

	const int problem_dim = state.problem->is_scalar() ? 1 : state.mesh->dimension();
	const int precond_num = problem_dim * state.n_bases;

	static bool factorized = false;
	if(!factorized)
	{
		velocity_mass = velocity_mass1/dt;
		AssemblerUtils::merge_mixed_matrices(state.n_bases, state.n_pressure_bases, problem_dim, state.use_avg_pressure,
										 velocity_stiffness + alpha * velocity_mass, mixed_stiffness, pressure_stiffness,
										 stoke_stiffness);
		
		logger().info("Prefactorization begins...");
		solver = LinearSolver::create(solver_type, precond_type);
		solver->setParameters(solver_param);
		polyfem::logger().debug("\tinternal solver {}", solver->name());
		StiffnessMatrix stoke_stiffness_temp = stoke_stiffness;
		prefactorize(*solver, stoke_stiffness_temp, state.boundary_nodes, precond_num, std::string());
		logger().info("Prefactorization ends!");
	}
	if(is_full)
	{
		factorized = true;
	}

	igl::Timer time;
	time.start();

	Eigen::VectorXd prev_sol_mass(rhs.size()); //prev_sol_mass=prev_sol
	prev_sol_mass.setZero();
	prev_sol_mass.block(0, 0, velocity_mass.rows(), 1) = velocity_mass * prev_sol.block(0, 0, velocity_mass.rows(), 1);
	for (int i : state.boundary_nodes)
		prev_sol_mass[i] = 0;

	assembler.assemble_energy_hessian(state.formulation() + "Picard", state.mesh->is_volume(), state.n_bases, state.bases, gbases, x, nl_matrix);
	
	Eigen::VectorXd last_sol_mass(rhs.size());
	last_sol_mass.setZero();
	last_sol_mass.block(0, 0, nl_matrix.rows(), 1) = nl_matrix * last_sol.block(0, 0, nl_matrix.rows(), 1);
	for (int i : state.boundary_nodes)
		last_sol_mass[i] = 0;

	time.stop();
	stokes_matrix_time = time.getElapsedTimeInSec();
	logger().debug("\tRhs calculation time {}s", time.getElapsedTimeInSec());

	time.start();

	Eigen::VectorXd b = rhs + prev_sol_mass + last_sol_mass;
	if (state.use_avg_pressure){
		b[b.size()-1] = 0;
	}

	dirichlet_solve_prefactorized(*solver, stoke_stiffness, b, state.boundary_nodes, x);
	time.stop();
	
	stokes_solve_time = time.getElapsedTimeInSec();
	logger().info("\tNavier-Stokes solve time {}s", time.getElapsedTimeInSec());
	// return;

	solver_info["time_assembly"] = stokes_matrix_time;
	solver_info["time_solve"] = stokes_solve_time;
}

} // namespace polyfem
