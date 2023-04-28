#pragma once

#include <polyfem/State.hpp>
#include <polyfem/utils/CompositeFunctional.hpp>
#include <polyfem/utils/MatrixUtils.hpp>

#include <cppoptlib/problem.h>
#include <filesystem>

namespace polyfem
{
	class OptimizationProblem : public cppoptlib::Problem<double>
	{
	public:
		using typename cppoptlib::Problem<double>::Scalar;
		using typename cppoptlib::Problem<double>::TVector;

		OptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		virtual ~OptimizationProblem() = default;

		void solve_pde(const TVector &x);

		virtual void smoothing(const TVector &x, TVector &new_x) {}
		virtual bool is_intersection_free(const TVector &x) { return true; }

		virtual void save_to_file(const TVector &x0)
		{
			if (iter % save_freq == 0)
			{
				logger().debug("Save to file {} ...", state.resolve_output_path(fmt::format("opt_{:d}.vtu", iter)));
				state.save_vtu(state.resolve_output_path(fmt::format("opt_{:d}.vtu", iter)), 0.);
			}
		}

		bool stop(const TVector &x) { return false; }

		virtual int optimization_dim() { return 0; }

		virtual bool solution_changed_pre(const TVector &newX) = 0;

		virtual void solution_changed_post(const TVector &newX) 
		{
			cur_x = newX;
			cur_grad.resize(0);
			cur_val = std::nan(""); 
		}

		virtual TVector get_lower_bound(const TVector& x) 
		{
			TVector min(x.size());
			min.setConstant(std::numeric_limits<double>::min());
			return min; 
		}
		virtual TVector get_upper_bound(const TVector& x) 
		{
			TVector max(x.size());
			max.setConstant(std::numeric_limits<double>::max());
			return max; 
		}

		void solution_changed(const TVector &newX);

		virtual void post_step(const int iter_num, const TVector &x0) { iter++; }

		virtual void line_search_begin(const TVector &x0, const TVector &x1);

		double heuristic_max_step(const TVector &dx)
		{
			assert(opt_nonlinear_params.contains("max_step_size"));
			return opt_nonlinear_params["max_step_size"];
		};

		virtual double max_step_size(const TVector &x0, const TVector &x1) { return 1; }
		virtual bool is_step_collision_free(const TVector &x0, const TVector &x1) { return true; }

		virtual int n_inequality_constraints() { return 0; }
		virtual double inequality_constraint_val(const TVector &x, const int index) { assert(false); return std::nan(""); }
		virtual TVector inequality_constraint_grad(const TVector &x, const int index) { assert(false); return TVector(); }

	protected:
		State &state;
		std::string optimization_name = "";

		int iter = 0;
		int save_iter = -1;

		int dim;
		int actual_dim;

		int save_freq = 1;

		std::shared_ptr<CompositeFunctional> j;

		TVector descent_direction;

		json opt_nonlinear_params;
		json opt_output_params;
		json opt_params;

		// better initial guess for forward solves
		Eigen::MatrixXd sol_at_ls_begin;
		TVector x_at_ls_begin;

		std::vector<Eigen::MatrixXd> sols_at_ls_begin;

		// store value and grad of current solution
		double cur_val;
		TVector cur_x, cur_grad;

		double max_change;
	};
} // namespace polyfem
