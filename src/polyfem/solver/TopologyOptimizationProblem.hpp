#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class TopologyOptimizationProblem : public OptimizationProblem
	{
	public:
		TopologyOptimizationProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		double target_value(const TVector &x) { return j->energy(state) * target_weight; }

		void target_gradient(const TVector &x, TVector &gradv);

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x, const bool only_elastic) { return value(x); };
		void gradient(const TVector &x, TVector &gradv, const bool only_elastic) { gradient(x, gradv); };

		bool is_step_valid(const TVector &x0, const TVector &x1);
		double max_step_size(const TVector &x0, const TVector &x1) override;
		TVector force_inequality_constraint(const TVector &x0, const TVector &dx);
		bool remesh(TVector &x) { return false; };
		void line_search_begin(const TVector &x0, const TVector &x1) override;

		void direction_filtering(const TVector &x0, TVector &direc);

		TVector get_lower_bound(const TVector& x) override
		{
			TVector min(x.size());
			min.setConstant(min_density);
			for (int i = 0; i < min.size(); i++)
			{
				if (x(i) - min(i) > max_change)
					min(i) = x(i) - max_change;
			}
			return min; 
		}
		TVector get_upper_bound(const TVector& x) override
		{
			TVector max(x.size());
			max.setConstant(max_density);
			for (int i = 0; i < max.size(); i++)
			{
				if (max(i) - x(i) > max_change)
					max(i) = x(i) + max_change;
			}
			return max; 
		}

		void line_search_end(bool failed);

		bool solution_changed_pre(const TVector &newX) override;

		TVector apply_filter(const TVector &x);
		TVector apply_filter_to_grad(const TVector &x, const TVector &grad);

		int n_inequality_constraints() override;
		double inequality_constraint_val(const TVector &x, const int index) override;
		TVector inequality_constraint_grad(const TVector &x, const int index) override;
		
	private:
		double min_density = 0;
		double max_density = 1;

		double min_mass = 0;
		double max_mass = 1;

		double target_weight = 1;
		bool has_mass_constraint;

		json top_params;

		bool has_filter;
		Eigen::SparseMatrix<double> tt_radius_adjacency;
		Eigen::VectorXd tt_radius_adjacency_row_sum;
	};
} // namespace polyfem
