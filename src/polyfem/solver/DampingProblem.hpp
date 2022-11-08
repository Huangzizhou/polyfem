#pragma once

#include "OptimizationProblem.hpp"

namespace polyfem
{
	class DampingProblem : public OptimizationProblem
	{
	public:
		DampingProblem(State &state_, const std::shared_ptr<CompositeFunctional> j_);

		using OptimizationProblem::gradient;
		using OptimizationProblem::value;

		double target_value(const TVector &x) override;
		void target_gradient(const TVector &x, TVector &gradv) override;

		double value(const TVector &x) override;
		void gradient(const TVector &x, TVector &gradv) override;

		bool is_step_valid(const TVector &x0, const TVector &x1) override;

		int optimization_dim() override { return 2; }

		void line_search_end() override;
		bool remesh(TVector &x) override { return false; }

		bool solution_changed_pre(const TVector &newX) override;

		// TVector get_lower_bound(const TVector &x) const override;
		// TVector get_upper_bound(const TVector &x) const override;

		// map parameters to x
		std::function<void(TVector &x, const State &state)> param_to_x;

	private:
		json material_params;

		double target_weight = 1;

		double min_phi, min_psi;
		double max_phi, max_psi;
	};
} // namespace polyfem
