#pragma once

#include <polyfem/ProblemWithSolution.hpp>

#include <vector>
#include <Eigen/Dense>

namespace polyfem
{

	class FrankeProblem: public ProblemWithSolution
	{
	public:
		FrankeProblem(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return true; }

	};

	class FrankeProblemOld : public ProblemWithSolution
	{
	public:
		FrankeProblemOld(const std::string &name);

		VectorNd eval_fun(const VectorNd &pt) const override;
		AutodiffGradPt eval_fun(const AutodiffGradPt &pt) const override;
		AutodiffHessianPt eval_fun(const AutodiffHessianPt &pt) const override;

		bool is_scalar() const override { return true; }
	};
}


