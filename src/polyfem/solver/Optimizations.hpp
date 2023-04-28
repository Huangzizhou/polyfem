#pragma once

#include <polyfem/utils/CompositeFunctional.hpp>
#include "OptimizationProblem.hpp"

namespace polyfem
{
	void topology_optimization(State &state, const std::shared_ptr<CompositeFunctional> j);
} // namespace polyfem