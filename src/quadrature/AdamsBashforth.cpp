#include <polyfem/AdamsBashforth.hpp>

#include <vector>
#include <array>

namespace polyfem
{
static const std::array<std::vector<double>, 4> weights =
    {{{1},
      {-1. / 2., 3. / 2.},
      {5. / 12., -16. / 12., 23. / 12.},
      {-9. / 24., 37. / 24., -59. / 24., 55. / 24.}}};

AdamsBashforth::AdamsBashforth(int order) : order_(order)
{
    order_ = std::max(1, std::min(order_, 4));
}

void AdamsBashforth::rhs(Eigen::VectorXd &rhs) const
{
    assert(history_.size() > 0);
    rhs.resize(history_.front().size());
    rhs.setZero();
    const auto &w = weights[history_.size() - 1];
    for (int i = 0; i < history_.size(); ++i)
    {
        rhs += history_[i] * w[i];
    }
}

void AdamsBashforth::new_solution(const Eigen::VectorXd &rhs)
{
    if (history_.size() >= order_)
    {
        history_.pop_front();
    }

    history_.push_back(rhs);
    assert(history_.size() <= order_);
}
} // namespace polyfem
