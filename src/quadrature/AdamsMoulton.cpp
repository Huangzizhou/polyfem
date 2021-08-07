#include <polyfem/AdamsMoulton.hpp>

#include <vector>
#include <array>

namespace polyfem
{
static const std::array<double, 4> alphas = {{1,
                                              1. / 2.,
                                              5. / 12.,
                                              9. / 24.}};

static const std::array<std::vector<double>, 4> weights =
    {{{},
      {1. / 2.},
      {2. / 3., -1. / 12.},
      {19. / 24., -5. / 24., 1. / 24.}}};

AdamsMoulton::AdamsMoulton(int order) : order_(order)
{
    order_ = std::max(1, std::min(order_, 4));
}

double AdamsMoulton::alpha() const
{
    return alphas[history_.size()];
}

void AdamsMoulton::rhs(Eigen::VectorXd &rhs) const
{
    if (history_.size() == 0) {
        rhs.resize(0);
        return;
    }
    rhs.resize(history_.front().size());
    rhs.setZero();
    const auto &w = weights[history_.size()];
    for (int i = 0; i < history_.size(); ++i)
    {
        rhs += history_[i] * w[i];
    }
}

void AdamsMoulton::new_solution(const Eigen::VectorXd &rhs)
{
    if (order_ > 1) {
        if (history_.size() >= order_ - 1)
        {
            history_.pop_front();
        }

        history_.push_back(rhs);
    }
    assert(history_.size() <= order_ - 1);
}
} // namespace polyfem
