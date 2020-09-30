#pragma once

#include <Eigen/Core>

#include <deque>

namespace polyfem
{
class BDF
{
public:
    BDF(int order);

    double alpha() const;
    bool is_full() const;
    void rhs(Eigen::VectorXd &rhs) const;
    void last_sol(Eigen::VectorXd &sol) const;

    void new_solution(Eigen::VectorXd &rhs);

private:
    std::deque<Eigen::VectorXd> history_;
    int order_;
};
} // namespace polyfem
