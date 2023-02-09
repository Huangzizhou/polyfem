#include "AdjointForm.hpp"

namespace polyfem::solver
{
    void AdjointForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
    {
        gradv.setZero(x.size());
        for (const auto &param_map : *variable_to_simulations)
        {
            const auto &parametrization = param_map.get_parameterization();
            const auto &state = param_map.get_state();
            const auto &param_type = param_map.get_parameter_type();

            gradv += parametrization.apply_jacobian(x, compute_adjoint_term(*state, param_type));
        }

        Eigen::VectorXd partial_grad;
        compute_partial_gradient(x, partial_grad);
        gradv += partial_grad;
    }

    Eigen::VectorXd AdjointForm::compute_partial_gradient(const Eigen::VectorXd &x, Eigen::VectorXd &gradv)
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
    }

    void AdjointForm::compute_adjoint_rhs(std::vector<Eigen::MatrixXd> &rhss)
    {
        log_and_throw_error("Should override this function in any AdjointForm!");
    }

    Eigen::VectorXd AdjointForm::compute_adjoint_term(const State &state, const ParameterType &param) const
    {
        Eigen::VectorXd term;
        AdjointTools::compute_adjoint_term(state, state.get_adjoint_mat(), param, term);
        return term;
    }
}