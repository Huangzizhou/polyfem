#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/LocalBoundary.hpp>
#include <polyfem/Problem.hpp>
#include <polysolve/FEMSolver.hpp>
#include <polyfem/Logger.hpp>
#include <Eigen/Core>

#include <polyfem/AssemblerUtils.hpp>
#include <memory>

#ifdef POLYFEM_WITH_TBB
#include <tbb/tbb.h>
#endif

using namespace polysolve;

namespace polyfem
{
    class OperatorSplittingSolver
    {
    public:
        // initialization
        void initialize_density_grid(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const double& density_dx);
        void initialize_mesh(const polyfem::Mesh& mesh, const int shape_, const int n_el_, const std::vector<LocalBoundary>& local_boundary);
        void initialize_hashtable(const polyfem::Mesh& mesh);
        void initialize_linear_solver(const std::string &solver_type, const std::string &precond, const json& params);

        OperatorSplittingSolver() {};
        OperatorSplittingSolver(const polyfem::Mesh& mesh, const int shape_, const int n_el_, const std::vector<LocalBoundary>& local_boundary, const std::string &solver_type, const std::string &precond, const json& params);

        // tools
        int handle_boundary_advection(RowVectorNd& pos);
        int trace_back(const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const RowVectorNd& pos_1, const RowVectorNd& vel_1, RowVectorNd& pos_2, RowVectorNd& vel_2, Eigen::MatrixXd& local_pos, const Eigen::MatrixXd& sol, const double dt);
        int interpolator(const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const RowVectorNd& pos, RowVectorNd& vel, Eigen::MatrixXd& local_pos, const Eigen::MatrixXd& sol);
        int search_cell(const std::vector<polyfem::ElementBases>& gbases, const RowVectorNd& pos, Eigen::MatrixXd& local_pts);
        bool outside_quad(const std::vector<RowVectorNd>& vert, const RowVectorNd& pos);
        void calculate_local_pts(const polyfem::ElementBases& gbase, const int elem_idx, const RowVectorNd& pos, Eigen::MatrixXd& local_pos);

        // different advections
        void advection(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1);
        void advection_FLIP(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1);
        void advection_PIC(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1);

        void solve_diffusion_1st(const StiffnessMatrix& mass, const StiffnessMatrix& stiffness_viscosity, const std::vector<int>& bnd_nodes, const Eigen::MatrixXd& bc, const Eigen::MatrixXd& force, Eigen::MatrixXd& sol, const double dt, const double visc);
        void solve_diffusion(const Eigen::VectorXd& history, const double alpha, const StiffnessMatrix& mass, const StiffnessMatrix& stiffness_viscosity, const std::vector<int>& bnd_nodes, const Eigen::MatrixXd& bc, Eigen::MatrixXd& sol, const double dt, const double visc);

        void external_force(const polyfem::Mesh& mesh, const AssemblerUtils& assembler, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const double dt, Eigen::MatrixXd& sol, const Eigen::MatrixXd& local_pts, const std::shared_ptr<Problem> problem, const double time);

        void solve_pressure(const StiffnessMatrix& stiffness_velocity, const StiffnessMatrix& mixed_stiffness, const Eigen::VectorXd& pressure_integrals, const std::vector<int>& pressure_boundary_nodes, Eigen::MatrixXd& sol, Eigen::MatrixXd& pressure);
        void projection(const StiffnessMatrix& velocity_mass, const StiffnessMatrix& mixed_stiffness, const std::vector<int>& boundary_nodes_, Eigen::MatrixXd& sol, const Eigen::MatrixXd& pressure);
        void projection(const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const std::vector<polyfem::ElementBases>& pressure_bases, const Eigen::MatrixXd& local_pts, Eigen::MatrixXd& pressure, Eigen::MatrixXd& sol);

        // functions for density
        void initialize_density(const std::shared_ptr<Problem>& problem);
        void save_density();
        void interpolator(const RowVectorNd& pos, double& val);
        void advect_density_exact(const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const std::shared_ptr<Problem> problem, const double t, const double dt, const int RK = 3);
        void advect_density(const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const Eigen::MatrixXd& sol, const double dt, const int RK = 3);
        
        // mesh info
        int dim;
        int n_el;
        int shape;
        RowVectorNd min_domain;
        RowVectorNd max_domain;
        Eigen::MatrixXd V;
        Eigen::MatrixXi T;

        // hash grid
        std::vector<std::vector<int> > hash_table;
        std::array<int, 3>          hash_table_cell_num;

        // FLIP and PIC
        std::vector<RowVectorNd> position_particle;
		std::vector<RowVectorNd> velocity_particle;
        std::vector<int> cellI_particle;
        Eigen::MatrixXd new_sol;
        Eigen::MatrixXd new_sol_w;

        // boundary condition
        std::vector<int> boundary_elem_id;

        // factorized matrices
        std::unique_ptr<polysolve::LinearSolver> solver_diffusion;
        std::unique_ptr<polysolve::LinearSolver> solver_projection;
        std::unique_ptr<polysolve::LinearSolver> solver_mass;

        // matrices
        StiffnessMatrix mat_projection;
        StiffnessMatrix mat_diffusion;

        // density
        Eigen::VectorXd density;
        Eigen::VectorXi grid_cell_num;
        double resolution;
    };
}
