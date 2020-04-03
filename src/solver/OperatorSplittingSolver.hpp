#include <polyfem/Mesh2D.hpp>
#include <polyfem/AssemblerUtils.hpp>
#include <memory>

#include <igl/AABB.h>
#include <igl/in_element.h>

namespace polyfem
{

    class OperatorSplittingSolver
    {
    public:
        OperatorSplittingSolver(polyfem::Mesh& mesh, const int shape, const int n_el) : shape(shape), n_el(n_el)
        {
            dim = mesh.dimension();
            if (shape == 3)
            {
                T.resize(n_el, 3);
                for (int e = 0; e < n_el; e++)
                {
                    for (int i = 0; i < shape; i++)
                    {
                        T(e, i) = mesh.cell_vertex_(e, i);
                    }
                }
            }
            else
            {
                T.resize(n_el * 2, 3);
                for (int e = 0; e < n_el; e++)
                {
                    for (int i = 0; i < 3; i++)
                    {
                        T(e, i) = mesh.cell_vertex_(e, i);
                        T(e + n_el, i) = mesh.cell_vertex_(e, (i + 2) % 4);
                    }
                }
            }

            V = Eigen::MatrixXd::Zero(mesh.n_vertices(), dim);
            for (int i = 0; i < V.rows(); i++)
            {
                auto p = mesh.point(i);
                for (int d = 0; d < dim; d++)
                {
                    V(i, d) = p(d);
                }
            }
            // to find the cell which a point is in
            tree.init(V, T);
        }

        void set_bc(polyfem::Mesh& mesh, const std::vector<polyfem::LocalBoundary>& local_boundary, const std::vector<int>& bnd_nodes, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const Eigen::MatrixXd& local_pts, std::shared_ptr<Problem> problem, const double time)
        {
            for (auto e = local_boundary.begin(); e != local_boundary.end(); e++)
            {
                auto elem = *e;
                int elem_idx = elem.element_id();

                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(elem_idx, i));
                }

                ElementAssemblyValues gvals;
                gvals.compute(elem_idx, mesh.is_volume(), local_pts, gbases[elem_idx], gbases[elem_idx]);

                for (int local_idx = 0; local_idx < bases[elem_idx].bases.size(); local_idx++)
                {
                    int global_idx = bases[elem_idx].bases[local_idx].global()[0].index;
                    if (find(bnd_nodes.begin(), bnd_nodes.end(), global_idx) == bnd_nodes.end())
                        continue;

                    Eigen::MatrixXd pos = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * vert[j](d);
                        }
                    }

                    Eigen::MatrixXd val;
                    problem->exact(pos, time, val);

                    for (int d = 0; d < dim; d++)
                    {
                        sol(global_idx * dim + d) = val(d);
                    }
                }
            }
        }

        void projection(polyfem::Mesh& mesh, int n_bases, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, const std::vector<polyfem::ElementBases>& pressure_bases, const Eigen::MatrixXd& local_pts, Eigen::MatrixXd& pressure, Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd grad_pressure = Eigen::VectorXd::Zero(n_bases * dim);
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_bases);

            ElementAssemblyValues vals;
            for (int e = 0; e < n_el; ++e)
            {
                vals.compute(e, mesh.is_volume(), local_pts, pressure_bases[e], gbases[e]);
                for (int j = 0; j < local_pts.rows(); j++)
                {
                    int global_ = bases[e].bases[j].global()[0].index;
                    for (int i = 0; i < vals.basis_values.size(); i++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            assert(pressure(pressure_bases[e].bases[i].global().size() == 1));
                            grad_pressure(global_ * dim + d) += vals.basis_values[i].grad_t_m(j, d) * pressure(pressure_bases[e].bases[i].global()[0].index);
                        }
                    }
                    traversed(global_)++;
                }
            }
            for (int i = 0; i < traversed.size(); i++)
            {
                for (int d = 0; d < dim; d++)
                {
                    sol(i * dim + d) -= grad_pressure(i * dim + d) / traversed(i);
                }
            }
        }

        void initialize_solution(polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, std::shared_ptr<Problem> problem, Eigen::MatrixXd& sol, const Eigen::MatrixXd& local_pts)
        {
            for (int e = 0; e < n_el; e++)
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, mesh.is_volume(), local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < vert.size(); j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pts(0, d) += vert[j](d) * gvals.basis_values[j].val(i);
                        }
                    }
                    Eigen::MatrixXd val;
                    problem->initial_solution(pts, val);
                    int global = bases[e].bases[i].global()[0].index;
                    for (int d = 0; d < dim; d++)
                    {
                        sol(global * dim + d) = val(d);
                    }
                }
            }
        }

        void advection(polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts)
        {
            auto det_func = [=](std::vector<RowVectorNd> V) ->double {
                Eigen::MatrixXd temp(dim + 1, dim + 1);
                for (int i = 0; i < dim + 1; i++)
                {
                    for (int j = 0; j < dim; j++)
                    {
                        temp(i, j) = V[i](j);
                    }
                    temp(i, dim) = 1;
                }
                return temp.determinant() / ((dim == 2) ? 2. : 6.);
            };

            // to store new velocity
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            // number of FEM nodes
            const int n_vert = sol.size() / dim;
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_vert);

            for (int e = 0; e < n_el; e++)
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; i++)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, mesh.is_volume(), local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    // global index of this FEM node
                    int global = bases[e].bases[i].global()[0].index;

                    if (traversed(global)) continue;
                    traversed(global) = 1;

                    // velocity of this FEM node
                    RowVectorNd vel(dim);
                    for (int d = 0; d < dim; d++)
                    {
                        vel(d) = sol(global * dim + d);
                    }

                    // global position of this FEM node
                    RowVectorNd pos = RowVectorNd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        pos += gvals.basis_values[j].val(i) * vert[j];
                    }

                    // semi-lagrangian trace back
                    pos = pos - vel * dt;

                    // to avoid that pos is out of domain
                    RowVectorNd min, max;
                    mesh.bounding_box(min, max);
                    for (int d = 0; d < dim; d++)
                    {
                        if (pos(d) <= min(d)) pos(d) = min(d) + 1e-12;
                        if (pos(d) >= max(d)) pos(d) = max(d) - 1e-12;
                    }

                    // a naive way to find the cell in which pos is
                    VectorXi I;
                    assert(dim == 2);
                    igl::in_element(V, T, pos, tree, I);
                    I(0) = I(0) % n_el;

                    std::vector<RowVectorNd> new_vert(shape);
                    for (int i = 0; i < shape; i++)
                    {
                        new_vert[i] = mesh.point(mesh.cell_vertex_(I(0), i));
                    }

                    // local coordinate of pos
                    Eigen::MatrixXd local_pos = Eigen::MatrixXd::Zero(1, dim);
                    ElementAssemblyValues gvals_;
                    gvals_.compute(I(0), mesh.is_volume(), local_pos, gbases[I(0)], gbases[I(0)]);

                    Eigen::MatrixXd res = -pos;
                    for (int i = 0; i < shape; i++)
                    {
                        res += new_vert[i] * gvals_.basis_values[i].val(0);
                    }

                    Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(dim, dim);
                    for (int d1 = 0; d1 < dim; d1++)
                    {
                        for (int d2 = 0; d2 < dim; d2++)
                        {
                            for (int i = 0; i < shape; i++)
                            {
                                jacobi(d1, d2) += new_vert[i](d1) * gvals_.basis_values[i].grad(0, d2);
                            }
                        }
                    }

                    Eigen::VectorXd delta = jacobi.colPivHouseholderQr().solve(res.transpose());
                    for (int d = 0; d < dim; d++)
                    {
                        local_pos(d) -= delta(d);
                    }

                    // interpolation
                    ElementAssemblyValues vals;
                    vals.compute(I(0), mesh.is_volume(), local_pos, bases[I(0)], gbases[I(0)]);
                    for (int d = 0; d < dim; d++)
                    {
                        for (int i = 0; i < vals.basis_values.size(); i++)
                        {
                            new_sol(global * dim + d) += vals.basis_values[i].val(0) * sol(bases[I(0)].bases[i].global()[0].index * dim + d);
                        }
                    }
                }
            }
            sol.swap(new_sol);
        }

        int dim;
        int n_el;
        int shape;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T;
        igl::AABB<MatrixXd, 2> tree;
    };
}
