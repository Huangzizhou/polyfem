#pragma once

#include <polyfem/Common.hpp>
#include <polyfem/State.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polyfem/Logger.hpp>

#include <polyfem/AssemblerUtils.hpp>
#include <memory>

#include <igl/AABB.h>
#include <igl/in_element.h>

#ifdef POLYFEM_WITH_TBB
#include <tbb/tbb.h>
#endif

using namespace polysolve;

namespace polyfem
{

    class OperatorSplittingSolver
    {
    public:
        OperatorSplittingSolver(const polyfem::Mesh& mesh, 
        const int shape, const int n_el, 
        const std::vector<polyfem::LocalBoundary>& local_boundary,
        const std::vector<int>& boundary_nodes) : shape(shape), n_el(n_el), boundary_nodes(boundary_nodes)
        {
            dim = mesh.dimension();

            const int size = local_boundary.size();
            boundary_elem_id.reserve(size);
            for(int e = 0; e < size; e++)
            {
                boundary_elem_id.push_back(local_boundary[e].element_id());
            }

            T.resize(n_el, shape);
            for (int e = 0; e < n_el; e++)
            {
                for (int i = 0; i < shape; i++)
                {
                    T(e, i) = mesh.cell_vertex_(e, i);
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

            cell_num = (int)pow(n_el, 1./dim);
            hash_table.resize((int)pow(cell_num, dim));

            mesh.bounding_box(min_domain, max_domain);

            for(int e = 0; e < T.rows(); e++)
            {
                Eigen::VectorXd min_ = V.row(T(e, 0));
                Eigen::VectorXd max_ = min_;

                for(int i = 1; i < T.cols(); i++)
                {
                    Eigen::VectorXd p = V.row(T(e, i));
                    for(int d = 0; d < dim; d++)
                    {
                        if(min_(d) > p(d)) min_(d) = p(d);
                        if(max_(d) < p(d)) max_(d) = p(d);
                    }
                }

                Eigen::VectorXi min_int(dim), max_int(dim);

                for(int d = 0; d < dim; d++)
                {
                    double temp = cell_num / (max_domain(d) - min_domain(d));
                    min_int(d) = floor((min_(d) - min_domain(d)) * temp);
                    max_int(d) = ceil((max_(d) - min_domain(d)) * temp);

                    if(min_int(d) < 0) 
                        min_int(d) = 0;
                    if(max_int(d) > cell_num)
                        max_int(d) = cell_num;
                }

                for(int x = min_int(0); x < max_int(0); x++)
                {
                    for(int y = min_int(1); y < max_int(1); y++)
                    {
                        if(dim == 2)
                        {
                            int idx = x + y * cell_num;
                            hash_table[idx].push_front(e);
                        }
                        else
                        {
                            for(int z = min_int(2); z < max_int(2); z++)
                            {
                                int idx = x + (y + z * cell_num) * cell_num;
                                hash_table[idx].push_front(e);
                            }
                        }
                    }
                }
            }
        }

        int handle_boundary_advection(RowVectorNd& pos)
        {
            double dist = 1e10;
            int idx = -1, local_idx = -1;
            const int size = boundary_elem_id.size();
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, size, 1, [&](int e)
#else
            for(int e = 0; e < size; e++)
#endif
            {
                int elem_idx = boundary_elem_id[e];

                for (int i = 0; i < shape; i++)
                {
                    double dist_ = 0;
                    for(int d = 0; d < dim; d++)
                    {
                        dist_ += pow( pos(d) - V(T(elem_idx, i), d), 2);
                    }
                    dist_ = sqrt(dist_);
                    if(dist_ < dist)
                    {
                        dist = dist_;
                        idx = elem_idx;
                        local_idx = i;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            for(int d = 0; d < dim; d++)
                pos(d) = V(T(idx, local_idx), d);
            return idx;
        }

        void trace_back(const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const RowVectorNd& pos_1, 
        const RowVectorNd& vel_1, 
        RowVectorNd& pos_2, 
        RowVectorNd& vel_2, 
        const Eigen::MatrixXd& sol,
        const double dt)
        {
            int new_elem;
            Eigen::MatrixXd local_pos;

            pos_2 = pos_1 - vel_1 * dt;

            if((new_elem = search_cell(gbases, pos_2, local_pos)) == -1)
            {
                new_elem = handle_boundary_advection(pos_2);
                calculate_local_pts(gbases[new_elem], new_elem, pos_2, local_pos);
            }

            // interpolation
            vel_2 = RowVectorNd::Zero(dim);
            ElementAssemblyValues vals;
            vals.compute(new_elem, dim == 3, local_pos, bases[new_elem], gbases[new_elem]);
            for (int d = 0; d < dim; d++)
            {
                for (int i = 0; i < vals.basis_values.size(); i++)
                {
                    vel_2(d) += vals.basis_values[i].val(0) * sol(bases[new_elem].bases[i].global()[0].index * dim + d);
                }
            }
        }

        void advection(const polyfem::Mesh& mesh, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        Eigen::MatrixXd& sol, 
        const double dt, 
        const Eigen::MatrixXd& local_pts, 
        const int order = 1,
        const int RK = 1)
        {
            // to store new velocity
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            // number of FEM nodes
            const int n_vert = sol.size() / dim;
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_vert);

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape, RowVectorNd::Zero(1, dim));
                for (int i = 0; i < shape; i++)
                {
                    int tmp = T(e,i);
                    for(int d = 0; d < dim; d++)
                        vert[i](d) = V(tmp,d);
                }

                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    // global index of this FEM node
                    int global = bases[e].bases[i].global()[0].index;

                    if (traversed(global)) continue;
                    traversed(global) = 1;

                    RowVectorNd vel_1[4], pos_1[4];

                    // velocity of this FEM node
                    vel_1[0] = sol.block(global * dim, 0, dim, 1).transpose();

                    // global position of this FEM node
                    pos_1[0] = RowVectorNd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        pos_1[0] += gvals.basis_values[j].val(i) * vert[j];
                    }

                    if(RK>=3)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[1], vel_1[1], sol, 0.5 * dt);
                        trace_back( gbases, bases, pos_1[0], vel_1[1], pos_1[2], vel_1[2], sol, 0.75 * dt);
                        trace_back( gbases, bases, pos_1[0], 2 * vel_1[0] + 3 * vel_1[1] + 4 * vel_1[2], pos_1[3], vel_1[3], sol, dt / 9);
                    }
                    else if(RK==2)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[1], vel_1[1], sol, 0.5 * dt);
                        trace_back( gbases, bases, pos_1[0], vel_1[1], pos_1[3], vel_1[3], sol, dt);
                    }
                    else if(RK==1)
                    {
                        trace_back( gbases, bases, pos_1[0], vel_1[0], pos_1[3], vel_1[3], sol, dt);
                    }

                    new_sol.block(global * dim, 0, dim, 1) = vel_1[3].transpose();

                    if(order == 2)
                    {
                        RowVectorNd vel_2[3], pos_2[3];

                        if(RK>=3)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[0], vel_2[0], sol, -0.5 * dt);
                            trace_back( gbases, bases, pos_1[3], vel_1[1], pos_2[1], vel_2[1], sol, -0.75 * dt);
                            trace_back( gbases, bases, pos_1[3], 2 * vel_1[3] + 3 * vel_2[0] + 4 * vel_2[1], pos_2[2], vel_2[2], sol, -dt / 9);
                        }
                        else if(RK==2)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[0], vel_2[0], sol, -0.5 * dt);
                            trace_back( gbases, bases, pos_1[3], vel_2[0], pos_2[2], vel_2[2], sol, -dt);
                        }
                        else if(RK==1)
                        {
                            trace_back( gbases, bases, pos_1[3], vel_1[3], pos_2[2], vel_2[2], sol, -dt);
                        }
                        
                        new_sol.block(global * dim, 0, dim, 1) += (vel_1[0] - vel_2[2]).transpose() / 2;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            sol.swap(new_sol);
        }

        void advection_FLIP(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1)
        {
            const int ppe = shape; // particle per element
            const double FLIPRatio = 1;
            // initialize or resample particles and update velocity via g2p
            if (position_particle.empty()) {
                // initialize particles
                position_particle.resize(n_el * ppe);
                velocity_particle.resize(n_el * ppe);
                cellI_particle.resize(n_el * ppe);
#ifdef POLYFEM_WITH_TBB
                tbb::parallel_for(0, n_el, 1, [&](int e)
#else
                for (int e = 0; e < n_el; ++e)
#endif
                {
                    // sample particle in element e
                    Eigen::MatrixXd local_pts_particle;
                    local_pts_particle.setRandom(ppe, dim);
                    local_pts_particle.array() += 1;
                    local_pts_particle.array() /= 2;

                    // geometry vertices of element e
                    std::vector<RowVectorNd> vert(shape);
                    for (int i = 0; i < shape; ++i)
                    {
                        cellI_particle[e * ppe + i] = e;
                        vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                    }

                    // compute global position and velocity of particles
                    // construct interpolant (linear for position)
                    ElementAssemblyValues gvals;
                    gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);
                    // construct interpolant (for velocity)
                    ElementAssemblyValues vals;
                    vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                    for (int j = 0; j < ppe; ++j) {
                        position_particle[ppe * e + j].setZero(1, dim);
                        for (int i = 0; i < shape; ++i)
                        {
                            position_particle[ppe * e + j] += gvals.basis_values[i].val(j) * vert[i];
                        }
                        
                        velocity_particle[e * ppe + j].setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }
                    }
                }
#ifdef POLYFEM_WITH_TBB
                );
#endif
            }
            else {
                // resample particles
                // count particle per cell
                std::vector<int> counter(n_el, 0);
                std::vector<int> redundantPI;
                std::vector<bool> isRedundant(cellI_particle.size(), false);
                for (int pI = 0; pI < cellI_particle.size(); ++pI) {
                    ++counter[cellI_particle[pI]];
                    if (counter[cellI_particle[pI]] > ppe) {
                        redundantPI.emplace_back(pI);
                        isRedundant[pI] = true;
                    }
                }
                // g2p -- update velocity 
#ifdef POLYFEM_WITH_TBB
                tbb::parallel_for(0, (int)cellI_particle.size(), 1, [&](int pI)
#else
                for (int pI = 0; pI < cellI_particle.size(); ++pI) 
#endif
                {
                    if (!isRedundant[pI]) {
                        int e = cellI_particle[pI];
                        Eigen::MatrixXd local_pts_particle;
                        calculate_local_pts(gbases[e], e, position_particle[pI], local_pts_particle);
                        
                        ElementAssemblyValues vals;
                        vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                        RowVectorNd FLIPdVel, PICVel;
                        FLIPdVel.setZero(1, dim);
                        PICVel.setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            FLIPdVel += vals.basis_values[i].val(0) * 
                                (sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1) -
                                new_sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1)).transpose();
                            PICVel += vals.basis_values[i].val(0) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }
                        velocity_particle[pI] = (1.0 - FLIPRatio) * PICVel + 
                            FLIPRatio * (velocity_particle[pI] + FLIPdVel);
                    }
                }
#ifdef POLYFEM_WITH_TBB
                );
#endif
                // resample
                for (int e = 0; e < n_el; ++e) {
                    if (counter[e] >= ppe) {
                        continue;
                    }

                    // geometry vertices of element e
                    std::vector<RowVectorNd> vert(shape);
                    for (int i = 0; i < shape; ++i)
                    {
                        vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                    }
                    while (counter[e] < ppe) {
                        int pI = redundantPI.back();
                        redundantPI.pop_back();
                        
                        cellI_particle[pI] = e;

                        // sample particle in element e
                        Eigen::MatrixXd local_pts_particle;
                        local_pts_particle.setRandom(1, dim);
                        local_pts_particle.array() += 1;
                        local_pts_particle.array() /= 2;

                        // compute global position and velocity of particles
                        // construct interpolant (linear for position)
                        ElementAssemblyValues gvals;
                        gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);
                        position_particle[pI].setZero(1, dim);
                        for (int i = 0; i < shape; ++i)
                        {
                            position_particle[pI] += gvals.basis_values[i].val(0) * vert[i];
                        }

                        // construct interpolant (for velocity)
                        ElementAssemblyValues vals;
                        vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                        velocity_particle[pI].setZero(1, dim);
                        for (int i = 0; i < vals.basis_values.size(); ++i)
                        {
                            velocity_particle[pI] += vals.basis_values[i].val(0) * 
                                sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                        }

                        ++counter[e];
                    }
                }
            }

            // advect
            std::vector<ElementAssemblyValues> velocity_interpolator(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)(ppe * n_el), 1, [&](int pI)
#else
            for (int pI = 0; pI < ppe * n_el; ++pI) 
#endif
            {
                // update particle position via advection
                RowVectorNd newvel;
                trace_back( gbases, bases, position_particle[pI], velocity_particle[pI], 
                    position_particle[pI], newvel, sol, -dt);

                // RK3:
                // RowVectorNd bypass, vel2, vel3;
                // trace_back( gbases, bases, position_particle[pI], velocity_particle[pI], 
                //     bypass, vel2, sol, -0.5 * dt);
                // trace_back( gbases, bases, position_particle[pI], vel2, 
                //     bypass, vel3, sol, -0.75 * dt);
                // trace_back( gbases, bases, position_particle[pI], 
                //     2 * velocity_particle[pI] + 3 * vel2 + 4 * vel3, 
                //     position_particle[pI], bypass, sol, -dt / 9);

                // prepare P2G
                Eigen::VectorXi I(1);
                Eigen::MatrixXd local_pos;

                if((I(0) = search_cell(gbases, position_particle[pI], local_pos)) == -1)
                {
                    I(0) = handle_boundary_advection(position_particle[pI]);
                    calculate_local_pts(gbases[I(0)], I(0), position_particle[pI], local_pos);
                }

                // construct interpolator (always linear for P2G, can use gaussian or bspline later)
                velocity_interpolator[pI].compute(I(0), dim == 3, local_pos, gbases[I(0)], gbases[I(0)]);
                cellI_particle[pI] = I(0);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif

            // P2G
            new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
            new_sol_w.array() += 1e-13;
            for (int pI = 0; pI < ppe * n_el; ++pI) {
                int cellI = cellI_particle[pI];
                ElementAssemblyValues& vals = velocity_interpolator[pI];
                for (int i = 0; i < vals.basis_values.size(); ++i)
                {
                    new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) += 
                        vals.basis_values[i].val(0) * velocity_particle[pI].transpose();
                    new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
                }
            }
            //TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
            for (int i = 0; i < new_sol.rows() / dim; ++i) 
#endif
            {
                new_sol.block(i * dim, 0, dim, 1) /= new_sol_w(i, 0);
                sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void advection_PIC(const polyfem::Mesh& mesh, const std::vector<polyfem::ElementBases>& gbases, const std::vector<polyfem::ElementBases>& bases, Eigen::MatrixXd& sol, const double dt, const Eigen::MatrixXd& local_pts, const int order = 1)
        {
            // to store new velocity and weights for particle grid transfer
            Eigen::MatrixXd new_sol = Eigen::MatrixXd::Zero(sol.size(), 1);
            Eigen::MatrixXd new_sol_w = Eigen::MatrixXd::Zero(sol.size() / dim, 1);
            new_sol_w.array() += 1e-13;

            const int ppe = shape; // particle per element
            std::vector<ElementAssemblyValues> velocity_interpolator(ppe * n_el);
            position_particle.resize(ppe * n_el);
            velocity_particle.resize(ppe * n_el);
            cellI_particle.resize(ppe * n_el);
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // resample particle in element e
                Eigen::MatrixXd local_pts_particle;
                local_pts_particle.setRandom(ppe, dim);
                local_pts_particle.array() += 1;
                local_pts_particle.array() /= 2;

                // geometry vertices of element e
                std::vector<RowVectorNd> vert(shape);
                for (int i = 0; i < shape; ++i)
                {
                    vert[i] = mesh.point(mesh.cell_vertex_(e, i));
                }

                // construct interpolant (linear for position)
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts_particle, gbases[e], gbases[e]);

                // compute global position of particles
                for (int i = 0; i < ppe; ++i)
                {
                    position_particle[ppe * e + i].setZero(1, dim);
                    for (int j = 0; j < shape; ++j)
                    {
                        position_particle[ppe * e + i] += gvals.basis_values[j].val(i) * vert[j];
                    }
                }

                // compute velocity
                ElementAssemblyValues vals;
                vals.compute(e, dim == 3, local_pts_particle, bases[e], gbases[e]); // possibly higher-order
                for (int j = 0; j < ppe; ++j) {
                    velocity_particle[e * ppe + j].setZero(1, dim);
                    for (int i = 0; i < vals.basis_values.size(); ++i)
                    {
                        velocity_particle[e * ppe + j] += vals.basis_values[i].val(j) * 
                            sol.block(bases[e].bases[i].global()[0].index * dim, 0, dim, 1).transpose();
                    }
                }

                // update particle position via advection
                for (int i = 0; i < ppe; ++i) {
                    RowVectorNd newvel;
                    trace_back( gbases, bases, position_particle[ppe * e + i], velocity_particle[e * ppe + i], 
                        position_particle[ppe * e + i], newvel, sol, -dt);

                    // RK3:
                    // RowVectorNd bypass, vel2, vel3;
                    // trace_back( gbases, bases, position_particle[ppe * e + i], velocity_particle[e * ppe + i], 
                    //     bypass, vel2, sol, -0.5 * dt);
                    // trace_back( gbases, bases, position_particle[ppe * e + i], vel2, 
                    //     bypass, vel3, sol, -0.75 * dt);
                    // trace_back( gbases, bases, position_particle[ppe * e + i], 
                    //     2 * velocity_particle[e * ppe + i] + 3 * vel2 + 4 * vel3, 
                    //     position_particle[ppe * e + i], bypass, sol, -dt / 9);
                }

                // prepare P2G
                for (int j = 0; j < ppe; ++j) {
                    Eigen::VectorXi I(1);
                    Eigen::MatrixXd local_pos;
                    
                    // find cell
                    if((I(0) = search_cell(gbases, position_particle[ppe * e + j],local_pos)) == -1)
                    {
                        I(0) = handle_boundary_advection(position_particle[ppe * e + j]);
                        calculate_local_pts(gbases[I(0)], I(0), position_particle[ppe * e + j], local_pos);
                    }

                    // construct interpolator (always linear for P2G, can use gaussian or bspline later)
                    velocity_interpolator[ppe * e + j].compute(I(0), dim == 3, local_pos, gbases[I(0)], gbases[I(0)]);
                    cellI_particle[ppe * e + j] = I(0);
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif

            // P2G
            for (int e = 0; e < n_el; ++e)
            {
                for (int j = 0; j < ppe; ++j) 
                {
                    int cellI = cellI_particle[ppe * e + j];
                    ElementAssemblyValues& vals = velocity_interpolator[ppe * e + j];
                    for (int i = 0; i < vals.basis_values.size(); ++i)
                    {
                        new_sol.block(bases[cellI].bases[i].global()[0].index * dim, 0, dim, 1) += 
                            vals.basis_values[i].val(0) * velocity_particle[ppe * e + j].transpose();
                        new_sol_w(bases[cellI].bases[i].global()[0].index) += vals.basis_values[i].val(0);
                    }
                }
            }
            //TODO: need to add up boundary velocities and weights because of perodic BC

#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, (int)new_sol.rows() / dim, 1, [&](int i)
#else
            for (int i = 0; i < new_sol.rows() / dim; ++i) 
#endif
            {
                sol.block(i * dim, 0, dim, 1) = new_sol.block(i * dim, 0, dim, 1) / new_sol_w(i, 0);
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
            //TODO: need to think about what to do with negative quadratic weight
        }

        void set_bc(const polyfem::Mesh& mesh, 
        const std::vector<int>& bnd_nodes,
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts, 
        const std::shared_ptr<Problem> problem, 
        const double time)
        {
            const int size = boundary_elem_id.size();
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, size, 1, [&](int e)
#else
            for(int e = 0; e < size; e++)
#endif
            {
                int elem_idx = boundary_elem_id[e];

                // geometry vertices of element e
                Eigen::MatrixXd vert(shape, dim);
                for (int i = 0; i < shape; i++)
                {
                    for(int d = 0; d < dim; d++)
                        vert(i, d) = V(T(elem_idx, i), d);
                }

                ElementAssemblyValues gvals;
                gvals.compute(elem_idx, dim == 3, local_pts, gbases[elem_idx], gbases[elem_idx]);

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
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * vert(j, d);
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
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void solve_diffusion_1st(const std::string &solver_type, 
        const std::string &precond,
        const json& params,
        const StiffnessMatrix& mass,
        const StiffnessMatrix& stiffness_viscosity,
        const std::vector<int>& bnd_nodes,
        const double& dt,
        const double& viscosity_,
        const std::string &save_path,
	    bool compute_spectrum,
        Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd rhs;
            StiffnessMatrix A;
            
            for(int d = 0; d < dim; d++)
            {
                auto solver = LinearSolver::create(solver_type, precond);
                solver->setParameters(params);
                logger().info("{}...", solver->name());

                Eigen::VectorXd x(sol.size() / dim);
                for(int j = 0; j < x.size(); j++)
                {
                    x(j) = sol(j * dim + d);
                }
                A = mass + viscosity_ * dt * stiffness_viscosity;
                rhs = mass * x;

                // keep dirichlet bc
                for (int i = 0; i < bnd_nodes.size(); i++)
                {
                    rhs(bnd_nodes[i]) = x(bnd_nodes[i]);
                }

                const int precond_num = A.rows();

                auto spectrum = dirichlet_solve(*solver, A, rhs, bnd_nodes, x, precond_num, save_path, compute_spectrum);

                for(int j = 0; j < x.size(); j++)
                {
                    sol(j * dim + d) = x(j);
                }
            }
        }

        void solve_diffusion_2nd(const std::string &solver_type, 
        const std::string &precond,
        const json& params,
        const StiffnessMatrix& mass,
        const StiffnessMatrix& stiffness_viscosity,
        const std::vector<int>& bnd_nodes,
        const double& dt,
        const double& viscosity_,
        const std::string &save_path,
	    bool compute_spectrum,
        Eigen::MatrixXd& sol)
        {
            for(int d = 0; d < dim; d++)
            {
                auto solver = LinearSolver::create(solver_type, precond);
                solver->setParameters(params);
                logger().info("{}...", solver->name());

                Eigen::VectorXd x(sol.size() / dim);
                for(int j = 0; j < x.size(); j++)
                {
                    x(j) = sol(j * dim + d);
                }
                Eigen::VectorXd rhs = mass * x - 0.5 * dt * viscosity_ * stiffness_viscosity * x;
				StiffnessMatrix A = mass;

                // keep dirichlet bc
                for (int i = 0; i < bnd_nodes.size(); i++)
                {
                    rhs(bnd_nodes[i]) = x(bnd_nodes[i]);
                }

                const int precond_num = A.rows();

                auto spectrum = dirichlet_solve(*solver, A, rhs, bnd_nodes, x, precond_num, save_path, compute_spectrum);

                A = mass + 0.5 * dt * viscosity_ * stiffness_viscosity;
                rhs = mass * x;

                // keep dirichlet bc
                for (int i = 0; i < bnd_nodes.size(); i++)
                {
                    rhs(bnd_nodes[i]) = x(bnd_nodes[i]);
                }

                spectrum = dirichlet_solve(*solver, A, rhs, bnd_nodes, x, precond_num, save_path, compute_spectrum);

                for(int j = 0; j < x.size(); j++)
                {
                    sol(j * dim + d) = x(j);
                }
            }
        }

        void external_force(const polyfem::Mesh& mesh,
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const double dt, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts, 
        const std::shared_ptr<Problem> problem, 
        const double time)
        {
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for(int e = 0; e < n_el; e++)
#endif
            {
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int local_idx = 0; local_idx < bases[e].bases.size(); local_idx++)
                {
                    int global_idx = bases[e].bases[local_idx].global()[0].index;

                    Eigen::MatrixXd pos = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pos(0, d) += gvals.basis_values[j].val(local_idx) * V(T(e, j), d);
                        }
                    }

                    Eigen::MatrixXd val;
                    problem->rhs(std::string(), pos, time, val);

                    for (int d = 0; d < dim; d++)
                    {
                        sol(global_idx * dim + d) += val(d) * dt;
                    }
                }
            }
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        void solve_pressure(const std::string &solver_type, 
        const std::string &precond, 
        const json& params,
        const StiffnessMatrix& stiffness,
        const StiffnessMatrix& mixed_stiffness,
        const std::string &save_path,
	    bool compute_spectrum,
        Eigen::MatrixXd& sol, 
        Eigen::MatrixXd& pressure)
        {
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(mixed_stiffness.rows() + 1); // mixed_stiffness * sol;
			StiffnessMatrix A(stiffness.rows() + 1, stiffness.cols() + 1); // stiffness;
            
            Eigen::VectorXd temp = mixed_stiffness * sol;
            for(int i = 0; i < temp.rows(); i++)
            {
                rhs(i) = temp(i);
            }

            std::vector<Eigen::Triplet<double> > coefficients;
            coefficients.reserve(stiffness.nonZeros() + 2 * stiffness.rows());

            for(int i = 0; i < stiffness.outerSize(); i++)
            {
                for(StiffnessMatrix::InnerIterator it(stiffness,i); it; ++it)
                {
                    coefficients.push_back(Eigen::Triplet<double>(it.row(),it.col(),it.value()));
                }
            }

            const double val = 1. / (A.rows() - 1);
            for (int i = 0; i < A.rows() - 1; i++)
            {
                coefficients.push_back(Eigen::Triplet<double>(i, A.cols() - 1, val));
                coefficients.push_back(Eigen::Triplet<double>(A.rows() - 1, i, val));
            }

            A.setFromTriplets(coefficients.begin(), coefficients.end());

            auto solver = LinearSolver::create(solver_type, precond);
            solver->setParameters(params);
            logger().info("{}...", solver->name());

            Eigen::VectorXd x;

            const int precond_num = A.rows() - 1;

            auto spectrum = dirichlet_solve(*solver, A, rhs, std::vector<int>(), x, precond_num, save_path, compute_spectrum);
            pressure = x;
        }

        void projection(const polyfem::Mesh& mesh, 
        int n_bases, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const std::vector<polyfem::ElementBases>& pressure_bases, 
        const Eigen::MatrixXd& local_pts, 
        Eigen::MatrixXd& pressure, 
        Eigen::MatrixXd& sol)
        {
            Eigen::VectorXd grad_pressure = Eigen::VectorXd::Zero(n_bases * dim);
            Eigen::VectorXi traversed = Eigen::VectorXi::Zero(n_bases);

            ElementAssemblyValues vals;
            for (int e = 0; e < n_el; ++e)
            {
                vals.compute(e, dim == 3, local_pts, pressure_bases[e], gbases[e]);
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

        void initialize_solution(const polyfem::Mesh& mesh, 
        const std::vector<polyfem::ElementBases>& gbases, 
        const std::vector<polyfem::ElementBases>& bases, 
        const std::shared_ptr<Problem> problem, 
        Eigen::MatrixXd& sol, 
        const Eigen::MatrixXd& local_pts)
        {
#ifdef POLYFEM_WITH_TBB
            tbb::parallel_for(0, n_el, 1, [&](int e)
#else
            for (int e = 0; e < n_el; ++e)
#endif
            {
                // to compute global position with barycentric coordinate
                ElementAssemblyValues gvals;
                gvals.compute(e, dim == 3, local_pts, gbases[e], gbases[e]);

                for (int i = 0; i < local_pts.rows(); i++)
                {
                    Eigen::MatrixXd pts = Eigen::MatrixXd::Zero(1, dim);
                    for (int j = 0; j < shape; j++)
                    {
                        for (int d = 0; d < dim; d++)
                        {
                            pts(0, d) += V(T(e, j), d) * gvals.basis_values[j].val(i);
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
#ifdef POLYFEM_WITH_TBB
            );
#endif
        }

        int search_cell(const std::vector<polyfem::ElementBases>& gbases, RowVectorNd& pos, Eigen::MatrixXd& local_pts)
        {
            Eigen::VectorXi pos_int(dim);
            for(int d = 0; d < dim; d++)
            {
                pos_int(d) = floor((pos(d) - min_domain(d)) / (max_domain(d) - min_domain(d)) * cell_num);
                if(pos_int(d) < 0) pos_int(d) = 0;
                else if(pos_int(d) >= cell_num) pos_int(d) = cell_num - 1;
            }

            int idx = 0, dim_num = 1;
            for(int d = 0; d < dim; d++)
            {
                idx += pos_int(d) * dim_num;
                dim_num *= cell_num;
            }

            const std::list<int>& list = hash_table[idx];
            for(auto it = list.begin(); it != list.end(); it++)
            {
                calculate_local_pts(gbases[*it], *it, pos, local_pts);

                if(shape == dim + 1)
                {
                    if(local_pts.minCoeff() > -1e-13 && local_pts.sum() < 1 + 1e-13)
                        return *it;
                }
                else
                {
                    if(local_pts.minCoeff() > -1e-13 && local_pts.maxCoeff() < 1 + 1e-13)
                        return *it;
                }
            }
            return -1; // not inside any elem
        }

        bool outside_quad(const std::vector<RowVectorNd>& vert, const RowVectorNd& pos)
        {
            double a = (vert[1](0) - vert[0](0)) * (pos(1) - vert[0](1)) - (vert[1](1)-vert[0](1)) * (pos(0) - vert[0](0));
            double b = (vert[2](0) - vert[1](0)) * (pos(1) - vert[1](1)) - (vert[2](1)-vert[1](1)) * (pos(0) - vert[1](0));
            double c = (vert[3](0) - vert[2](0)) * (pos(1) - vert[2](1)) - (vert[3](1)-vert[2](1)) * (pos(0) - vert[2](0));
            double d = (vert[0](0) - vert[3](0)) * (pos(1) - vert[3](1)) - (vert[0](1)-vert[3](1)) * (pos(0) - vert[3](0));

            if((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0))
                return false;
            return true;
        }

        void calculate_local_pts(const polyfem::ElementBases& gbase, 
        const int elem_idx,
        const RowVectorNd& pos, 
        Eigen::MatrixXd& local_pos)
        {
            local_pos = Eigen::MatrixXd::Zero(1, dim);
            
            std::vector<RowVectorNd> vert(shape,RowVectorNd::Zero(1, dim));
            for (int i = 0; i < shape; i++)
            {
                for(int d = 0; d < dim; d++)
                    vert[i](d) = V(T(elem_idx, i), d);
            }
            // if(shape == 4 && dim == 2 && outside_quad(vert, pos))
            // {
            //     local_pos(0) = local_pos(1) = -1;
            //     return;
            // }
            Eigen::MatrixXd res;
            int iter_times = 0;
            int max_iter = 20;
            do
            {
                res = -pos;
                ElementAssemblyValues gvals_;
                gvals_.compute(elem_idx, dim == 3, local_pos, gbase, gbase);
                for (int i = 0; i < shape; i++)
                {
                    res += vert[i] * gvals_.basis_values[i].val(0);
                }

                Eigen::MatrixXd jacobi = Eigen::MatrixXd::Zero(dim, dim);
                for (int d1 = 0; d1 < dim; d1++)
                {
                    for (int d2 = 0; d2 < dim; d2++)
                    {
                        for (int i = 0; i < shape; i++)
                        {
                            jacobi(d1, d2) += vert[i](d1) * gvals_.basis_values[i].grad(0, d2);
                        }
                    }
                }

                Eigen::VectorXd delta = jacobi.colPivHouseholderQr().solve(res.transpose());
                for (int d = 0; d < dim; d++)
                {
                    local_pos(d) -= delta(d);
                }
                iter_times++;
            }
            while(res.norm() > 1e-12 && iter_times < max_iter);

            if(iter_times >= max_iter)
            {
                for(int d=0; d<dim; d++)
                    local_pos(d) = -1;
            }
        }

        void solve_stokes_1st(const std::string &solver_type, 
        const std::string &precond, 
        const json& params,
        const StiffnessMatrix& mass,
        const StiffnessMatrix& stiffness,
        const double& dt,
        const double& viscosity_,
        const std::string &save_path,
	    bool compute_spectrum,
        Eigen::MatrixXd& sol, 
        Eigen::MatrixXd& pressure,
        const int& n_pressure_bases)
        {
            auto solver = LinearSolver::create(solver_type, precond);
            solver->setParameters(params);
            logger().info("{}...", solver->name());

            StiffnessMatrix A;
            Eigen::VectorXd b, x;

            A = dt * stiffness;
            A.block(0, 0, sol.rows(), sol.rows()) *= viscosity_;
            A += mass;
            b = Eigen::VectorXd::Zero(sol.rows() + n_pressure_bases + 1);
            b.block(0, 0, sol.rows(), sol.cols()) = mass.block(0, 0, sol.rows(), sol.rows()) * sol;

            for (int i = 0; i < boundary_nodes.size(); i++)
            {
                b(boundary_nodes[i]) = sol(boundary_nodes[i]);
            }

            const int precond_num = sol.rows();

            auto spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, save_path, compute_spectrum);
            sol = x.block(0, 0, sol.rows(), sol.cols());
            pressure = x.block(sol.rows(), 0, n_pressure_bases, sol.cols());
        }

        void solve_stokes_2nd(const std::string &solver_type, 
        const std::string &precond, 
        const json& params,
        const StiffnessMatrix& mass,
        const StiffnessMatrix& stiffness,
        const double& dt,
        const double& viscosity_,
        const std::string &save_path,
	    bool compute_spectrum,
        Eigen::MatrixXd& sol, 
        Eigen::MatrixXd& pressure,
        const int& n_pressure_bases)
        {
            auto solver = LinearSolver::create(solver_type, precond);
            solver->setParameters(params);
            logger().info("{}...", solver->name());

            StiffnessMatrix A;
            Eigen::VectorXd b, x;

            A = dt / 2 * stiffness;
            A.block(0,0,sol.rows(),sol.rows()) *= 0;
            A += mass;
            b = Eigen::VectorXd::Zero(sol.rows() + n_pressure_bases + 1);
            b.block(0, 0, sol.rows(), sol.cols()) = mass.block(0, 0, sol.rows(), sol.rows()) * sol - dt / 2 * viscosity_ * stiffness.block(0, 0, sol.rows(), sol.rows()) * sol;

            for (int i = 0; i < boundary_nodes.size(); i++)
            {
                b(boundary_nodes[i]) = sol(boundary_nodes[i]);
            }

            const int precond_num = sol.rows();

            auto spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, save_path, compute_spectrum);
            sol = x.block(0, 0, sol.rows(), sol.cols());

            A = dt / 2 * stiffness;
            A.block(0, 0, sol.rows(), sol.rows()) *= viscosity_;
            A += mass;
            b = Eigen::VectorXd::Zero(sol.rows() + n_pressure_bases + 1);
            b.block(0, 0, sol.rows(), sol.cols()) = mass.block(0, 0, sol.rows(), sol.rows()) * sol;

            for (int i = 0; i < boundary_nodes.size(); i++)
            {
                b(boundary_nodes[i]) = sol(boundary_nodes[i]);
            }

            spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, save_path, compute_spectrum);
            sol = x.block(0, 0, sol.rows(), sol.cols());
            pressure = x.block(sol.rows(), 0, n_pressure_bases, sol.cols());
        }

        int dim;
        int n_el;
        int shape;

        RowVectorNd min_domain;
        RowVectorNd max_domain;

        Eigen::MatrixXd V;
        Eigen::MatrixXi T;

        std::vector<std::list<int>> hash_table;
        int                         cell_num;

        std::vector<RowVectorNd> position_particle;
		std::vector<RowVectorNd> velocity_particle;
        std::vector<int> cellI_particle;
        Eigen::MatrixXd new_sol;
        Eigen::MatrixXd new_sol_w;

        std::vector<int> boundary_elem_id;
        std::vector<int> boundary_nodes;
    };
}