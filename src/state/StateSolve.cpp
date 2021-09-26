#include <polyfem/State.hpp>

#include <polyfem/BDF.hpp>
#include <polyfem/AdamsBashforth.hpp>
#include <polyfem/AdamsMoulton.hpp>
#include <polyfem/TransientNavierStokesSolver.hpp>
#include <polyfem/OperatorSplittingSolver.hpp>
#include <polyfem/NavierStokesSolver.hpp>

#include <polyfem/NLProblem.hpp>
#include <polyfem/ALNLProblem.hpp>

#include <polyfem/LbfgsSolver.hpp>
#include <polyfem/SparseNewtonDescentSolver.hpp>

#include <polysolve/LinearSolver.hpp>
#include <polysolve/FEMSolver.hpp>

#include <polyfem/StringUtils.hpp>

#include <polyfem/auto_p_bases.hpp>
#include <polyfem/auto_q_bases.hpp>

#include <ipc/ipc.hpp>

#include <fstream>
#include <tbb/concurrent_vector.h>

namespace polyfem
{
    namespace
    {
        void import_matrix(const std::string &path, const json &import, Eigen::MatrixXd &mat)
        {
            if (import.contains("offset"))
            {
                const int offset = import["offset"];

                Eigen::MatrixXd tmp;
                read_matrix_binary(path, tmp);
                mat.block(0, 0, offset, 1) = tmp.block(0, 0, offset, 1);
            }
            else
            {
                read_matrix_binary(path, mat);
            }
        }
    } // namespace

    void State::solve_transient_navier_stokes_split(const int time_steps, const double dt, const RhsAssembler &rhs_assembler)
    {
        assert(formulation() == "OperatorSplitting" && problem->is_time_dependent());
        const json &params = solver_params();
        const int dim = mesh->dimension();
        const int n_el = int(bases.size());       // number of elements
        const auto &gbases = iso_parametric() ? bases : geom_bases;
        const int shape = gbases[0].bases.size(); // number of geometry vertices in an element
        const double viscosity_ = build_json_params()["viscosity"];
        const int BDF_order = args["BDF_order"];

		std::vector<int> bnd_nodes;
		bnd_nodes.reserve(boundary_nodes.size() / mesh->dimension());
		for (auto node : boundary_nodes)
		{
			if (!(node % mesh->dimension()))
				continue;
			bnd_nodes.push_back(node / mesh->dimension());
		}

		Eigen::MatrixXd local_pts;
		if (mesh->dimension() == 2)
		{
			if (gbases[0].bases.size() == 3)
				autogen::p_nodes_2d(args["discr_order"], local_pts);
			else
				autogen::q_nodes_2d(args["discr_order"], local_pts);
		}
		else
		{
			if (gbases[0].bases.size() == 4)
				autogen::p_nodes_3d(args["discr_order"], local_pts);
			else
				autogen::q_nodes_3d(args["discr_order"], local_pts);
		}

        if (BDF_order == 1) {

            logger().info("Matrices assembly...");
            StiffnessMatrix stiffness_viscosity, mixed_stiffness, velocity_mass;
            // coefficient matrix of viscosity
            assembler.assemble_problem("Laplacian", mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, stiffness_viscosity);
            assembler.assemble_mass_matrix("Laplacian", mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, mass);

            // coefficient matrix of pressure projection
            assembler.assemble_problem("Laplacian", mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, stiffness);

            // matrix used to calculate divergence of velocity
            assembler.assemble_mixed_problem("Stokes", mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
            assembler.assemble_mass_matrix("Stokes", mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, velocity_mass);
            mixed_stiffness = mixed_stiffness.transpose();
            logger().info("Matrices assembly ends!");

            OperatorSplittingSolver ss;
            ss.initialize_mesh(*mesh, shape, n_el, local_boundary);
            ss.initialize_hashtable(*mesh);
            ss.initialize_linear_solver(args["solver_type"], args["precond_type"], params);

            /* initialize solution */
            pressure = Eigen::MatrixXd::Zero(n_pressure_bases, 1);

            Eigen::VectorXd integrals;
            getPressureIntegral(integrals);

            rhs.resize(n_bases * dim, 1);
            Eigen::MatrixXd current_rhs = rhs;

            for (int t = 1; t <= time_steps; t++)
            {
                double time = t * dt;
                logger().info("{}/{} steps, t={}s", t, time_steps, time);

                /* advection */
                logger().info("Advection...");
                const int RK_order = args["RK"];
                if (args["particle"])
                    ss.advection_FLIP(*mesh, gbases, bases, sol, dt, local_pts);
                else {
                    ss.advection(*mesh, gbases, bases, sol, dt, local_pts, RK_order);
                    // {
                    //     auto solx = sol;
                    //     ss.advection(*mesh, gbases, bases, sol, dt, local_pts, RK_order);
                    //     save_vtu(resolve_output_path(fmt::format("advect1_{:d}.vtu", t)), time);
                    //     auto soly = sol;
                    //     ss.advection(*mesh, gbases, bases, sol, -dt, local_pts, RK_order);
                    //     save_vtu(resolve_output_path(fmt::format("advect2_{:d}.vtu", t)), time);
                    //     auto solz = sol;
                    //     sol = soly + 0.5 * (solx - solz);
                    //     save_vtu(resolve_output_path(fmt::format("advect3_{:d}.vtu", t)), time);
                    // }
                }
                logger().info("Advection finished!");

                /* apply boundary condition */
                Eigen::MatrixXd bc(sol.rows(), sol.cols());
                bc.setZero();
                current_rhs.setZero();
                rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
                rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, bc, time);

                /* viscosity */
                logger().info("Solving diffusion...");
                if (viscosity_ > 0)
                    ss.solve_diffusion_1st(mass, stiffness_viscosity, bnd_nodes, bc, current_rhs, sol, dt, viscosity_);
                logger().info("Diffusion solved!");

                /* incompressibility */
                logger().info("Pressure projection...");
                ss.solve_pressure(stiffness, mixed_stiffness, integrals, pressure_dirichlet_boundary_nodes, sol, pressure);

                ss.projection(gbases, bases, pressure_bases, local_pts, pressure, sol);
                // ss.projection(velocity_mass, mixed_stiffness, boundary_nodes, sol, pressure);
                logger().info("Pressure projection finished!");

                pressure = pressure / dt;

                /* apply boundary condition */
                rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, time);

                // check nan
                for (int i = 0; i < sol.size(); i++) {
                    if (!std::isfinite(sol(i))) {
                        logger().error("NAN Detected!!");
                        return;
                    }
                }

                /* export to vtu */
                if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
                {
                    if (!solve_export_to_file)
                        solution_frames.emplace_back();
                    save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
                    // save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
                }
            }
        }
        else {

			std::vector<int> pressure_bnd_nodes;
			pressure_bnd_nodes.reserve(pressure_boundary_nodes.size() / mesh->dimension());
			for (auto node : pressure_boundary_nodes)
			{
				if (!(node % mesh->dimension()))
					continue;
				pressure_bnd_nodes.push_back(node / mesh->dimension());
			}

			std::vector<bool> pressure_boundary_nodes_mask;
			pressure_boundary_nodes_mask.assign(n_bases, false);
			for (auto node : pressure_bnd_nodes)
				pressure_boundary_nodes_mask[node] = true;
            // coefficient matrix of viscosity
            assembler.assemble_problem("Stokes", mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, stiffness);
            // assembler.assemble_mixed_problem("Stokes", mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
            assembler.assemble_mass_matrix("Stokes", mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, mass);

            // coefficient matrix of pressure projection
			StiffnessMatrix pressure_stiffness;
            assembler.assemble_problem("Laplacian", mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

			auto set_exact_solution = [&](const double t, Eigen::MatrixXd& sol_c) -> void {
				sol_c.resize(dim*n_bases, 1);
				sol_c.setZero();

				std::vector<bool> flag(n_bases, false);
#ifdef POLYFEM_WITH_TBB
    			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
    			for (int e = 0; e < n_el; e++)
#endif
                {
					ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), local_pts, bases[e], gbases[e]);

					Eigen::MatrixXd vel_exact;
                    problem->exact(vals.val, t, vel_exact);

                    const int n_loc_bases = int(vals.basis_values.size());
                    for (int i = 0; i < n_loc_bases; ++i) {
						const auto &val = vals.basis_values[i];
						assert(val.global.size() == 1);
						if (!flag[val.global[0].index]) {
							flag[val.global[0].index] = true;
							for (int d = 0; d < dim; d++) {
								sol_c(val.global[0].index * dim + d) = vel_exact(i, d);
							}
						}
                    }
                }
#ifdef POLYFEM_WITH_TBB
    			);
#endif
			};
// 			auto set_exact_force = [&](const double t, Eigen::MatrixXd& F) -> void {
// 				F.resize(dim*n_bases, 1);
// 				F.setZero();

// 				std::vector<bool> flag(n_bases, false);
// #ifdef POLYFEM_WITH_TBB
//     			tbb::parallel_for(0, n_el, 1, [&](int e)
// #else
//     			for (int e = 0; e < n_el; e++)
// #endif
//                 {
// 					ElementAssemblyValues vals;
// 					vals.compute(e, mesh->is_volume(), local_pts, bases[e], gbases[e]);

// 					AssemblerUtils assembler;
// 					Eigen::MatrixXd force;
// 					problem->rhs(assembler, formulation(), local_pts, t, force);

//                     const int n_loc_bases = int(vals.basis_values.size());
//                     for (int i = 0; i < n_loc_bases; ++i) {
// 						const auto &val = vals.basis_values[i];
// 						assert(val.global.size() == 1);
// 						if (!flag[val.global[0].index]) {
// 							flag[val.global[0].index] = true;
// 							for (int d = 0; d < dim; d++) {
// 								F(val.global[0].index * dim + d) = force(i, d);
// 							}
// 						}
//                     }
//                 }
// #ifdef POLYFEM_WITH_TBB
//     			);
// #endif
// 			};
			auto set_exact_pressure = [&](const double t, Eigen::MatrixXd& pressure_c) -> void {
				pressure_c.resize(n_pressure_bases, 1);
				pressure_c.setZero();

				std::vector<bool> flag(n_pressure_bases, false);
#ifdef POLYFEM_WITH_TBB
    			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
    			for (int e = 0; e < n_el; e++)
#endif
				{
					ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), local_pts, pressure_bases[e], gbases[e]);

					Eigen::MatrixXd p_exact;
                    problem->exact_pressure(vals.val, t, p_exact);

                    const int n_loc_pressure_bases = int(vals.basis_values.size());
                    for (int i = 0; i < n_loc_pressure_bases; ++i) {
                        const auto &val = vals.basis_values[i];
                        assert(val.global.size() == 1);
						if (!flag[val.global[0].index]) {
							flag[val.global[0].index] = true;
							pressure_c(val.global[0].index) = p_exact(i);
						}
                    }
                }
#ifdef POLYFEM_WITH_TBB
    			);
#endif
			};

			// initialization
			set_exact_solution(0., sol);
			set_exact_pressure(0., pressure);
			// pressure.resize(n_pressure_bases, 1);
			// pressure.setZero();

			auto assemble_rhs_pressure_1 = [&](const Eigen::MatrixXd& sol_c, const double alpha, const double time, Eigen::VectorXd& rhs_) -> void {
				tbb::concurrent_vector<Eigen::Triplet<double> > nonzeros;
#ifdef POLYFEM_WITH_TBB
    			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
    			for (int e = 0; e < n_el; e++)
#endif
				{
					ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), pressure_bases[e], gbases[e]);

					const Eigen::VectorXd da = vals.det.array() * vals.quadrature.weights.array();
					Eigen::MatrixXd quadrature_points = vals.quadrature.points;

					Eigen::MatrixXd vel, vel_grad;
					interpolate_at_local_vals(e, dim, bases, quadrature_points, sol_c, vel, vel_grad);

					Eigen::VectorXd gradU_gradUT(vel_grad.rows());
					gradU_gradUT.setZero();
					for (int d1 = 0; d1 < dim; d1++)
						for (int d2 = 0; d2 < dim; d2++)
							for (int j = 0; j < vel_grad.rows(); j++)
								gradU_gradUT(j) += vel_grad(j, d1 * dim + d2) * vel_grad(j, d2 * dim + d1);
					
					Eigen::MatrixXd vel_div(vel_grad.rows(), 1);
					vel_div.setZero();
					for (int d = 0; d < dim; d++)
						vel_div += vel_grad.col(d * dim + d);

					Eigen::MatrixXd final_vec = gradU_gradUT - alpha * vel_div;// - f_div;
					final_vec = final_vec.array() * da.array();

					const int n_loc_pressure_bases = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_pressure_bases; ++i) {
						const auto &val = vals.basis_values[i];
						assert(val.global.size() == 1);
						nonzeros.push_back(Eigen::Triplet<double>(0, val.global[0].index, (final_vec.array() * val.val.array()).sum() * val.global[0].val));
					}
				}
#ifdef POLYFEM_WITH_TBB
    			);
#endif
                rhs_.resize(n_pressure_bases, 1);
                rhs_.setZero();
				for (auto i = begin(nonzeros); i != end(nonzeros); i++) {
					rhs_(i->col()) += i->value();
				}
            };
            auto assemble_rhs_pressure_2 = [&](const Eigen::MatrixXd& sol_c, const double alpha, const double time, Eigen::VectorXd& rhs_) -> void {
                ElementAssemblyValues vals;
                rhs_.resize(n_pressure_bases, 1);
                rhs_.setZero();
                for (int e = 0; e < n_el; e++) {
                    vals.compute(e, mesh->is_volume(), pressure_bases[e], gbases[e]);

                    const Eigen::VectorXd da = vals.det.array() * vals.quadrature.weights.array();
                    Eigen::MatrixXd quadrature_points = vals.quadrature.points;

                    Eigen::MatrixXd vel, vel_grad;
                    interpolate_at_local_vals(e, dim, bases, quadrature_points, sol_c, vel, vel_grad);

                    Eigen::MatrixXd U_gradU(vel_grad.rows(), vel_grad.cols());
                    U_gradU.setZero();
                    for (int d1 = 0; d1 < dim; d1++) {
						for (int d2 = 0; d2 < dim; d2++) {
							U_gradU.col(d1) += (vel.col(d2).array() * vel_grad.col(d1 * dim + d2).array()).matrix();
						}
					}

                    Eigen::MatrixXd vel_div(vel_grad.rows(), 1);
                    vel_div.setZero();
                    for (int d = 0; d < dim; d++)
                        vel_div += vel_grad.col(d * dim + d);

					// Eigen::MatrixXd force;
					// set_exact_force(time, force);
                    // Eigen::MatrixXd f, f_grad;
                    // interpolate_at_local_vals(e, dim, bases, quadrature_points, force, f, f_grad);

					Eigen::MatrixXd final_mat = - U_gradU; // + f
					for (int d = 0; d < dim; d++)
						final_mat.col(d) = final_mat.col(d).array() * da.array();

                    const int n_loc_pressure_bases = int(vals.basis_values.size());
                    for (int i = 0; i < n_loc_pressure_bases; ++i) {
                        const auto &val = vals.basis_values[i];
                        assert(val.global.size() == 1);
						for (int d = 0; d < dim; d++)
							rhs_(val.global[0].index) += (final_mat.col(d).array() * val.grad_t_m.col(d).array()).sum() * val.global[0].val;
						rhs_(val.global[0].index) -= alpha * (vel_div.array() * val.val.array()).sum() * val.global[0].val;
                    }
                }
            };

			auto assemble_velocity_rhs = [&](const Eigen::MatrixXd& sol_c, const Eigen::MatrixXd& pressure_c, const double time, Eigen::MatrixXd& v_rhs) -> void {
				tbb::concurrent_vector<Eigen::Triplet<double> > nonzeros;
#ifdef POLYFEM_WITH_TBB
    			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
    			for (int e = 0; e < n_el; e++)
#endif
                {
					ElementAssemblyValues vals;
                    vals.compute(e, mesh->is_volume(), bases[e], gbases[e]);

                    const Eigen::VectorXd da = vals.det.array() * vals.quadrature.weights.array();
                    Eigen::MatrixXd quadrature_points = vals.quadrature.points;

                    Eigen::MatrixXd vel, vel_grad;
                    interpolate_at_local_vals(e, dim, bases, quadrature_points, sol_c, vel, vel_grad);
                    
                    Eigen::MatrixXd pres, pres_grad;
                    interpolate_at_local_vals(e, 1, pressure_bases, quadrature_points, pressure_c, pres, pres_grad);

					Eigen::MatrixXd final_mat = -pres_grad;
					for (int d = 0; d < dim; ++d) {
						for (int d_ = 0; d_ < dim; ++d_)
							for (int j = 0; j < final_mat.rows(); j++)
								final_mat(j, d) -= vel(j, d_) * vel_grad(j, d * dim + d_);
						final_mat.col(d) = (final_mat.col(d).array() * da.array()).matrix();
					}

                    const int n_loc_bases = int(vals.basis_values.size());
                    for (int i = 0; i < n_loc_bases; ++i) {
                        const auto &val = vals.basis_values[i];
                        assert(val.global.size() == 1);

						for (int d = 0; d < dim; ++d)
							nonzeros.push_back(Eigen::Triplet<double>(0, val.global[0].index * dim + d, (final_mat.col(d).array() * val.val.array()).sum() * val.global[0].val));
                    }
                }
#ifdef POLYFEM_WITH_TBB
    			);
#endif
                v_rhs.resize(n_bases*dim, 1);
                v_rhs.setZero();
				for (auto i = begin(nonzeros); i != end(nonzeros); i++) {
					v_rhs(i->col()) += i->value();
				}
				v_rhs -= stiffness * sol_c;
				// Eigen::MatrixXd force_rhs(v_rhs.rows(), v_rhs.cols()); force_rhs.setZero();
				// rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, v_rhs, time, force_rhs);
				// v_rhs = v_rhs + force_rhs;
			};

			Eigen::MatrixXd node_normals(n_bases, dim);
			node_normals.setZero();
			{
				// compute boundary edge normals
				// todo: if velocity and pressure are not the same order
				Eigen::MatrixXd edge_normals(mesh->n_edges(), dim);
				edge_normals.setZero();

				assert(dim == 2);
				Mesh2D &mesh2d = *dynamic_cast<Mesh2D *>(mesh.get());
				for (const auto &lb : local_boundary)
				{
					const int e = lb.element_id();
					for (int i = 0; i < lb.size(); i++) {
						const int edge_id = lb.global_primitive_id(i);
						const int v1 = mesh2d.edge_vertex(edge_id, 0);
						const int v2 = mesh2d.edge_vertex(edge_id, 1);
						RowVectorNd edge = mesh2d.point(v2) - mesh2d.point(v1);
						const double norm = edge.norm();
						edge_normals(edge_id, 0) = edge(1) / norm;
						edge_normals(edge_id, 1) = -edge(0) / norm;

						// determine sign
						RowVectorNd mid_point = 0.5 * (mesh2d.point(v2) + mesh2d.point(v1));
						RowVectorNd third_point;
						for (int third_id = 0; third_id < 3; third_id++) {
							int third = mesh2d.cell_vertex(e, third_id);
							if (third == v1 || third == v2) continue;
							else third_point = mesh2d.point(third);
						}
						bool sign = (third_point - mid_point).dot(edge_normals.row(edge_id)) > 0;
						if (sign)
							edge_normals.row(edge_id) *= -1;
					}
				}

				Eigen::VectorXi traverse_times(n_bases); traverse_times.setZero();
				for (const auto &lb : local_boundary)
				{
					ElementAssemblyValues vals;
					const int e = lb.element_id();
					vals.compute(e, mesh->is_volume(), pressure_bases[e], gbases[e]);

					const int n_loc_pressure_bases = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_pressure_bases; i++) {
						const auto &val = vals.basis_values[i];
						if (!pressure_boundary_nodes_mask[val.global[0].index]) continue;
						int local_edge_id = -1;
						if (i < 3) {
							const int e1 = (i+2)%3;
							const int e2 = i;
							for (int j = 0; j < lb.size(); j++) {
								if (e1 == lb[j] || e2 == lb[j]) {
									local_edge_id = j;
									break;
								}
							}
						}
						else if (i < 3 + 3 * (disc_orders[e] - 1)) {
							const int e_ = (i - 3) / (int)(disc_orders[e] - 1);
							for (int j = 0; j < lb.size(); j++) {
								if (e_ == lb[j]) {
									local_edge_id = j;
									break;
								}
							}
						}
						else
							continue;

						node_normals.row(val.global[0].index) += edge_normals.row(lb.global_primitive_id(local_edge_id));
						traverse_times(val.global[0].index)++;
					}
				}
				for (int i = 0; i < n_bases; i++) {
					if (traverse_times(i) > 1)
						node_normals.row(i) /= traverse_times(i);
				}
			}

			auto set_pressure_neumann_bc_1 = [&](const Eigen::MatrixXd& sol_c, const Eigen::MatrixXd& dsol_dt, const double time, Eigen::VectorXd& rhs_) -> void {
				tbb::concurrent_vector<Eigen::Triplet<double> > nonzeros;
#ifdef POLYFEM_WITH_TBB
    			tbb::parallel_for(0, n_el, 1, [&](int e)
#else
    			for (int e = 0; e < n_el; e++)
#endif
				{	
					ElementAssemblyValues vals;
					vals.compute(e, mesh->is_volume(), pressure_bases[e], gbases[e]);

					const Eigen::VectorXd da = vals.det.array() * vals.quadrature.weights.array();
					Eigen::MatrixXd quadrature_points = vals.quadrature.points;

					Eigen::MatrixXd vel, vel_grad;
					interpolate_at_local_vals(e, dim, bases, quadrature_points, sol_c, vel, vel_grad);

					Eigen::MatrixXd dvel_dt, dvel_dt_grad;
					interpolate_at_local_vals(e, dim, bases, quadrature_points, dsol_dt, dvel_dt, dvel_dt_grad);

					// Eigen::MatrixXd force;
					// AssemblerUtils assembler;
					// problem->rhs(assembler, formulation(), quadrature_points, time, force);

					Eigen::MatrixXd final_mat = dvel_dt;
					for (int d = 0; d < dim; ++d)
						for (int d_ = 0; d_ < dim; ++d_)
							final_mat.col(d) += (vel.col(d_).array() * vel_grad.col(d * dim + d_).array()).matrix();

					// final_mat = final_mat - force;
					for (int d = 0; d < dim; d++)
						for (int j = 0; j < final_mat.rows(); j++)
							final_mat(j, d) *= -da(j);

					const int n_loc_pressure_bases = int(vals.basis_values.size());
					Eigen::MatrixXd curl_u = (vel_grad.col(1*dim+0) - vel_grad.col(0*dim+1)).cwiseProduct(da);
					for (int i = 0; i < n_loc_pressure_bases; ++i) {
						const auto &val = vals.basis_values[i];
						if (!pressure_boundary_nodes_mask[val.global[0].index]) continue;
						double value = 0;
						for (int d = 0; d < dim; d++) {
							value += (final_mat.col(d).array() * val.val.array()).sum() * node_normals(val.global[0].index, d) * val.global[0].val;
						}
						Eigen::VectorXd n_cross_gradp = node_normals(val.global[0].index, 0) * val.grad_t_m.col(1) - node_normals(val.global[0].index, 1) * val.grad_t_m.col(0);
						value += viscosity_ * (curl_u.array() * n_cross_gradp.array()).sum() * val.global[0].val;
						nonzeros.push_back(Eigen::Triplet<double>(0, val.global[0].index, value));
					}
				}
#ifdef POLYFEM_WITH_TBB
    			);
#endif
				for (auto i = begin(nonzeros); i != end(nonzeros); i++) {
					rhs_(i->col()) += i->value();
				}
			};

			auto set_pressure_neumann_bc_coef = [&](std::vector<Eigen::Triplet<double> >& coef) -> void {
				ElementAssemblyValues vals;
				for (int e = 0; e < n_el; e++) {
					if (iso_parametric())
						vals.compute(e, mesh->is_volume(), pressure_bases[e], bases[e]);
					else
						vals.compute(e, mesh->is_volume(), pressure_bases[e], geom_bases[e]);

					const Eigen::VectorXd da = vals.det.array() * vals.quadrature.weights.array();
					Eigen::MatrixXd quadrature_points = vals.quadrature.points;

					const int n_loc_pressure_bases = int(vals.basis_values.size());
					for (int i = 0; i < n_loc_pressure_bases; i++) {
						const auto &val = vals.basis_values[i];
						if (!pressure_boundary_nodes_mask[val.global[0].index]) continue;

						for (int j = 0; j < n_loc_pressure_bases; j++) {
							const auto &val_ = vals.basis_values[j];

							double tmp = 0;
							for (int d = 0; d < dim; d++) {
								tmp += (val.val.array() * val_.grad_t_m.col(d).array() * da.array()).sum() * node_normals(val.global[0].index, d);
							}
							coef.emplace_back(val.global[0].index, val_.global[0].index, tmp);
						}
					}
				}
			};

			auto set_pressure_neumann_bc_2 = [&](const Eigen::MatrixXd& sol_c, const Eigen::MatrixXd& dsol_dt, const double time, Eigen::VectorXd& rhs_) -> void {
				Eigen::MatrixXd points, normals, uv;
				Eigen::VectorXd weights;
				Eigen::VectorXi global_primitive_ids;

				ElementAssemblyValues vals;

				for (const auto &lb : local_boundary) {
					const int e = lb.element_id();
					bool has_samples = rhs_assembler.boundary_quadrature(lb, args["n_boundary_samples"], false, uv, points, normals, weights, global_primitive_ids);

					if (!has_samples)
						continue;

					const ElementBases &gbs = gbases[e];
					const ElementBases &pbs = pressure_bases[e];
					// const ElementBases &bs = bases[e];

					vals.compute(e, mesh->is_volume(), points, pbs, gbs);

					for (int n = 0; n < vals.jac_it.size(); ++n)
					{
						normals.row(n) = normals.row(n) * vals.jac_it[n];
						normals.row(n).normalize();
					}

					Eigen::MatrixXd vel, vel_grad;
					interpolate_at_local_vals(e, dim, bases, points, sol_c, vel, vel_grad);

					Eigen::MatrixXd dvel_dt, dvel_dt_grad;
					interpolate_at_local_vals(e, dim, bases, points, dsol_dt, dvel_dt, dvel_dt_grad);

					// Eigen::MatrixXd force;
					// AssemblerUtils assembler;
					// problem->rhs(assembler, formulation(), points, time, force);

					Eigen::MatrixXd final_mat = -dvel_dt;
					for (int d = 0; d < dim; ++d)
						for (int d_ = 0; d_ < dim; ++d_)
							final_mat.col(d) -= (vel.col(d_).array() * vel_grad.col(d * dim + d_).array()).matrix();
					// final_mat = force + final_mat;

					Eigen::VectorXd final_vec(final_mat.rows()); final_vec.setZero();
					for (int d = 0; d < dim; d++)
						final_vec += (final_mat.col(d).array() * normals.col(d).array()).matrix();
					final_vec = final_vec.array() * weights.array();

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = pbs.local_nodes_for_primitive(primitive_global_id, *mesh);

						for (long n = 0; n < nodes.size(); ++n)
						{
							const AssemblyValues &v = vals.basis_values[nodes(n)];

							assert(v.global.size() == 1);
							rhs_(v.global[0].index) += (final_vec.array() * v.val.array()).sum() * v.global[0].val;
						}
					}

					Eigen::VectorXd curl_u = vel_grad.col(1*dim+0) - vel_grad.col(0*dim+1);

					for (int i = 0; i < lb.size(); ++i)
					{
						const int primitive_global_id = lb.global_primitive_id(i);
						const auto nodes = pbs.local_nodes_for_primitive(primitive_global_id, *mesh);

						for (long n = 0; n < nodes.size(); ++n)
						{
							const AssemblyValues &v = vals.basis_values[nodes(n)];

							assert(v.global.size() == 1);
							Eigen::VectorXd n_cross_gradp = normals.col(0).array() * v.grad_t_m.col(1).array() - normals.col(1).array() * v.grad_t_m.col(0).array();
							rhs_(v.global[0].index) += viscosity_ * (curl_u.array() * n_cross_gradp.array() * weights.array()).sum() * v.global[0].val;
						}
					}
				}
			};

            std::unique_ptr<polysolve::LinearSolver> solver1 = LinearSolver::create(args["solver_type"], args["precond_type"]);
            solver1->setParameters(params);
            {
                auto A = mass;
                prefactorize(*solver1, A, boundary_nodes, A.rows());
            }

            std::unique_ptr<polysolve::LinearSolver> solver2 = LinearSolver::create(args["solver_type"], args["precond_type"]);
            solver2->setParameters(params);
            {
                Eigen::VectorXd integrals;
                getPressureIntegral(integrals);

                std::vector<Eigen::Triplet<double> > coefficients;
                for(int i = 0; i < pressure_stiffness.outerSize(); i++)
                    for(StiffnessMatrix::InnerIterator it(pressure_stiffness,i); it; ++it)
						if (!args["WABE"].get<bool>() || !pressure_boundary_nodes_mask[it.row()])
                        	coefficients.emplace_back(it.row(),it.col(),it.value());
				
				if (args["WABE"].get<bool>())
					set_pressure_neumann_bc_coef(coefficients);

                for (int i = 0; i < pressure_stiffness.rows(); i++)
                {
                    coefficients.emplace_back(i, pressure_stiffness.rows(), integrals[i]);
                    coefficients.emplace_back(pressure_stiffness.rows(), i, integrals[i]);
                }

                StiffnessMatrix pressure_stiffness_tmp;
                pressure_stiffness_tmp.resize(pressure_stiffness.rows()+1, pressure_stiffness.rows()+1);
                pressure_stiffness_tmp.setFromTriplets(coefficients.begin(), coefficients.end());
                pressure_stiffness = pressure_stiffness_tmp;

                prefactorize(*solver2, pressure_stiffness_tmp, std::vector<int>(), pressure_stiffness_tmp.rows());
            }

			auto solve_pressure = [&](const Eigen::MatrixXd& sol_c, const Eigen::MatrixXd& dsol_dt, const double time, Eigen::MatrixXd& pressure_c) -> void {
				if (pressure_c.size() != n_pressure_bases) {
					pressure_c.resize(n_pressure_bases, 1);
					pressure_c.setZero();
				}

				Eigen::VectorXd pressure_extended(n_pressure_bases+1, 1);
				pressure_extended.block(0, 0, n_pressure_bases, 1) = pressure_c;
				pressure_extended(n_pressure_bases) = 0;
				
				const double alpha = 1. / min_edge_length / min_edge_length;
				Eigen::VectorXd rhs_pressure(n_pressure_bases);
				rhs_pressure.setZero();

				// set_exact_pressure(time, pressure_c);
				// for (int i = 0; i < pressure_bnd_nodes.size(); i++)
					// rhs_pressure(pressure_bnd_nodes[i]) = pressure_c(pressure_bnd_nodes[i]);
				assemble_rhs_pressure_1(sol_c, alpha, time, rhs_pressure);
				if (args["WABE"].get<bool>()) {
					for (int i = 0; i < pressure_bnd_nodes.size(); i++)
						rhs_pressure(pressure_bnd_nodes[i]) = 0;
					set_pressure_neumann_bc_1(sol_c, dsol_dt, time, rhs_pressure);
				}
				else
					set_pressure_neumann_bc_2(sol_c, dsol_dt, time, rhs_pressure);
				
				rhs_pressure.conservativeResize(n_pressure_bases+1);
				rhs_pressure(n_pressure_bases) = 0;

				dirichlet_solve_prefactorized(*solver2, pressure_stiffness, rhs_pressure, std::vector<int>(), pressure_extended);
				pressure_c = pressure_extended.block(0, 0, n_pressure_bases, 1);
			};

			Eigen::MatrixXd sol_n_1, sol_n = sol;
			Eigen::MatrixXd rhs_n_1, rhs_n;
			rhs_n_1.resize(0, 0);
			assemble_velocity_rhs(sol, pressure, 0., rhs_n);
            for (int t = 1; t <= time_steps; t++)
            {
                double time = t * dt;
                logger().info("{}/{} steps, t={}s", t, time_steps, time);

				sol_n_1 = sol_n;
				sol_n = sol;
				if (t == 1) {
					set_exact_solution(time, sol);
					set_exact_pressure(time, pressure);
				}
				else {
					if (args.contains("only_solve_pressure") && args["only_solve_pressure"]) {
						set_exact_solution(time, sol);
						solve_pressure(sol, (sol - sol_n) / dt, time, pressure);
					}
					else if (args.contains("only_solve_velocity") && args["only_solve_velocity"]) {
						// velocity prediction
						if (rhs_n_1.size() != rhs_n.size())
							rhs = rhs_n * dt + mass * sol_n;
						else
							rhs = (1.5 * rhs_n - 0.5 * rhs_n_1) * dt + mass * sol_n;

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time);
						Eigen::VectorXd rhs_vec = rhs;

						Eigen::VectorXd sol_vec = sol;
						dirichlet_solve_prefactorized(*solver1, mass, rhs_vec, boundary_nodes, sol_vec);
						sol = sol_vec;

						// pressure update
						set_exact_pressure(time, pressure);

						// velocity correction
						Eigen::MatrixXd rhs_p;
						assemble_velocity_rhs(sol, pressure, time, rhs_p);
						rhs = mass * sol_n + (0.5*dt) * (rhs_p + rhs_n);

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time);
						rhs_vec = rhs;
						
						sol_vec = sol;
						dirichlet_solve_prefactorized(*solver1, mass, rhs_vec, boundary_nodes, sol_vec);
						sol = sol_vec;
						
						// pressure correction
						set_exact_pressure(time, pressure);
					}
					else {
						// velocity prediction
						if (rhs_n_1.size() != rhs_n.size())
							rhs = rhs_n * dt + mass * sol_n;
						else
							rhs = (1.5 * rhs_n - 0.5 * rhs_n_1) * dt + mass * sol_n;

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time);
						Eigen::VectorXd rhs_vec = rhs;

						Eigen::VectorXd sol_vec = sol;
						dirichlet_solve_prefactorized(*solver1, mass, rhs_vec, boundary_nodes, sol_vec);
						sol = sol_vec;

						// pressure update
						solve_pressure(sol, (sol - sol_n) / dt, time, pressure);
						// solve_pressure(sol, (1.5*sol - 2*sol_n + 0.5*sol_n_1) / dt, time, pressure);
						// set_exact_pressure(time, pressure);

						// velocity correction
						Eigen::MatrixXd rhs_p;
						assemble_velocity_rhs(sol, pressure, time, rhs_p);
						rhs = mass * sol_n + (0.5*dt) * (rhs_p + rhs_n);

						rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs, time);
						rhs_vec = rhs;
						
						sol_vec = sol;
						dirichlet_solve_prefactorized(*solver1, mass, rhs_vec, boundary_nodes, sol_vec);
						sol = sol_vec;
						// set_exact_solution(time, sol);
						
						// pressure correction
						solve_pressure(sol, (sol - sol_n) / dt, time, pressure);
						// solve_pressure(sol, (1.5*sol - 2*sol_n + 0.5*sol_n_1) / dt, time, pressure);
						// set_exact_pressure(time, pressure);
					}
				}

                // compute rhs
				rhs_n_1 = rhs_n;
				assemble_velocity_rhs(sol, pressure, time, rhs_n);

                // check nan
                for (int i = 0; i < sol.size(); i++) {
                    if (!std::isfinite(sol(i))) {
                        logger().error("NAN Detected!!");
                        return;
                    }
                }

                /* export to vtu */
                if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
                {
                    if (!solve_export_to_file)
                        solution_frames.emplace_back();
                    save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
                    // save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
                }

				if (args.contains("compute_error_time") && args["compute_error_time"])
					compute_errors(time);
            }

        }

        save_pvd(
            resolve_output_path("sim.pvd"),
            [](int i)
            { return fmt::format("step_{:d}.vtu", i); },
            time_steps, /*t0=*/0, dt);

        const bool export_surface = args["export"]["surface"];

        if (export_surface)
        {
            save_pvd(
                resolve_output_path("sim_surf.pvd"),
                [](int i)
                { return fmt::format("step_{:d}_surf.vtu", i); },
                time_steps, /*t0=*/0, dt);
        }
    }

    void State::solve_transient_navier_stokes(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler, Eigen::VectorXd &c_sol)
    {
        assert(formulation() == "NavierStokes" && problem->is_time_dependent());

        const auto &gbases = iso_parametric() ? bases : geom_bases;
        Eigen::MatrixXd current_rhs = rhs;

        StiffnessMatrix velocity_mass;
        assembler.assemble_mass_matrix(formulation(), mesh->is_volume(), n_bases, density, bases, gbases, ass_vals_cache, velocity_mass);

        StiffnessMatrix velocity_stiffness, mixed_stiffness, pressure_stiffness;

        Eigen::VectorXd prev_sol;

        int BDF_order = args["BDF_order"];
        // int aux_steps = BDF_order-1;
        BDF bdf(BDF_order);
        bdf.new_solution(c_sol);

        assembler.assemble_problem(formulation(), mesh->is_volume(), n_bases, bases, gbases, ass_vals_cache, velocity_stiffness);
        assembler.assemble_mixed_problem(formulation(), mesh->is_volume(), n_pressure_bases, n_bases, pressure_bases, bases, gbases, pressure_ass_vals_cache, ass_vals_cache, mixed_stiffness);
        assembler.assemble_pressure_problem(formulation(), mesh->is_volume(), n_pressure_bases, pressure_bases, gbases, pressure_ass_vals_cache, pressure_stiffness);

        TransientNavierStokesSolver ns_solver(solver_params(), build_json_params(), solver_type(), precond_type());
        const int n_larger = n_pressure_bases + (use_avg_pressure ? 1 : 0);

        for (int t = 1; t <= time_steps; ++t)
        {
            double time = t0 + t * dt;
            double current_dt = dt;

            logger().info("{}/{} steps, dt={}s t={}s", t, time_steps, current_dt, time);

            bdf.rhs(prev_sol);
            rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
            rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

            const int prev_size = current_rhs.size();
            if (prev_size != rhs.size())
            {
                current_rhs.conservativeResize(prev_size + n_larger, current_rhs.cols());
                current_rhs.block(prev_size, 0, n_larger, current_rhs.cols()).setZero();
            }

            ns_solver.minimize(*this, bdf.alpha(), current_dt, prev_sol,
                               velocity_stiffness, mixed_stiffness, pressure_stiffness,
                               velocity_mass, current_rhs, c_sol);
            bdf.new_solution(c_sol);
            sol = c_sol;
            sol_to_pressure();

            if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
            {
                if (!solve_export_to_file)
                    solution_frames.emplace_back();
                save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
                save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
            }
        }

        save_pvd(
            resolve_output_path("sim.pvd"),
            [](int i)
            { return fmt::format("step_{:d}.vtu", i); },
            time_steps, t0, dt);

        const bool export_surface = args["export"]["surface"];

        if (export_surface)
        {
            save_pvd(
                resolve_output_path("sim_surf.pvd"),
                [](int i)
                { return fmt::format("step_{:d}_surf.vtu", i); },
                time_steps, t0, dt);
        }
    }

    void State::solve_transient_scalar(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler, Eigen::VectorXd &x)
    {
        assert((problem->is_scalar() || assembler.is_mixed(formulation())) && problem->is_time_dependent());

        const json &params = solver_params();
        auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
        solver->setParameters(params);
        logger().info("{}...", solver->name());

        StiffnessMatrix A;
        Eigen::VectorXd b;
        Eigen::MatrixXd current_rhs = rhs;

        const int BDF_order = args["BDF_order"];
        // const int aux_steps = BDF_order-1;
        BDF bdf(BDF_order);
        bdf.new_solution(x);

        const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
        const int precond_num = problem_dim * n_bases;

        for (int t = 1; t <= time_steps; ++t)
        {
            double time = t0 + t * dt;
            double current_dt = dt;

            logger().info("{}/{} {}s", t, time_steps, time);
            rhs_assembler.compute_energy_grad(local_boundary, boundary_nodes, density, args["n_boundary_samples"], local_neumann_boundary, rhs, time, current_rhs);
            rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, current_rhs, time);

            if (assembler.is_mixed(formulation()))
            {
                //divergence free
                int fluid_offset = use_avg_pressure ? (assembler.is_fluid(formulation()) ? 1 : 0) : 0;
                current_rhs.block(current_rhs.rows() - n_pressure_bases - use_avg_pressure, 0, n_pressure_bases + use_avg_pressure, current_rhs.cols()).setZero();
            }

            A = (bdf.alpha() / current_dt) * mass + stiffness;
            bdf.rhs(x);
            b = (mass * x) / current_dt;
            for (int i : boundary_nodes)
                b[i] = 0;
            b += current_rhs;

            spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == time_steps && args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
            bdf.new_solution(x);
            sol = x;

            if (assembler.is_mixed(formulation()))
            {
                sol_to_pressure();
            }

            if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
            {
                if (!solve_export_to_file)
                    solution_frames.emplace_back();

                save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), time);
                save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
            }
        }

        save_pvd(
            resolve_output_path("sim.pvd"),
            [](int i)
            { return fmt::format("step_{:d}.vtu", i); },
            time_steps, t0, dt);
    }

    void State::solve_transient_tensor_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
    {
        assert(!problem->is_scalar() && assembler.is_linear(formulation()) && !args["has_collision"] && problem->is_time_dependent());
        assert(!assembler.is_mixed(formulation()));

        const json &params = solver_params();
        auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
        solver->setParameters(params);
        logger().info("{}...", solver->name());

        const std::string v_path = resolve_path(args["import"]["v_path"], args["root_path"]);
        const std::string a_path = resolve_path(args["import"]["a_path"], args["root_path"]);

        Eigen::MatrixXd velocity, acceleration;

        if (!v_path.empty())
            import_matrix(v_path, args["import"], velocity);
        else
            rhs_assembler.initial_velocity(velocity);
        if (!a_path.empty())
            import_matrix(a_path, args["import"], acceleration);
        else
            rhs_assembler.initial_acceleration(acceleration);

        Eigen::MatrixXd current_rhs = rhs;

        const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
        const int precond_num = problem_dim * n_bases;

        //Newmark
        const double gamma = 0.5;
        const double beta = 0.25;
        // makes the algorithm implicit and equivalent to the trapezoidal rule (unconditionally stable).

        Eigen::MatrixXd temp, b;
        StiffnessMatrix A;
        Eigen::VectorXd x, btmp;

        for (int t = 1; t <= time_steps; ++t)
        {
            const double dt2 = dt * dt;

            const Eigen::MatrixXd aOld = acceleration;
            const Eigen::MatrixXd vOld = velocity;
            const Eigen::MatrixXd uOld = sol;

            rhs_assembler.assemble(density, current_rhs, t0 + dt * t);
            current_rhs *= -1;

            temp = -(uOld + dt * vOld + ((1 / 2. - beta) * dt2) * aOld);
            b = stiffness * temp + current_rhs;

            rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, b, t0 + dt * t);

            A = stiffness * beta * dt2 + mass;
            btmp = b;
            spectrum = dirichlet_solve(*solver, A, btmp, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], t == 1 && args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
            acceleration = x;

            sol += dt * vOld + dt2 * ((1 / 2.0 - beta) * aOld + beta * acceleration);
            velocity += dt * ((1 - gamma) * aOld + gamma * acceleration);

            rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, t0 + dt * t);
            rhs_assembler.set_velocity_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, velocity, t0 + dt * t);
            rhs_assembler.set_acceleration_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, acceleration, t0 + dt * t);

            if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
            {
                if (!solve_export_to_file)
                    solution_frames.emplace_back();
                save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
                save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
            }

            logger().info("{}/{} t={}", t, time_steps, t0 + dt * t);
        }

        {
            const std::string u_path = resolve_output_path(args["export"]["u_path"]);
            const std::string v_path = resolve_output_path(args["export"]["v_path"]);
            const std::string a_path = resolve_output_path(args["export"]["a_path"]);

            if (!u_path.empty())
                write_matrix_binary(u_path, sol);
            if (!v_path.empty())
                write_matrix_binary(v_path, velocity);
            if (!a_path.empty())
                write_matrix_binary(a_path, acceleration);
        }

        save_pvd(
            resolve_output_path("sim.pvd"),
            [](int i)
            { return fmt::format("step_{:d}.vtu", i); },
            time_steps, t0, dt);

        const bool export_surface = args["export"]["surface"];

        if (export_surface)
        {
            save_pvd(
                resolve_output_path("sim_surf.pvd"),
                [](int i)
                { return fmt::format("step_{:d}_surf.vtu", i); },
                time_steps, t0, dt);
        }
    }

	void State::solve_transient_tensor_non_linear(const int time_steps, const double t0, const double dt, const RhsAssembler &rhs_assembler)
	{
		assert(!problem->is_scalar() && (!assembler.is_linear(formulation()) || args["has_collision"]) && problem->is_time_dependent());
		assert(!assembler.is_mixed(formulation()));

		// FD for debug
		// {
		// 	Eigen::MatrixXd velocity, acceleration;
		// 	boundary_nodes.clear();
		// 	local_boundary.clear();
		// 	// local_neumann_boundary.clear();
		// 	NLProblem nl_problem(*this, rhs_assembler, t0, args["dhat"], false);
		// 	Eigen::MatrixXd tmp_sol = rhs;

		// 	// tmp_sol.setRandom();
		// 	tmp_sol.setZero();
		// 	// tmp_sol /=10000.;

		// 	velocity.setZero();
		// 	VectorXd xxx = tmp_sol;
		// 	velocity = tmp_sol;
		// 	velocity.setZero();
		// 	acceleration = tmp_sol;
		// 	acceleration.setZero();
		// 	nl_problem.init_time_integrator(xxx, velocity, acceleration, dt);

		// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
		// 	nl_problem.gradient(tmp_sol, actual_grad);

		// 	StiffnessMatrix hessian;
		// 	Eigen::MatrixXd expected_hessian;
		// 	nl_problem.hessian(tmp_sol, hessian);

		// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
		// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

		// 	for (int i = 0; i < actual_hessian.rows(); ++i)
		// 	{
		// 		double hhh = 1e-6;
		// 		VectorXd xp = tmp_sol;
		// 		xp(i) += hhh;
		// 		VectorXd xm = tmp_sol;
		// 		xm(i) -= hhh;

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
		// 		nl_problem.gradient(xp, tmp_grad_p);

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
		// 		nl_problem.gradient(xm, tmp_grad_m);

		// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m) / (hhh * 2.);

		// 		const double vp = nl_problem.value(xp);
		// 		const double vm = nl_problem.value(xm);

		// 		const double fd = (vp - vm) / (hhh * 2.);
		// 		const double diff = std::abs(actual_grad(i) - fd);
		// 		if (diff > 1e-6)
		// 			std::cout << "diff grad " << i << ": " << actual_grad(i) << " vs " << fd << " error: " << diff << " rrr: " << actual_grad(i) / fd << std::endl;

		// 		for (int j = 0; j < actual_hessian.rows(); ++j)
		// 		{
		// 			const double diff = std::abs(actual_hessian(i, j) - fd_h(j));

		// 			if (diff > 1e-5)
		// 				std::cout << "diff H " << i << ", " << j << ": " << actual_hessian(i, j) << " vs " << fd_h(j) << " error: " << diff << " rrr: " << actual_hessian(i, j) / fd_h(j) << std::endl;
		// 		}
		// 	}

		// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
		// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
		// 	exit(0);
		// }

		const std::string v_path = resolve_path(args["import"]["v_path"], args["root_path"]);
		const std::string a_path = resolve_path(args["import"]["a_path"], args["root_path"]);

		Eigen::MatrixXd velocity, acceleration;

		if (!v_path.empty())
			import_matrix(v_path, args["import"], velocity);
		else
			rhs_assembler.initial_velocity(velocity);
		if (!a_path.empty())
			import_matrix(a_path, args["import"], acceleration);
		else
			rhs_assembler.initial_acceleration(acceleration);

		if (args["has_collision"])
		{
			const int problem_dim = mesh->dimension();
			Eigen::MatrixXd tmp = boundary_nodes_pos;
			assert(tmp.rows() * problem_dim == sol.size());
			for (int i = 0; i < sol.size(); i += problem_dim)
			{
				for (int d = 0; d < problem_dim; ++d)
				{
					tmp(i / problem_dim, d) += sol(i + d);
				}
			}

			if (ipc::has_intersections(tmp, boundary_edges, boundary_triangles))
			{
				logger().error("Unable to solve, initial solution has intersections!");
				throw "Unable to solve, initial solution has intersections!";
			}
		}

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();
		VectorXd tmp_sol;

		NLProblem nl_problem(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"]);
		nl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		solver_info = json::array();

		// if (args["use_al"] || args["has_collision"])
		// {
		double al_weight = args["al_weight"];
		const double max_al_weight = args["max_al_weight"];
		ALNLProblem alnl_problem(*this, rhs_assembler, t0 + dt, args["dhat"], args["project_to_psd"], al_weight);
		alnl_problem.init_time_integrator(sol, velocity, acceleration, dt);

		for (int t = 1; t <= time_steps; ++t)
		{
			nl_problem.full_to_reduced(sol, tmp_sol);
			assert(sol.size() == rhs.size());
			assert(tmp_sol.size() < rhs.size());

			nl_problem.update_lagging(tmp_sol, /*start_of_timestep=*/true);
			alnl_problem.update_lagging(sol, /*start_of_timestep=*/true);

			if (args["friction_iterations"] > 0)
			{
				logger().debug("Lagging iteration 1");
			}

			nl_problem.line_search_begin(sol, tmp_sol);
			while (!std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
			{
				nl_problem.line_search_end();
				alnl_problem.set_weight(al_weight);
				logger().debug("Solving AL Problem with weight {}", al_weight);

				cppoptlib::SparseNewtonDescentSolver<ALNLProblem> alnlsolver(solver_params(), solver_type(), precond_type());
				alnlsolver.setLineSearch(args["line_search"]);
				alnl_problem.init(sol);
				tmp_sol = sol;
				alnlsolver.minimize(alnl_problem, tmp_sol);
				json alnl_solver_info;
				alnlsolver.getInfo(alnl_solver_info);

				solver_info.push_back({{"type", "al"},
									   {"t", t},
									   {"weight", al_weight},
									   {"info", alnl_solver_info}});

				sol = tmp_sol;
				nl_problem.full_to_reduced(sol, tmp_sol);
				nl_problem.line_search_begin(sol, tmp_sol);

				al_weight *= 2;

				if (al_weight >= max_al_weight)
				{
					logger().error("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
					break;
				}
			}
			nl_problem.line_search_end();
			al_weight = args["al_weight"];
			logger().debug("Solving Problem");

			cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
			nlsolver.setLineSearch(args["line_search"]);
			nl_problem.init(sol);
			nlsolver.minimize(nl_problem, tmp_sol);
			json nl_solver_info;
			nlsolver.getInfo(nl_solver_info);
			nl_problem.reduced_to_full(tmp_sol, sol);

			// Lagging loop (start at 1 because we already did an iteration above)
			int lag_i;
			for (lag_i = 1; lag_i < args["friction_iterations"] && !nl_problem.lagging_converged(tmp_sol, /*do_lagging_update=*/true); lag_i++)
			{
				logger().debug("Lagging iteration {:d}", lag_i + 1);
				nl_problem.init(sol);
				nlsolver.minimize(nl_problem, tmp_sol);
				json nl_solver_info;
				nlsolver.getInfo(nl_solver_info);
				nl_problem.reduced_to_full(tmp_sol, sol);
			}

			if (args["friction_iterations"] > 0)
			{
				logger().info(
					lag_i >= args["friction_iterations"]
						? "Maxed out at {:d} lagging iteration{}"
						: "Converged using {:d} lagging iteration{}",
					lag_i, lag_i > 1 ? "s" : "");
			}

			nl_problem.update_quantities(t0 + (t + 1) * dt, sol);
			alnl_problem.update_quantities(t0 + (t + 1) * dt, sol);

			if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(resolve_output_path(fmt::format("step_{:d}.vtu", t)), t0 + dt * t);
				save_wire(resolve_output_path(fmt::format("step_{:d}.obj", t)));
			}

			logger().info("{}/{}  t={}", t, time_steps, t0 + dt * t);

			solver_info.push_back({{"type", "rc"},
								   {"t", t},
								   {"info", nl_solver_info}});
		}
		// }
		// else
		// {
		// 	nl_problem.full_to_reduced(sol, tmp_sol);

		// 	for (int t = 1; t <= time_steps; ++t)
		// 	{
		// 		cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		// 		nlsolver.setLineSearch(args["line_search"]);
		// 		nl_problem.init(sol);
		// 		nlsolver.minimize(nl_problem, tmp_sol);

		// 		if (nlsolver.error_code() == -10)
		// 		{
		// 			double substep_delta = 0.5;
		// 			double substep = substep_delta;
		// 			bool solved = false;

		// 			while (substep_delta > 1e-4 && !solved)
		// 			{
		// 				logger().debug("Substepping {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
		// 				nl_problem.substepping((t - 1 + substep) * dt);
		// 				nl_problem.full_to_reduced(sol, tmp_sol);
		// 				nlsolver.minimize(nl_problem, tmp_sol);

		// 				if (nlsolver.error_code() == -10)
		// 				{
		// 					substep -= substep_delta;
		// 					substep_delta /= 2;
		// 				}
		// 				else
		// 				{
		// 					logger().trace("Done {}/{}, dt={}", (t - 1 + substep) * dt, t * dt, substep_delta);
		// 					nl_problem.reduced_to_full(tmp_sol, sol);
		// 					substep_delta *= 2;
		// 				}

		// 				solved = substep >= 1;

		// 				substep += substep_delta;
		// 				if (substep >= 1)
		// 				{
		// 					substep_delta -= substep - 1;
		// 					substep = 1;
		// 				}
		// 			}
		// 		}

		// 		if (nlsolver.error_code() == -10)
		// 		{
		// 			logger().error("Unable to solve t={}", t * dt);
		// 			save_vtu("stop.vtu", dt * t);
		// 			break;
		// 		}

		// 		logger().debug("Step solved!");

		// 		nlsolver.getInfo(solver_info);
		// 		nl_problem.reduced_to_full(tmp_sol, sol);
		// 		if (assembler.is_mixed(formulation()))
		// 		{
		// 			sol_to_pressure();
		// 		}

		// 		// rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, sol, dt * t);

		// 		nl_problem.update_quantities((t + 1) * dt, sol);

		// 		if (args["save_time_sequence"] && !(t % (int)args["skip_frame"]))
		// 		{
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();
		// 			save_vtu(fmt::format("step_{:d}.vtu", t), dt * t);
		// 			save_wire(fmt::format("step_{:d}.obj", t));
		// 		}

		// 		logger().info("{}/{}", t, time_steps);
		// 	}
		// }
		nl_problem.save_raw(
			resolve_output_path(args["export"]["u_path"]),
			resolve_output_path(args["export"]["v_path"]),
			resolve_output_path(args["export"]["a_path"]));

		save_pvd(
			resolve_output_path("sim.pvd"),
			[](int i)
			{ return fmt::format("step_{:d}.vtu", i); },
			time_steps, t0, dt);

		const bool export_surface = args["export"]["surface"];
		const bool contact_forces = args["export"]["contact_forces"] && !problem->is_scalar();

		if (export_surface)
		{
			save_pvd(
				resolve_output_path("sim_surf.pvd"),
				[](int i)
				{ return fmt::format("step_{:d}_surf.vtu", i); },
				time_steps, t0, dt);

			if (contact_forces)
			{
				save_pvd(
					resolve_output_path("sim_surf_contact.pvd"),
					[](int i)
					{ return fmt::format("step_{:d}_surf_contact.vtu", i); },
					time_steps, t0, dt);
			}
		}
	}

	void State::solve_linear()
	{
		assert(!problem->is_time_dependent());
		assert(assembler.is_linear(formulation()) && !args["has_collision"]);
		const json &params = solver_params();
		auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);
		solver->setParameters(params);
		StiffnessMatrix A;
		Eigen::VectorXd b;
		logger().info("{}...", solver->name());
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		if (formulation() != "Bilaplacian")
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
		else
			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], std::vector<LocalBoundary>(), rhs);

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		A = stiffness;
		Eigen::VectorXd x;
		b = rhs;
		spectrum = dirichlet_solve(*solver, A, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"], assembler.is_fluid(formulation()), use_avg_pressure);
		sol = x;
		solver->getInfo(solver_info);

		logger().debug("Solver error: {}", (A * sol - b).norm());

		if (assembler.is_mixed(formulation()))
		{
			sol_to_pressure();
		}
	}

	void State::solve_navier_stokes()
	{
		assert(!problem->is_time_dependent());
		assert(formulation() == "NavierStokes");
		auto params = build_json_params();
		const double viscosity = params.count("viscosity") ? double(params["viscosity"]) : 1.;
		NavierStokesSolver ns_solver(viscosity, solver_params(), build_json_params(), solver_type(), precond_type());
		Eigen::VectorXd x;
		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)

		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, mesh->dimension(),
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);
		rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, rhs);
		ns_solver.minimize(*this, rhs, x);
		sol = x;
		sol_to_pressure();
	}

	void State::solve_non_linear()
	{
		assert(!problem->is_time_dependent());
		assert(!assembler.is_linear(formulation()) || args["has_collision"]);

		const int full_size = n_bases * mesh->dimension();
		const int reduced_size = n_bases * mesh->dimension() - boundary_nodes.size();

		const int problem_dim = problem->is_scalar() ? 1 : mesh->dimension();
		const int precond_num = problem_dim * n_bases;

		const auto &gbases = iso_parametric() ? bases : geom_bases;

		json rhs_solver_params = args["rhs_solver_params"];
		rhs_solver_params["mtype"] = -2; // matrix type for Pardiso (2 = SPD)
		const int size = problem->is_scalar() ? 1 : mesh->dimension();
		RhsAssembler rhs_assembler(assembler, *mesh,
								   n_bases, size,
								   bases, iso_parametric() ? bases : geom_bases, ass_vals_cache,
								   formulation(), *problem,
								   args["rhs_solver_type"], args["rhs_precond_type"], rhs_solver_params);

		Eigen::VectorXd tmp_sol;

		sol.resizeLike(rhs);
		sol.setZero();

		const std::string u_path = resolve_path(args["import"]["u_path"], args["root_path"]);
		if (!u_path.empty())
			import_matrix(u_path, args["import"], sol);

		// if (args["use_al"] || args["has_collision"])
		// {
		//FD
		{
			// 	ALNLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"], false, 1e6);
			// 	tmp_sol = rhs;
			// 	tmp_sol.setRandom();
			// 	// tmp_sol.setOnes();
			// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
			// 	nl_problem.gradient(tmp_sol, actual_grad);

			// 	StiffnessMatrix hessian;
			// 	// Eigen::MatrixXd expected_hessian;
			// 	nl_problem.hessian(tmp_sol, hessian);
			// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

			// 	// Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
			// 	// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

			// 	for (int i = 0; i < hessian.rows(); ++i)
			// 	{
			// 		double hhh = 1e-6;
			// 		VectorXd xp = tmp_sol;
			// 		xp(i) += hhh;
			// 		VectorXd xm = tmp_sol;
			// 		xm(i) -= hhh;

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
			// 		nl_problem.gradient(xp, tmp_grad_p);

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
			// 		nl_problem.gradient(xm, tmp_grad_m);

			// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m) / (hhh * 2.);

			// 		const double vp = nl_problem.value(xp);
			// 		const double vm = nl_problem.value(xm);

			// 		const double fd = (vp - vm) / (hhh * 2.);
			// 		const double diff = std::abs(actual_grad(i) - fd);
			// 		if (diff > 1e-5)
			// 			std::cout << "diff grad " << i << ": " << actual_grad(i) << " vs " << fd << " error: " << diff << " rrr: " << actual_grad(i) / fd << std::endl;

			// 		for (int j = 0; j < hessian.rows(); ++j)
			// 		{
			// 			const double diff = std::abs(hessian.coeffRef(i, j) - fd_h(j));

			// 			if (diff > 1e-4)
			// 				std::cout << "diff H " << i << ", " << j << ": " << hessian.coeffRef(i, j) << " vs " << fd_h(j) << " error: " << diff << " rrr: " << hessian.coeffRef(i, j) / fd_h(j) << std::endl;
			// 		}
			// 	}

			// 	// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
			// 	// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
			// 	exit(0);
		}

		ALNLProblem alnl_problem(*this, rhs_assembler, 1, args["dhat"], args["project_to_psd"], args["al_weight"]);
		NLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"], args["project_to_psd"]);

		double al_weight = args["al_weight"];
		const double max_al_weight = args["max_al_weight"];
		nl_problem.full_to_reduced(sol, tmp_sol);

		nl_problem.update_lagging(tmp_sol, /*start_of_timestep=*/true);
		alnl_problem.update_lagging(sol, /*start_of_timestep=*/true);

		//TODO: maybe add linear solver here?

		solver_info = json::array();

		int index = 0;
		nl_problem.line_search_begin(sol, tmp_sol);
		while (!std::isfinite(nl_problem.value(tmp_sol)) || !nl_problem.is_step_valid(sol, tmp_sol) || !nl_problem.is_step_collision_free(sol, tmp_sol))
		{
			nl_problem.line_search_end();
			alnl_problem.set_weight(al_weight);
			logger().debug("Solving AL Problem with weight {}", al_weight);

			cppoptlib::SparseNewtonDescentSolver<ALNLProblem> alnlsolver(solver_params(), solver_type(), precond_type());
			alnlsolver.setLineSearch(args["line_search"]);
			alnl_problem.init(sol);
			tmp_sol = sol;
			alnlsolver.minimize(alnl_problem, tmp_sol);
			json alnl_solver_info;
			alnlsolver.getInfo(alnl_solver_info);

			solver_info.push_back({{"type", "al"},
								   {"weight", al_weight},
								   {"info", alnl_solver_info}});

			sol = tmp_sol;
			nl_problem.full_to_reduced(sol, tmp_sol);
			nl_problem.line_search_begin(sol, tmp_sol);

			al_weight *= 2;

			if (al_weight >= max_al_weight)
			{
				logger().error("Unable to solve AL problem, weight {} >= {}, stopping", al_weight, max_al_weight);
				break;
			}

			if (args["save_solve_sequence_debug"])
			{
				if (!solve_export_to_file)
					solution_frames.emplace_back();
				save_vtu(fmt::format("step_{:d}.vtu", index), 1);
				save_wire(fmt::format("step_{:d}.obj", index));
			}
			++index;
		}
		nl_problem.line_search_end();
		logger().debug("Solving Problem");
		cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		nlsolver.setLineSearch(args["line_search"]);
		nl_problem.init(sol);
		nlsolver.minimize(nl_problem, tmp_sol);
		json nl_solver_info;
		nlsolver.getInfo(nl_solver_info);

		nl_problem.reduced_to_full(tmp_sol, sol);
		solver_info.push_back({{"type", "rc"},
							   {"info", nl_solver_info}});

		{
			const std::string u_path = resolve_path(args["export"]["u_path"], args["root_path"]);
			if (!u_path.empty())
				write_matrix_binary(u_path, sol);
		}
		// }
		// else
		// {
		// 	int steps = args["nl_solver_rhs_steps"];
		// 	if (steps <= 0)
		// 	{
		// 		RowVectorNd min, max;
		// 		mesh->bounding_box(min, max);
		// 		steps = problem->n_incremental_load_steps((max - min).norm());
		// 	}
		// 	steps = std::max(steps, 1);

		// 	double step_t = 1.0 / steps;
		// 	double t = step_t;
		// 	double prev_t = 0;

		// 	StiffnessMatrix nlstiffness;
		// 	Eigen::VectorXd b;
		// 	Eigen::MatrixXd grad;
		// 	Eigen::MatrixXd prev_rhs;

		// 	prev_rhs.resizeLike(rhs);
		// 	prev_rhs.setZero();

		// 	b.resizeLike(sol);
		// 	b.setZero();

		// 	if (args["save_solve_sequence"])
		// 	{
		// 		if (!solve_export_to_file)
		// 			solution_frames.emplace_back();
		// 		save_vtu(fmt::format("step_{:d}.vtu", prev_t), 1);
		// 		save_wire(fmt::format("step_{:d}.obj", prev_t));
		// 	}

		// 	igl::Timer update_timer;
		// 	auto solver = polysolve::LinearSolver::create(args["solver_type"], args["precond_type"]);

		// 	while (t <= 1)
		// 	{
		// 		if (step_t < 1e-10)
		// 		{
		// 			logger().error("Step too small, giving up");
		// 			break;
		// 		}

		// 		logger().info("t: {} prev: {} step: {}", t, prev_t, step_t);

		// 		NLProblem nl_problem(*this, rhs_assembler, t, args["dhat"], args["project_to_psd"]);

		// 		logger().debug("Updating starting point...");
		// 		update_timer.start();

		// 		{
		// 			nl_problem.hessian_full(sol, nlstiffness);
		// 			nl_problem.gradient_no_rhs(sol, grad);
		// 			rhs_assembler.set_bc(local_boundary, boundary_nodes, args["n_boundary_samples"], local_neumann_boundary, grad, t);

		// 			b = grad;
		// 			for (int bId : boundary_nodes)
		// 				b(bId) = -(nl_problem.current_rhs()(bId) - prev_rhs(bId));
		// 			dirichlet_solve(*solver, nlstiffness, b, boundary_nodes, x, precond_num, args["export"]["stiffness_mat"], args["export"]["spectrum"]);
		// 			logger().trace("Checking step");
		// 			const bool valid = nl_problem.is_step_collision_free(sol, (sol - x).eval());
		// 			if (valid)
		// 				x = sol - x;
		// 			else
		// 				x = sol;
		// 			logger().trace("Done checking step, was {}valid", valid ? "" : "in");
		// 			// logger().debug("Solver error: {}", (nlstiffness * sol - b).norm());
		// 		}

		// 		nl_problem.full_to_reduced(x, tmp_sol);
		// 		update_timer.stop();
		// 		logger().debug("done!, took {}s", update_timer.getElapsedTime());

		// 		if (args["save_solve_sequence_debug"])
		// 		{
		// 			Eigen::MatrixXd xxx = sol;
		// 			sol = x;
		// 			if (assembler.is_mixed(formulation()))
		// 				sol_to_pressure();
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();

		// 			save_vtu(fmt::format("step_s_{:d}.vtu", t), 1);
		// 			save_wire(fmt::format("step_s_{:d}.obj", t));

		// 			sol = xxx;
		// 		}

		// 		bool has_nan = false;
		// 		for (int k = 0; k < tmp_sol.size(); ++k)
		// 		{
		// 			if (std::isnan(tmp_sol[k]))
		// 			{
		// 				has_nan = true;
		// 				break;
		// 			}
		// 		}

		// 		if (has_nan)
		// 		{
		// 			do
		// 			{
		// 				step_t /= 2;
		// 				t = prev_t + step_t;
		// 			} while (t >= 1);
		// 			continue;
		// 		}

		// 		if (args["nl_solver"] == "newton")
		// 		{
		// 			cppoptlib::SparseNewtonDescentSolver<NLProblem> nlsolver(solver_params(), solver_type(), precond_type());
		// 			nlsolver.setLineSearch(args["line_search"]);
		// 			nl_problem.init(x);
		// 			nlsolver.minimize(nl_problem, tmp_sol);

		// 			if (nlsolver.error_code() == -10) //Nan
		// 			{
		// 				do
		// 				{
		// 					step_t /= 2;
		// 					t = prev_t + step_t;
		// 				} while (t >= 1);
		// 				continue;
		// 			}
		// 			else
		// 			{
		// 				prev_t = t;
		// 				step_t *= 2;
		// 			}

		// 			if (step_t > 1.0 / steps)
		// 				step_t = 1.0 / steps;

		// 			nlsolver.getInfo(solver_info);
		// 		}
		// 		else if (args["nl_solver"] == "lbfgs")
		// 		{
		// 			cppoptlib::LbfgsSolverL2<NLProblem> nlsolver;
		// 			nlsolver.setLineSearch(args["line_search"]);
		// 			nlsolver.setDebug(cppoptlib::DebugLevel::High);
		// 			nlsolver.minimize(nl_problem, tmp_sol);

		// 			prev_t = t;
		// 		}
		// 		else
		// 		{
		// 			throw std::invalid_argument("[State] invalid solver type for non-linear problem");
		// 		}

		// 		t = prev_t + step_t;
		// 		if ((prev_t < 1 && t > 1) || abs(t - 1) < 1e-10)
		// 			t = 1;

		// 		nl_problem.reduced_to_full(tmp_sol, sol);

		// 		// std::ofstream of("sol.txt");
		// 		// of<<sol<<std::endl;
		// 		// of.close();
		// 		prev_rhs = nl_problem.current_rhs();
		// 		if (args["save_solve_sequence"])
		// 		{
		// 			if (!solve_export_to_file)
		// 				solution_frames.emplace_back();
		// 			save_vtu(fmt::format("step_{:d}.vtu", prev_t), 1);
		// 			save_wire(fmt::format("step_{:d}.obj", prev_t));
		// 		}
		// 	}

		// 	if (assembler.is_mixed(formulation()))
		// 	{
		// 		sol_to_pressure();
		// 	}

		// 	// {
		// 	// 	boundary_nodes.clear();
		// 	// 	NLProblem nl_problem(*this, rhs_assembler, 1, args["dhat"]);
		// 	// 	tmp_sol = rhs;

		// 	// 	// tmp_sol.setRandom();
		// 	// 	tmp_sol.setOnes();
		// 	// 	Eigen::Matrix<double, Eigen::Dynamic, 1> actual_grad;
		// 	// 	nl_problem.gradient(tmp_sol, actual_grad);

		// 	// 	StiffnessMatrix hessian;
		// 	// 	// Eigen::MatrixXd expected_hessian;
		// 	// 	nl_problem.hessian(tmp_sol, hessian);
		// 	// 	// nl_problem.finiteGradient(tmp_sol, expected_grad, 0);

		// 	// 	Eigen::MatrixXd actual_hessian = Eigen::MatrixXd(hessian);
		// 	// 	// std::cout << "hhh\n"<< actual_hessian<<std::endl;

		// 	// 	for (int i = 0; i < actual_hessian.rows(); ++i)
		// 	// 	{
		// 	// 		double hhh = 1e-6;
		// 	// 		VectorXd xp = tmp_sol; xp(i) += hhh;
		// 	// 		VectorXd xm = tmp_sol; xm(i) -= hhh;

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_p;
		// 	// 		nl_problem.gradient(xp, tmp_grad_p);

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> tmp_grad_m;
		// 	// 		nl_problem.gradient(xm, tmp_grad_m);

		// 	// 		Eigen::Matrix<double, Eigen::Dynamic, 1> fd_h = (tmp_grad_p - tmp_grad_m)/(hhh*2.);

		// 	// 		const double vp = nl_problem.value(xp);
		// 	// 		const double vm = nl_problem.value(xm);

		// 	// 		const double fd = (vp-vm)/(hhh*2.);
		// 	// 		const double  diff = std::abs(actual_grad(i) - fd);
		// 	// 		if(diff > 1e-5)
		// 	// 			std::cout<<"diff grad "<<i<<": "<<actual_grad(i)<<" vs "<<fd <<" error: " <<diff<<" rrr: "<<actual_grad(i)/fd<<std::endl;

		// 	// 		for(int j = 0; j < actual_hessian.rows(); ++j)
		// 	// 		{
		// 	// 			const double diff = std::abs(actual_hessian(i,j) - fd_h(j));

		// 	// 			if(diff > 1e-4)
		// 	// 				std::cout<<"diff H "<<i<<", "<<j<<": "<<actual_hessian(i,j)<<" vs "<<fd_h(j)<<" error: " <<diff<<" rrr: "<<actual_hessian(i,j)/fd_h(j)<<std::endl;

		// 	// 		}
		// 	// 	}

		// 	// 	// std::cout<<"diff grad max "<<(actual_grad - expected_grad).array().abs().maxCoeff()<<std::endl;
		// 	// 	// std::cout<<"diff \n"<<(actual_grad - expected_grad)<<std::endl;
		// 	// 	exit(0);
		// 	// }

		// 	// NLProblem::reduced_to_full_aux(full_size, reduced_size, tmp_sol, rhs, sol);
		// }
	}

} // namespace polyfem
