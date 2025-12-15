#include "WaveParallel.hpp"

WaveParallel::WaveParallel(const unsigned int degree_,
                           const double       T_,
                           const double       theta_,
                           const double       delta_t_,
                           const double       domain_left_,
                           const double       domain_right_,
                           const unsigned int n_refine_)
  : degree(degree_)
  , T(T_)
  , theta(theta_)
  , delta_t(delta_t_)
  , domain_left(domain_left_)
  , domain_right(domain_right_)
  , n_refine(n_refine_)
  , time(delta_t_)
  , timestep_number(1)
  , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
  , triangulation(MPI_COMM_WORLD)
  , pcout(std::cout, mpi_rank == 0)
{}

void
WaveParallel::setup_system()
{
  pcout << "===============================================" << std::endl;
  pcout << "Setting up the parallel system" << std::endl;

  // Create the mesh.
  {
    pcout << "  Creating mesh on domain [" << domain_left << ", "
          << domain_right << "]^" << dim << std::endl;

    // Create serial mesh first.
    Triangulation<dim> mesh_serial;
    GridGenerator::hyper_cube(mesh_serial, domain_left, domain_right);
    mesh_serial.refine_global(n_refine);

    // Partition and distribute the mesh.
    GridTools::partition_triangulation(mpi_size, mesh_serial);

    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    triangulation.create_triangulation(construction_data);

    pcout << "  Number of active cells: " << triangulation.n_global_active_cells()
          << std::endl;
  }

  // Initialize the finite element space.
  {
    pcout << "  Initializing finite element space" << std::endl;

    fe        = std::make_unique<FE_Q<dim>>(degree);
    quadrature = std::make_unique<QGauss<dim>>(fe->degree + 1);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell << std::endl;
    pcout << "  Quadrature points per cell = " << quadrature->size() << std::endl;
  }

  // Initialize the DoF handler.
  {
    pcout << "  Initializing DoF handler" << std::endl;

    dof_handler.reinit(triangulation);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  // Initialize the linear system.
  {
    pcout << "  Initializing the linear system" << std::endl;

    locally_owned_dofs    = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

    // Initialize sparsity pattern.
    pcout << "    Initializing sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    // Initialize matrices.
    pcout << "    Initializing matrices" << std::endl;
    mass_matrix.reinit(sparsity);
    laplace_matrix.reinit(sparsity);
    system_matrix.reinit(sparsity);

    // Initialize vectors.
    pcout << "    Initializing vectors" << std::endl;

    // Owned vectors (for solving).
    solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    old_solution_u_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    old_solution_v_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    tmp_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    forcing_terms_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // Ghosted vectors (for reading values on ghost cells).
    solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    old_solution_u.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    old_solution_v.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  }

  pcout << "===============================================" << std::endl;
}

void
WaveParallel::assemble_matrices()
{
  pcout << "  Assembling mass and laplace matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix    = 0.0;
  laplace_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_mass_matrix    = 0.0;
      cell_laplace_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass matrix.
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);

                  // Laplace matrix.
                  cell_laplace_matrix(i, j) +=
                    scalar_product(fe_values.shape_grad(i, q),
                                   fe_values.shape_grad(j, q)) *
                    fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      mass_matrix.add(dof_indices, cell_mass_matrix);
      laplace_matrix.add(dof_indices, cell_laplace_matrix);
    }

  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}

void
WaveParallel::assemble_system_u()
{
  // Assemble RHS for u equation:
  // M * u^{n-1} + dt * M * v^{n-1} - theta*(1-theta)*dt^2 * A * u^{n-1}
  // + theta*dt * forcing_terms

  system_rhs = 0.0;
  tmp_owned  = 0.0;

  // M * u^{n-1}
  mass_matrix.vmult(system_rhs, old_solution_u_owned);

  // + dt * M * v^{n-1}
  mass_matrix.vmult(tmp_owned, old_solution_v_owned);
  system_rhs.add(delta_t, tmp_owned);

  // - theta*(1-theta)*dt^2 * A * u^{n-1}
  laplace_matrix.vmult(tmp_owned, old_solution_u_owned);
  system_rhs.add(-theta * (1.0 - theta) * delta_t * delta_t, tmp_owned);

  // Compute forcing terms.
  forcing_terms_owned = 0.0;

  // For now, forcing is zero (RightHandSide returns 0).

  // Add forcing terms to RHS.
  system_rhs.add(theta * delta_t, forcing_terms_owned);

  // Build system matrix: M + theta^2 * dt^2 * A
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(theta * theta * delta_t * delta_t, laplace_matrix);

  // Apply boundary conditions manually for Trilinos.
  // Get boundary values.
  BoundaryValuesU boundary_values_u_function;
  boundary_values_u_function.set_time(time);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_values_u_function,
                                           boundary_values);

  // Apply boundary values: for each constrained DoF, set the matrix row
  // to have only 1 on the diagonal and set RHS to the boundary value.
  for (const auto &[dof_index, bc_value] : boundary_values)
    {
      if (locally_owned_dofs.is_element(dof_index))
        {
          // Set RHS to boundary value
          system_rhs(dof_index) = bc_value;

          // Clear the matrix row and set diagonal to 1
          // We need to get all column indices for this row
          const auto row_length = system_matrix.row_length(dof_index);
          std::vector<types::global_dof_index> column_indices(row_length);
          std::vector<double> column_values(row_length);

          // Get all entries in this row
          unsigned int n_entries = 0;
          for (auto it = system_matrix.begin(dof_index);
               it != system_matrix.end(dof_index);
               ++it, ++n_entries)
            {
              column_indices[n_entries] = it->column();
              column_values[n_entries]  = (it->column() == dof_index) ? 1.0 : 0.0;
            }

          // Set the row values
          system_matrix.set(dof_index,
                            n_entries,
                            column_indices.data(),
                            column_values.data());
        }
    }

  system_matrix.compress(VectorOperation::insert);
  system_rhs.compress(VectorOperation::insert);
}

void
WaveParallel::assemble_system_v()
{
  // Assemble RHS for v equation:
  // M * v^{n-1} - theta*dt * A * u^n - (1-theta)*dt * A * u^{n-1} + forcing_terms

  system_rhs = 0.0;
  tmp_owned  = 0.0;

  // -theta*dt * A * u^n
  laplace_matrix.vmult(system_rhs, solution_u_owned);
  system_rhs *= -theta * delta_t;

  // + M * v^{n-1}
  mass_matrix.vmult(tmp_owned, old_solution_v_owned);
  system_rhs += tmp_owned;

  // - (1-theta)*dt * A * u^{n-1}
  laplace_matrix.vmult(tmp_owned, old_solution_u_owned);
  system_rhs.add(-delta_t * (1.0 - theta), tmp_owned);

  // + forcing_terms
  system_rhs += forcing_terms_owned;

  // Build system matrix: M
  system_matrix.copy_from(mass_matrix);

  // Apply boundary conditions manually for Trilinos.
  BoundaryValuesV boundary_values_v_function;
  boundary_values_v_function.set_time(time);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_values_v_function,
                                           boundary_values);

  // Apply boundary values: for each constrained DoF, set the matrix row
  // to have only 1 on the diagonal and set RHS to the boundary value.
  for (const auto &[dof_index, bc_value] : boundary_values)
    {
      if (locally_owned_dofs.is_element(dof_index))
        {
          // Set RHS to boundary value
          system_rhs(dof_index) = bc_value;

          // Clear the matrix row and set diagonal to 1
          const auto row_length = system_matrix.row_length(dof_index);
          std::vector<types::global_dof_index> column_indices(row_length);
          std::vector<double> column_values(row_length);

          unsigned int n_entries = 0;
          for (auto it = system_matrix.begin(dof_index);
               it != system_matrix.end(dof_index);
               ++it, ++n_entries)
            {
              column_indices[n_entries] = it->column();
              column_values[n_entries]  = (it->column() == dof_index) ? 1.0 : 0.0;
            }

          system_matrix.set(dof_index,
                            n_entries,
                            column_indices.data(),
                            column_values.data());
        }
    }

  system_matrix.compress(VectorOperation::insert);
  system_rhs.compress(VectorOperation::insert);
}

void
WaveParallel::solve_u()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-12,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_u_owned, system_rhs, preconditioner);

  pcout << "   u-equation: " << solver_control.last_step()
        << " CG iterations." << std::endl;

  // Update ghosted vector.
  solution_u = solution_u_owned;
}

void
WaveParallel::solve_v()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-12,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_v_owned, system_rhs, preconditioner);

  pcout << "   v-equation: " << solver_control.last_step()
        << " CG iterations." << std::endl;

  // Update ghosted vector.
  solution_v = solution_v_owned;
}

void
WaveParallel::output_results() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution_u, "U");
  data_out.add_data_vector(dof_handler, solution_v, "V");

  // Add partitioning information.
  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./",
                                      "solution",
                                      timestep_number,
                                      MPI_COMM_WORLD);
}

void
WaveParallel::run()
{
  setup_system();

  // Assemble mass and laplace matrices (done once).
  assemble_matrices();

  // Project initial conditions.
  pcout << "Projecting initial conditions" << std::endl;

  VectorTools::interpolate(dof_handler,
                           InitialValuesU(),
                           old_solution_u_owned);
  old_solution_u = old_solution_u_owned;

  VectorTools::interpolate(dof_handler,
                           InitialValuesV(),
                           old_solution_v_owned);
  old_solution_v = old_solution_v_owned;

  pcout << std::endl;
  pcout << "Starting time-stepping loop..." << std::endl;
  pcout << "-----------------------------------------------" << std::endl;

  // Time-stepping loop.
  for (; time <= T; time += delta_t, ++timestep_number)
    {
      pcout << "Time step " << timestep_number << " at t=" << time << std::endl;

      // Solve for u.
      assemble_system_u();
      solve_u();

      // Solve for v.
      assemble_system_v();
      solve_v();

      // Output results.
      output_results();

      // Compute and output energy (approximate).
      // Note: This is a simplified energy computation for parallel.
      const double kinetic_energy  = solution_v_owned * solution_v_owned;
      const double potential_energy = solution_u_owned * solution_u_owned;
      // pcout << "   Approximate energy: " << 0.5 * (kinetic_energy + potential_energy)
      //       << std::endl;

      // Update old solutions.
      old_solution_u_owned = solution_u_owned;
      old_solution_v_owned = solution_v_owned;
      old_solution_u       = solution_u;
      old_solution_v       = solution_v;
    }

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Simulation complete." << std::endl;
}