#include "WaveParallel.hpp"

WaveParallel::WaveParallel(const unsigned int degree_,
                           const double       T_,
                           const double       theta_,
                           const double       delta_t_,
                           const double       domain_left_,
                           const double       domain_right_,
                           const unsigned int n_refine_,
                           const TestCase     test_case_)
  : degree(degree_)
  , T(T_)
  , theta(theta_)
  , delta_t(delta_t_)
  , domain_left(domain_left_)
  , domain_right(domain_right_)
  , n_refine(n_refine_)
  , test_case(test_case_)
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
  pcout << "Test case: EX" << test_case << std::endl;

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

// DISPERSION ANALYSIS
void
WaveParallel::find_center_point_dof()
{
  const Point<dim> center_point(0.0, 0.0);

  center_point_is_local = false;
  center_dof_index = numbers::invalid_dof_index;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    const Point<dim> cell_center = cell->center();
    const double cell_diameter = cell->diameter();

    if (center_point.distance(cell_center) < cell_diameter)
    {
      std::vector<types::global_dof_index> local_dof_indices(fe->dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      double min_distance = std::numeric_limits<double>::max();

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const Point<dim> vertex = cell->vertex(v);
        const double distance = center_point.distance(vertex);

        if (distance < min_distance)
        {
          min_distance = distance;
          center_dof_index = local_dof_indices[v];
        }
      }

      if (min_distance < 1e-10)
      {
        center_point_is_local = true;
        pcout << "Center point DoF found: index = " << center_dof_index
              << ", distance from (0,0) = " << min_distance << std::endl;
        break;
      }
    }
  }

  if (center_point_is_local)
  {
    center_point_file.open("center_point_solution_theta.csv");
    center_point_file << "time,solution,velocity\n";
    center_point_file << std::setprecision(12);
  }
}

void
WaveParallel::record_center_point_value()
{
  if (!center_point_is_local)
    return;

  const double u_center = solution_u[center_dof_index];
  const double v_center = solution_v[center_dof_index];

  center_point_file << time << ","
                    << u_center << ","
                    << v_center << "\n";
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
WaveParallel::assemble_forcing_terms()
{
  // Compute forcing terms: theta * f^n + (1-theta) * f^{n-1}
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Create appropriate forcing function based on test case
  std::unique_ptr<Function<dim>> rhs_function;
  if (test_case == EX1)
    rhs_function = std::make_unique<RightHandSideEX1>();
  else
    rhs_function = std::make_unique<RightHandSideEX2>();

  // First compute theta * f^n
  forcing_terms_owned = 0.0;
  rhs_function->set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double rhs_value = rhs_function->value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += fe_values.shape_value(i, q) * rhs_value *
                             fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if (locally_owned_dofs.is_element(dof_indices[i]))
            forcing_terms_owned(dof_indices[i]) += theta * cell_rhs(i);
        }
    }

  // Then add (1-theta) * f^{n-1}
  rhs_function->set_time(time - delta_t);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double rhs_value = rhs_function->value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += fe_values.shape_value(i, q) * rhs_value *
                             fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if (locally_owned_dofs.is_element(dof_indices[i]))
            forcing_terms_owned(dof_indices[i]) += (1.0 - theta) * cell_rhs(i);
        }
    }

  forcing_terms_owned.compress(VectorOperation::add);
}

void
WaveParallel::assemble_system_u()
{
  // Assemble forcing terms first.
  assemble_forcing_terms();

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

  // + theta*dt * forcing_terms
  system_rhs.add(theta * delta_t, forcing_terms_owned);

  // Build system matrix: M + theta^2 * dt^2 * A
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(theta * theta * delta_t * delta_t, laplace_matrix);

  // Apply boundary conditions manually for Trilinos.
  BoundaryValuesU boundary_values_u_function;
  boundary_values_u_function.set_time(time);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           boundary_values_u_function,
                                           boundary_values);

  // Apply boundary values.
  for (const auto &[dof_index, bc_value] : boundary_values)
    {
      if (locally_owned_dofs.is_element(dof_index))
        {
          system_rhs(dof_index) = bc_value;

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

  for (const auto &[dof_index, bc_value] : boundary_values)
    {
      if (locally_owned_dofs.is_element(dof_index))
        {
          system_rhs(dof_index) = bc_value;

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

  solver.solve(system_matrix, solution_u_owned, system_rhs, PreconditionIdentity());

  pcout << "   u-equation: " << solver_control.last_step()
        << " CG iterations." << std::endl;

  // Update ghosted vector.
  solution_u = solution_u_owned;
}

void
WaveParallel::solve_v()
{


  ReductionControl solver_control(/* maxiter = */ 1000,
                                  /* tolerance = */ 1.0e-12,
                                  /* reduce = */ 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_v_owned, system_rhs, PreconditionIdentity());

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

  // Compute and output the exact solution
  TrilinosWrappers::MPI::Vector exact_owned;
  exact_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  TrilinosWrappers::MPI::Vector exact_solution_vec;
  exact_solution_vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  // Choose the appropriate exact solution based on test case
  if (test_case == EX1)
  {
    ExactSolutionEX1 exact_func;
    exact_func.set_time(time);
    VectorTools::interpolate(dof_handler, exact_func, exact_owned);
  }
  else
  {
    ExactSolutionEX2 exact_func;
    exact_func.set_time(time);
    VectorTools::interpolate(dof_handler, exact_func, exact_owned);
  }
  exact_solution_vec = exact_owned;

  data_out.add_data_vector(dof_handler, exact_solution_vec, "U_exact");

  // Compute and output the pointwise error: U - U_exact
  TrilinosWrappers::MPI::Vector error_owned;
  error_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  error_owned = solution_u_owned;
  error_owned -= exact_owned;

  // Take absolute value of each entry
  for (const auto &idx : locally_owned_dofs)
  {
    error_owned[idx] = std::abs(error_owned[idx]);
  }

  TrilinosWrappers::MPI::Vector error_vec;
  error_vec.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  error_vec = error_owned;

  data_out.add_data_vector(dof_handler, error_vec, "Error");

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

double
WaveParallel::compute_error(const VectorTools::NormType &norm_type,
                            Function<dim>               &exact_solution) const
{
  // Assertions for safety (matching Newmark style)
  Assert(fe.get() != nullptr, ExcMessage("FE not initialized"));
  Assert(dof_handler.n_dofs() == solution_u.size(),
         ExcMessage("solution size != n_dofs"));
  Assert(triangulation.n_global_active_cells() > 0,
         ExcMessage("mesh has no active cells"));

  // Use Gauss quadrature for hypercube meshes
  const QGauss<dim> quadrature_error(fe->degree + 2);

  // Use MappingQ for hypercube meshes (matching Newmark style)
  MappingQ<dim> mapping(1);

  // Set the time for the exact solution
  exact_solution.set_time(time);

  Vector<double> error_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution_u,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error = VectorTools::compute_global_error(triangulation,
                                                         error_per_cell,
                                                         norm_type);
  return error;
}

double
WaveParallel::compute_total_energy() const
{
  // Compute total energy: E = 0.5 * (v^T M v + u^T A u)
  // This is the same formula used in the Newmark implementation

  TrilinosWrappers::MPI::Vector Mv(solution_v_owned);
  TrilinosWrappers::MPI::Vector Au(solution_u_owned);

  // Compute M * v
  mass_matrix.vmult(Mv, solution_v_owned);

  // Compute A * u
  laplace_matrix.vmult(Au, solution_u_owned);

  // Compute kinetic energy: 0.5 * v^T * M * v
  const double kinetic_energy = 0.5 * (solution_v_owned * Mv);

  // Compute potential energy: 0.5 * u^T * A * u
  const double potential_energy = 0.5 * (solution_u_owned * Au);

  return kinetic_energy + potential_energy;
}

void
WaveParallel::run(Function<dim> *exact_solution)
{
  setup_system();

  // Assemble mass and laplace matrices (done once).
  assemble_matrices();

  // Find center point DoF for dispersion analysis
  find_center_point_dof();

  // Open a file to save error history (only on rank 0).
  std::ofstream error_file;
  if (exact_solution != nullptr && mpi_rank == 0)
    {
      error_file.open("errors_parallel.csv");
      error_file << "time,L2_error,H1_error\n";
    }

  // Open a file to save energy history (only on rank 0).
  std::ofstream energy_file;
  if (mpi_rank == 0)
    {
      energy_file.open("energy_parallel_2.csv");
      energy_file << "time,total_energy,kinetic_energy,potential_energy\n";
    }

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

  // Record initial condition at center point (t=0)
  // We need to temporarily set solution_u/v to old values to record t=0
  solution_u = old_solution_u;
  solution_v = old_solution_v;
  double saved_time = time;
  time = 0.0;
  record_center_point_value();
  time = saved_time;

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

      // Record center point value for dispersion analysis
      record_center_point_value();

      // Compute and output energy using the Newmark formula:
      // E = 0.5 * (v^T M v + u^T A u)
      TrilinosWrappers::MPI::Vector Mv(solution_v_owned);
      TrilinosWrappers::MPI::Vector Au(solution_u_owned);
      mass_matrix.vmult(Mv, solution_v_owned);
      laplace_matrix.vmult(Au, solution_u_owned);
      const double kinetic_energy = 0.5 * (solution_v_owned * Mv);
      const double potential_energy = 0.5 * (solution_u_owned * Au);
      const double total_energy = kinetic_energy + potential_energy;

      pcout << "   Total energy: " << total_energy
            << " (kinetic: " << kinetic_energy
            << ", potential: " << potential_energy << ")" << std::endl;

      if (mpi_rank == 0)
        {
          energy_file << time << "," << total_energy << ","
                      << kinetic_energy << "," << potential_energy << "\n";
        }

      // Compute and output errors if exact solution is provided (matching Newmark style).
      const int error_interval = 10; // Compute every N time steps
      if (exact_solution != nullptr && timestep_number % error_interval == 0)
        {
          const double error_L2 = compute_error(VectorTools::L2_norm, *exact_solution);
          const double error_H1 = compute_error(VectorTools::H1_norm, *exact_solution);

          if (mpi_rank == 0)
            {
              error_file << time << "," << error_L2 << "," << error_H1 << "\n";
            }

          pcout << "   Time = " << time
                << ", L2 error = " << error_L2
                << ", H1 error = " << error_H1 << std::endl;
        }

      // Update old solutions.
      old_solution_u_owned = solution_u_owned;
      old_solution_v_owned = solution_v_owned;
      old_solution_u       = solution_u;
      old_solution_v       = solution_v;
    }

  if (error_file.is_open())
    {
      error_file.close();
    }
  if (energy_file.is_open())
    {
      energy_file.close();
    }
  if (center_point_file.is_open())
    {
      center_point_file.close();
      pcout << "Center point time series saved to: center_point_solution_theta.csv" << std::endl;
    }

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Simulation complete." << std::endl;
}