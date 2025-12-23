#include "Wave.hpp"

void
Wave::setup()
{
  pcout << "===============================================" << std::endl;

  // Create the mesh.
  {
    pcout << "Initializing the mesh on domain [" << domain_left << ", "
          << domain_right << "]^" << dim << std::endl;

    // Create serial mesh first.
    Triangulation<dim> mesh_serial;
    GridGenerator::hyper_cube(mesh_serial, domain_left, domain_right);
    mesh_serial.refine_global(n_refine);

    // Partition and distribute the mesh.
    GridTools::partition_triangulation(mpi_size, mesh_serial);

    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);

    pcout << "  Number of elements = " << mesh.n_global_active_cells()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    pcout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    pcout << "  Degree                     = " << fe->degree << std::endl;
    pcout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    pcout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    pcout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    pcout << "Initializing the linear system" << std::endl;

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);

    pcout << "  Initializing the sparsity pattern" << std::endl;
    TrilinosWrappers::SparsityPattern sparsity(locally_owned_dofs,
                                               MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler, sparsity);
    sparsity.compress();

    pcout << "  Initializing the system matrix" << std::endl;
    system_matrix.reinit(sparsity);

    pcout << "  Initializing vectors" << std::endl;
    system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
    acceleration_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    acceleration.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

    // Initialize mass and stiffness matrices for energy computation.
    pcout << "  Initializing the mass matrix" << std::endl;
    mass_matrix.reinit(sparsity);

    pcout << "  Initializing the stiffness matrix" << std::endl;
    stiffness_matrix.reinit(sparsity);

  }

  pcout << "-----------------------------------------------" << std::endl;

}

// DISPERSION
void
Wave::find_center_point_dof()
{
  const Point<dim> center_point(0.0, 0.0);

  center_point_is_local = false;
  center_dof_index = numbers::invalid_dof_index;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (! cell->is_locally_owned())
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
        pcout << "Center point DoF found:  index = " << center_dof_index
              << ", distance from (0,0) = " << min_distance << std::endl;
        break;
      }
    }
  }

  if (center_point_is_local)
  {
    center_point_file.open("center_point_solution.csv");
    center_point_file << "time,solution,velocity,acceleration\n";
    center_point_file << std::setprecision(12);
  }
}


void
Wave::record_center_point_value()
{
  if (!center_point_is_local)
    return;

  const double u_center = solution[center_dof_index];
  const double v_center = velocity[center_dof_index];
  const double a_center = acceleration[center_dof_index];

  center_point_file << time << ","
                    << u_center << ","
                    << v_center << ","
                    << a_center << "\n";
}

void
Wave::assemble()
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_matrix = 0.0;
  system_rhs    = 0.0;
  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  std::vector<double> solution_old_values(n_q);
  std::vector<Tensor<1, dim>> solution_old_grads(n_q);
  std::vector<double> v_old_values(n_q);
  std::vector<double> a_old_values(n_q);

  std::unique_ptr<Function<dim>> rhs_function;
  if (test_case == EX1)
    rhs_function = std::make_unique<RightHandSideEX1>();
  else
    rhs_function = std::make_unique<RightHandSideEX2>();

  rhs_function->set_time(time);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);

      cell_matrix = 0.0;
      cell_rhs    = 0.0;
      cell_mass_matrix     = 0.0;
      cell_stiffness_matrix = 0.0;

      const double c1 = 1.0 / (beta * delta_t * delta_t);
      const double c2 = 1.0 / (beta * delta_t);
      const double c3 = 1.0 / (2.0 * beta) - 1.0;

      fe_values.get_function_values(solution, solution_old_values);
      fe_values.get_function_gradients(solution, solution_old_grads);
      fe_values.get_function_values(velocity, v_old_values);
      fe_values.get_function_values(acceleration, a_old_values);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double f_new_loc = rhs_function->value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Mass term: (1/(beta*dt^2)) * M
                  cell_matrix(i, j) += (1.0 / (beta * delta_t * delta_t)) *
                                       fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) *
                                       fe_values.JxW(q);

                  // Stiffness term: K
                  cell_matrix(i, j) += scalar_product(fe_values.shape_grad(i, q),
                                                      fe_values.shape_grad(j, q)) *
                                       fe_values.JxW(q);

                  // For energy computation:
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) *
                                            fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                      scalar_product(fe_values.shape_grad(i, q),
                                     fe_values.shape_grad(j, q)) *
                      fe_values.JxW(q);
                }

              // RHS: M * (c1*u^n + c2*v^n + c3*a^n) + F^{n+1}
              cell_rhs(i) += fe_values.shape_value(i, q) *
                             (c1 * solution_old_values[q]
                              + c2 * v_old_values[q]
                              + c3 * a_old_values[q]) *
                             fe_values.JxW(q);

              cell_rhs(i) += f_new_loc *
                             fe_values.shape_value(i, q) *
                             fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);

      system_matrix.add(dof_indices, cell_matrix);
      system_rhs.add(dof_indices, cell_rhs);
      mass_matrix.add(dof_indices, cell_mass_matrix);
      stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
    }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);

  // ------------------- Dirichlet boundary conditions -------------------
  {
      std::map<types::global_dof_index, double> boundary_values;
      FunctionG bc_function;

      std::map<types::boundary_id, const Function<dim> *> boundary_functions;
      boundary_functions[0] = &bc_function;
      boundary_functions[1] = &bc_function;
      boundary_functions[2] = &bc_function;
      boundary_functions[3] = &bc_function;

      VectorTools::interpolate_boundary_values(dof_handler,
                                               boundary_functions,
                                               boundary_values);

      MatrixTools::apply_boundary_values(boundary_values,
                                         system_matrix,
                                         solution_owned,
                                         system_rhs,
                                         true);
  }
}

void
Wave::solve_linear_system()
{
  TrilinosWrappers::PreconditionSSOR preconditioner;
  preconditioner.initialize(
    system_matrix, TrilinosWrappers::PreconditionSSOR::AdditionalData(1.0));

  ReductionControl solver_control(10000, 1.0e-16, 1.0e-6);

  SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);

  solver.solve(system_matrix, solution_owned, system_rhs, preconditioner);
  pcout << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Wave::output() const
{
  DataOut<dim> data_out;

  data_out.add_data_vector(dof_handler, solution, "solution");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::string output_file_name = "output-hypercube";

  data_out.write_vtu_with_pvtu_record("./",
                                      output_file_name,
                                      timestep_number,
                                      MPI_COMM_WORLD);
}

void
Wave::run(Function<dim> *exact_solution)
{
  std::ofstream error_file;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && exact_solution != nullptr)
  {
    error_file.open("errors.csv");
    error_file << "time,L2_error,H1_error\n";
  }

  std::ofstream energy_file;
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    energy_file.open("energy_2.csv");
    energy_file << "time,total_energy,kinetic_energy,potential_energy\n";
  }

  // Setup initial conditions.
  {
    setup();
    find_center_point_dof();

    VectorTools::interpolate(dof_handler, FunctionU0(), solution_owned);
    solution = solution_owned;

    VectorTools::interpolate(dof_handler, FunctionU1(), velocity_owned);
    velocity = velocity_owned;

    if (test_case == EX1)
      VectorTools::interpolate(dof_handler, FunctionU2_EX1(), acceleration_owned);
    else
      VectorTools::interpolate(dof_handler, FunctionU2_EX2(), acceleration_owned);
    acceleration = acceleration_owned;

    time            = 0.0;
    timestep_number = 0;

    output();
    record_center_point_value();
  }

  pcout << "===============================================" << std::endl;

  // Time-stepping loop.
  while (time < T - 0.5 * delta_t)
    {
      time += delta_t;
      ++timestep_number;

      pcout << "Timestep " << std::setw(3) << timestep_number
            << ", time = " << std::setw(4) << std::fixed << std::setprecision(2)
            << time << " : ";

      // Save old OWNED values BEFORE assemble (needed for Newmark update)
      TrilinosWrappers::MPI::Vector u_old(solution_owned);
      TrilinosWrappers::MPI::Vector v_old(velocity_owned);
      TrilinosWrappers::MPI::Vector a_old(acceleration_owned);

      assemble();
      solve_linear_system();

      // Newmark coefficients
      const double N1 = 1.0 / (beta * delta_t * delta_t);
      const double N2 = 1.0 / (beta * delta_t);
      const double N3 = 1.0 / (2.0 * beta) - 1.0;

      // Update acceleration: a^{n+1} = N1*(u^{n+1}-u^n) - N2*v^n - N3*a^n
      acceleration_owned = 0.0;
      acceleration_owned.add( N1, solution_owned);
      acceleration_owned.add(-N1, u_old);
      acceleration_owned.add(-N2, v_old);
      acceleration_owned.add(-N3, a_old);

      // Update velocity: v^{n+1} = v^n + dt*((1-gamma)*a^n + gamma*a^{n+1})
      velocity_owned = v_old;
      velocity_owned.add(delta_t*(1.0 - gamma), a_old);
      velocity_owned.add(delta_t*gamma, acceleration_owned);

      // =====================================================================
      // FIX: Update ghost values BEFORE computing errors!
      // This was the bug - error was computed with old solution values
      // =====================================================================
      solution = solution_owned;
      velocity = velocity_owned;
      acceleration = acceleration_owned;

      // Compute errors at current time (NOW using UPDATED solution!)
      int error_interval = 10;
      if (exact_solution != nullptr && timestep_number % error_interval == 0)
      {
        const double error_L2 = compute_error(VectorTools::L2_norm, *exact_solution);
        const double error_H1 = compute_error(VectorTools::H1_norm, *exact_solution);

        // Write to CSV with full precision
        error_file << std::setprecision(12) << std::scientific
                   << time << "," << error_L2 << "," << error_H1 << "\n";

        // Reset formatting for console output (undo the fixed/setprecision(2) from timestep printout)
        pcout << std::scientific << std::setprecision(6)
              << "L2 error = " << error_L2
              << ", H1 error = " << error_H1 << std::endl;
      }

      record_center_point_value();

      const double E = compute_total_energy();

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        energy_file << std::setprecision(12) << std::scientific
                    << time << "," << E << "\n";
        pcout << std::scientific << std::setprecision(6)
              << "Energy = " << E << std::endl;
      }

      // Reset to default formatting for next iteration
      pcout << std::defaultfloat;

      output();
    }

    if (error_file.is_open())
      error_file.close();
    if (energy_file.is_open())
      energy_file.close();
    if (center_point_file.is_open())
    {
      center_point_file.close();
      pcout << "Center point time series saved to: center_point_solution.csv" << std::endl;
    }
}

double
Wave::compute_error(const VectorTools::NormType &norm_type,
                    Function<dim> &exact_solution) const
{
  Assert(fe.get() != nullptr, ExcMessage("FE not initialized"));
  Assert(dof_handler.n_dofs() == solution.size(), ExcMessage("solution size != n_dofs"));
  Assert(mesh.n_active_cells() > 0, ExcMessage("mesh has no active cells"));

  QGauss<dim> quadrature_error(r + 2);
  MappingQ<dim> mapping(1);

  exact_solution.set_time(time);

  Vector<double> error_per_cell(mesh.n_active_cells());

  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  const double error = VectorTools::compute_global_error(mesh,
                                                         error_per_cell,
                                                         norm_type);

  return error;
}


double
Wave::compute_total_energy() const
{
  TrilinosWrappers::MPI::Vector Mv(velocity_owned);
  TrilinosWrappers::MPI::Vector Au(solution_owned);

  mass_matrix.vmult(Mv, velocity_owned);
  stiffness_matrix.vmult(Au, solution_owned);

  const double kinetic = 0.5 * (velocity_owned * Mv);
  const double potential = 0.5 * (solution_owned * Au);

  return kinetic + potential;
}