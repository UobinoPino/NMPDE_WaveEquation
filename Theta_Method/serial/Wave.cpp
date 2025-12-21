#include "Wave.hpp"

Wave::Wave(const unsigned int degree_,
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
  , fe(degree_)
  , dof_handler(triangulation)
{}

void Wave::setup_system()
{
  std::cout << "===============================================" << std::endl;
  std::cout << "Setting up the system" << std::endl;
  std::cout << "Test case: EX" << test_case << std::endl;

  // Create the mesh.
  {
    std::cout << "  Creating mesh on domain [" << domain_left << ", "
              << domain_right << "]^" << dim << std::endl;

    GridGenerator::hyper_cube(triangulation, domain_left, domain_right);
    triangulation.refine_global(n_refine);

    std::cout << "  Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
  }

  // Initialize the DoF handler.
  {
    dof_handler.distribute_dofs(fe);
    std::cout << "  Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
  }

  // Initialize the sparsity pattern and matrices.
  {
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    matrix_u.reinit(sparsity_pattern);
    matrix_v.reinit(sparsity_pattern);

    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);
  }

  // Initialize solution vectors.
  {
    solution_u.reinit(dof_handler.n_dofs());
    solution_v.reinit(dof_handler.n_dofs());
    old_solution_u.reinit(dof_handler.n_dofs());
    old_solution_v.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  constraints.close();

  std::cout << "===============================================" << std::endl;
}

void Wave::solve_u()
{
  SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

  std::cout << "   u-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

void Wave::solve_v()
{
  SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);

  cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

  std::cout << "   v-equation: " << solver_control.last_step()
            << " CG iterations." << std::endl;
}

void Wave::output_results() const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution_u, "U");
  data_out.add_data_vector(solution_v, "V");

  data_out.build_patches();

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtu";

  DataOutBase::VtkFlags vtk_flags;
  vtk_flags.compression_level = DataOutBase::CompressionLevel::best_speed;
  data_out.set_flags(vtk_flags);

  std::ofstream output(filename);
  data_out.write_vtu(output);
}

double Wave::compute_error(const VectorTools::NormType &norm_type,
                    Function<dim>               &exact_solution) const
{
  // Assertions for safety (matching Newmark style)
  Assert(dof_handler.n_dofs() == solution_u.size(),
         ExcMessage("solution size != n_dofs"));
  Assert(triangulation.n_active_cells() > 0,
         ExcMessage("mesh has no active cells"));

  // Use Gauss quadrature for hypercube meshes
  QGauss<dim> quadrature_error(fe.degree + 2);

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

double Wave::compute_total_energy() const
{
  // Compute total energy: E = 0.5 * (v^T M v + u^T A u)
  // where M is mass matrix and A is stiffness (laplace) matrix

  Vector<double> Mv(solution_v.size());
  Vector<double> Au(solution_u.size());

  // Compute M * v
  mass_matrix.vmult(Mv, solution_v);

  // Compute A * u
  laplace_matrix.vmult(Au, solution_u);

  // Compute kinetic energy: 0.5 * v^T * M * v
  const double kinetic_energy = 0.5 * (solution_v * Mv);

  // Compute potential energy: 0.5 * u^T * A * u
  const double potential_energy = 0.5 * (solution_u * Au);

  return kinetic_energy + potential_energy;
}

void Wave::run(Function<dim> *exact_solution)
{
  setup_system();

  // Open a file to save error history
  std::ofstream error_file;
  if (exact_solution != nullptr)
  {
    error_file.open("errors.csv");
    error_file << "time,L2_error,H1_error\n";
  }

  // Open a file to save energy history
  std::ofstream energy_file;
  energy_file.open("energy.csv");
  energy_file << "time,total_energy,kinetic_energy,potential_energy\n";

  // Project initial conditions.
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe.degree + 1),
                       InitialValuesU(),
                       old_solution_u);
  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe.degree + 1),
                       InitialValuesV(),
                       old_solution_v);

  Vector<double> tmp(solution_u.size());
  Vector<double> forcing_terms(solution_u.size());

  // Create appropriate forcing function based on test case
  std::unique_ptr<Function<dim>> rhs_function;
  if (test_case == EX1)
    rhs_function = std::make_unique<RightHandSideEX1>();
  else
    rhs_function = std::make_unique<RightHandSideEX2>();

  std::cout << std::endl;
  std::cout << "Starting time-stepping loop..." << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;

  // Time-stepping loop.
  for (; time <= T; time += delta_t, ++timestep_number)
    {
      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      // Assemble RHS for u equation:
      // M * u^{n-1} + dt * M * v^{n-1} - theta*(1-theta)*dt^2 * A * u^{n-1}
      // + theta*dt * (theta * f^n + (1-theta) * f^{n-1})
      mass_matrix.vmult(system_rhs, old_solution_u);

      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs.add(delta_t, tmp);

      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-theta * (1 - theta) * delta_t * delta_t, tmp);

      // Forcing terms.
      rhs_function->set_time(time);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree + 1),
                                          *rhs_function,
                                          tmp);
      forcing_terms = tmp;
      forcing_terms *= theta * delta_t;

      rhs_function->set_time(time - delta_t);
      VectorTools::create_right_hand_side(dof_handler,
                                          QGauss<dim>(fe.degree + 1),
                                          *rhs_function,
                                          tmp);
      forcing_terms.add((1 - theta) * delta_t, tmp);

      system_rhs.add(theta * delta_t, forcing_terms);

      // Apply boundary conditions for u.
      {
        BoundaryValuesU boundary_values_u_function;
        boundary_values_u_function.set_time(time);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 boundary_values_u_function,
                                                 boundary_values);

        // System matrix: M + theta^2 * dt^2 * A
        matrix_u.copy_from(mass_matrix);
        matrix_u.add(theta * theta * delta_t * delta_t, laplace_matrix);

        MatrixTools::apply_boundary_values(boundary_values,
                                           matrix_u,
                                           solution_u,
                                           system_rhs);
      }

      solve_u();

      // Assemble RHS for v equation:
      // M * v^{n-1} - theta*dt * A * u^n - (1-theta)*dt * A * u^{n-1}
      // + forcing_terms
      laplace_matrix.vmult(system_rhs, solution_u);
      system_rhs *= -theta * delta_t;

      mass_matrix.vmult(tmp, old_solution_v);
      system_rhs += tmp;

      laplace_matrix.vmult(tmp, old_solution_u);
      system_rhs.add(-delta_t * (1 - theta), tmp);

      system_rhs += forcing_terms;

      // Apply boundary conditions for v.
      {
        BoundaryValuesV boundary_values_v_function;
        boundary_values_v_function.set_time(time);

        std::map<types::global_dof_index, double> boundary_values;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 boundary_values_v_function,
                                                 boundary_values);

        matrix_v.copy_from(mass_matrix);
        MatrixTools::apply_boundary_values(boundary_values,
                                           matrix_v,
                                           solution_v,
                                           system_rhs);
      }

      solve_v();

      output_results();

      // Compute and output energy using the Newmark formula:
      // E = 0.5 * (v^T M v + u^T A u)
      Vector<double> Mv(solution_v.size());
      Vector<double> Au(solution_u.size());
      mass_matrix.vmult(Mv, solution_v);
      laplace_matrix.vmult(Au, solution_u);
      const double kinetic_energy = 0.5 * (solution_v * Mv);
      const double potential_energy = 0.5 * (solution_u * Au);
      const double total_energy = kinetic_energy + potential_energy;

      std::cout << "   Total energy: " << total_energy
                << " (kinetic: " << kinetic_energy
                << ", potential: " << potential_energy << ")" << std::endl;

      energy_file << time << "," << total_energy << ","
                  << kinetic_energy << "," << potential_energy << "\n";

      // Compute errors periodically (matching Newmark style)
      const int error_interval = 10; // Compute every N time steps
      if (exact_solution != nullptr && timestep_number % error_interval == 0)
      {
        const double error_L2 = compute_error(VectorTools::L2_norm, *exact_solution);
        const double error_H1 = compute_error(VectorTools::H1_norm, *exact_solution);

        error_file << time << "," << error_L2 << "," << error_H1 << "\n";

        std::cout << "   Time = " << time
                  << ", L2 error = " << error_L2
                  << ", H1 error = " << error_H1 << std::endl;
      }

      // Update old solutions.
      old_solution_u = solution_u;
      old_solution_v = solution_v;
    }

  if (error_file.is_open())
  {
    error_file.close();
  }
  if (energy_file.is_open())
  {
    energy_file.close();
  }

  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "Simulation complete." << std::endl;
}