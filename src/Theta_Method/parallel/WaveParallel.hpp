#ifndef WAVE_THETA_PARALLEL_HPP
#define WAVE_THETA_PARALLEL_HPP

#include "../../common/WaveTestCases.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>

using namespace dealii;

/**
 * Parallel wave equation solver using the θ-method with MPI.
 *
 * Reformulates the wave equation as a first-order system in (u, v):
 *   u_t = v
 *   v_t = Δu + f
 *
 * Parameters:
 *   θ = 0.5 (Crank-Nicolson scheme, second-order accurate)
 *
 * Features:
 *   - MPI parallel implementation using Trilinos
 *   - Uses parallel::fullydistributed::Triangulation for mesh distribution
 *   - Solves two linear systems per time step (one for u, one for v)
 */
class WaveThetaParallel
{
public:
  // Use dimension from common header
  static constexpr unsigned int dim = WaveEquation::dim;

  // Use TestCase from common namespace
  using TestCase = WaveEquation::TestCase;

  /**
   * Constructor.
   *
   * @param degree_       Polynomial degree for finite elements
   * @param T_            Final simulation time
   * @param theta_        θ parameter (default: 0.5 for Crank-Nicolson)
   * @param delta_t_      Time step size
   * @param domain_left_  Left boundary of the square domain
   * @param domain_right_ Right boundary of the square domain
   * @param n_refine_     Number of global mesh refinements
   * @param test_case_    Test case selector (EX1 or EX2)
   */
  WaveThetaParallel(const unsigned int degree_,
                    const double       T_,
                    const double       theta_,
                    const double       delta_t_,
                    const double       domain_left_  = -1.0,
                    const double       domain_right_ = 1.0,
                    const unsigned int n_refine_     = 5,
                    const TestCase     test_case_    = TestCase::EX1);

  /**
   * Run the time-dependent simulation.
   *
   * @param exact_solution  Pointer to exact solution for error computation (optional)
   */
  void run(Function<dim> *exact_solution = nullptr);

  /**
   * Compute the error against a given exact solution.
   */
  double compute_error(const VectorTools::NormType &norm_type,
                       Function<dim>               &exact_solution) const;

  /**
   * Compute total energy: E = 0.5 * (v^T M v + u^T A u)
   */
  double compute_total_energy() const;

protected:
  // Initialization.
  void setup_system();

  // Assemble mass and laplace matrices.
  void assemble_matrices();

  // Assemble forcing term vector.
  void assemble_forcing_terms();

  // Assemble system for u equation.
  void assemble_system_u();

  // Assemble system for v equation.
  void assemble_system_v();

  // Solve for displacement u.
  void solve_u();

  // Solve for velocity v.
  void solve_v();

  // Output results.
  void output_results() const;

  // Polynomial degree.
  const unsigned int degree;

  // Final time.
  const double T;

  // Theta parameter for the theta method.
  const double theta;

  // Time step.
  const double delta_t;

  // Domain boundaries.
  const double domain_left;
  const double domain_right;

  // Number of global refinements.
  const unsigned int n_refine;

  // Test case selector.
  const TestCase test_case;

  // Current time.
  double time;

  // Current timestep number.
  unsigned int timestep_number;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // Rank of current MPI process.
  const unsigned int mpi_rank;

  // Triangulation (parallel fully distributed).
  parallel::fullydistributed::Triangulation<dim> triangulation;

  // Finite element space.
  std::unique_ptr<FE_Q<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<QGauss<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Locally owned and relevant DoFs.
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // Mass matrix.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Laplace (stiffness) matrix.
  TrilinosWrappers::SparseMatrix laplace_matrix;

  // System matrix for current equation.
  TrilinosWrappers::SparseMatrix system_matrix;

  // Solution vectors (owned, without ghost elements).
  TrilinosWrappers::MPI::Vector solution_u_owned;
  TrilinosWrappers::MPI::Vector solution_v_owned;
  TrilinosWrappers::MPI::Vector old_solution_u_owned;
  TrilinosWrappers::MPI::Vector old_solution_v_owned;

  // Solution vectors (with ghost elements for reading).
  TrilinosWrappers::MPI::Vector solution_u;
  TrilinosWrappers::MPI::Vector solution_v;
  TrilinosWrappers::MPI::Vector old_solution_u;
  TrilinosWrappers::MPI::Vector old_solution_v;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // Temporary vectors.
  TrilinosWrappers::MPI::Vector tmp_owned;
  TrilinosWrappers::MPI::Vector forcing_terms_owned;

  // ----- Dispersion analysis: center point tracking -----

  // File stream for center point time series.
  std::ofstream center_point_file;

  // DoF index corresponding to the center point (0,0).
  types::global_dof_index center_dof_index;

  // Flag indicating if center point is owned by this MPI process.
  bool center_point_is_local;

  // Find the DoF closest to the center point.
  void find_center_point_dof();

  // Record solution value at center point.
  void record_center_point_value();

  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif // WAVE_THETA_PARALLEL_HPP
