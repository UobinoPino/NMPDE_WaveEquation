#ifndef WAVE_NEWMARK_HPP
#define WAVE_NEWMARK_HPP

#include "../common/WaveTestCases.hpp"

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <fstream>
#include <iostream>

using namespace dealii;

/**
 * Wave equation solver using the Newmark-β method.
 *
 * This is a direct second-order formulation that advances displacement,
 * velocity, and acceleration simultaneously.
 *
 * Parameters:
 *   β = 0.25, γ = 0.5 (average acceleration, unconditionally stable)
 *
 * Features:
 *   - MPI parallel implementation
 *   - Energy-conserving for the chosen parameters
 *   - Solves one linear system per time step
 */
class WaveNewmark
{
public:
  // Use dimension from common header
  static constexpr unsigned int dim = WaveEquation::dim;

  // Use TestCase from common namespace
  using TestCase = WaveEquation::TestCase;

  /**
   * Constructor.
   *
   * @param domain_left_   Left boundary of the square domain
   * @param domain_right_  Right boundary of the square domain
   * @param n_refine_      Number of global mesh refinements
   * @param r_             Polynomial degree for finite elements
   * @param T_             Final simulation time
   * @param beta_          Newmark β parameter (default: 0.25)
   * @param gamma_         Newmark γ parameter (default: 0.5)
   * @param delta_t_       Time step size
   * @param test_case_     Test case selector (EX1 or EX2)
   */
  WaveNewmark(const double       &domain_left_,
              const double       &domain_right_,
              const unsigned int &n_refine_,
              const unsigned int &r_,
              const double       &T_,
              const double       &beta_,
              const double       &gamma_,
              const double       &delta_t_,
              const TestCase      test_case_ = TestCase::EX1)
    : domain_left(domain_left_)
    , domain_right(domain_right_)
    , n_refine(n_refine_)
    , r(r_)
    , T(T_)
    , beta(beta_)
    , gamma(gamma_)
    , delta_t(delta_t_)
    , test_case(test_case_)
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  /**
   * Run the time-dependent simulation.
   *
   * @param exact_solution  Pointer to exact solution for error computation (optional)
   */
  void run(Function<dim> *exact_solution = nullptr);

  /**
   * Compute the error against a given exact solution.
   *
   * @param norm_type       Type of norm (L2 or H1)
   * @param exact_solution  Reference to exact solution function
   * @return                The computed error norm
   */
  double compute_error(const VectorTools::NormType &norm_type,
                       Function<dim>               &exact_solution) const;

protected:
  // Initialization.
  void setup();

  // System assembly.
  void assemble();

  // System solution.
  void solve_linear_system();

  // Output.
  void output() const;

  // Compute total energy: E = 0.5 * (v^T M v + u^T A u)
  double compute_total_energy() const;

  // Domain boundaries for hyper cube.
  const double domain_left;
  const double domain_right;

  // Number of global refinements.
  const unsigned int n_refine;

  // Polynomial degree.
  const unsigned int r;

  // Final time.
  const double T;

  // Newmark parameters.
  const double beta;  // 0.25 for average acceleration
  const double gamma; // 0.5 for average acceleration

  // Time step.
  const double delta_t;

  // Current time.
  double time = 0.0;

  // Current timestep number.
  unsigned int timestep_number = 0;

  // Test case selector.
  const TestCase test_case;

  // Number of MPI processes.
  const unsigned int mpi_size;

  // Rank of the current MPI process.
  const unsigned int mpi_rank;

  // Triangulation.
  parallel::fullydistributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // System matrix.
  TrilinosWrappers::SparseMatrix system_matrix;

  // System right-hand side.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution, without ghost elements.
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution, with ghost elements.
  TrilinosWrappers::MPI::Vector solution;

  // Velocity, without ghost elements.
  TrilinosWrappers::MPI::Vector velocity_owned;

  // Velocity, with ghost elements.
  TrilinosWrappers::MPI::Vector velocity;

  // Acceleration, without ghost elements.
  TrilinosWrappers::MPI::Vector acceleration_owned;

  // Acceleration, with ghost elements.
  TrilinosWrappers::MPI::Vector acceleration;

  // Mass and stiffness matrices for energy computation.
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix stiffness_matrix;

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

#endif // WAVE_NEWMARK_HPP
