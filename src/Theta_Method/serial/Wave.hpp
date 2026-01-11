#ifndef WAVE_THETA_SERIAL_HPP
#define WAVE_THETA_SERIAL_HPP

#include "../../common/WaveTestCases.hpp"

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>

using namespace dealii;

/**
 * Serial wave equation solver using the θ-method.
 *
 * Reformulates the wave equation as a first-order system in (u, v):
 *   u_t = v
 *   v_t = Δu + f
 *
 * Parameters:
 *   θ = 0.5 (Crank-Nicolson scheme, second-order accurate)
 *
 * Features:
 *   - Sequential implementation
 *   - Solves two linear systems per time step (one for u, one for v)
 */
class WaveThetaSerial
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
  WaveThetaSerial(const unsigned int degree_,
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

  // Triangulation.
  Triangulation<dim> triangulation;

  // Finite element space.
  FE_Q<dim> fe;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // Constraints.
  AffineConstraints<double> constraints;

  // Sparsity pattern.
  SparsityPattern sparsity_pattern;

  // Mass matrix.
  SparseMatrix<double> mass_matrix;

  // Laplace (stiffness) matrix.
  SparseMatrix<double> laplace_matrix;

  // System matrix for u equation.
  SparseMatrix<double> matrix_u;

  // System matrix for v equation.
  SparseMatrix<double> matrix_v;

  // Solution vectors.
  Vector<double> solution_u;
  Vector<double> solution_v;
  Vector<double> old_solution_u;
  Vector<double> old_solution_v;

  // System right-hand side.
  Vector<double> system_rhs;
};

#endif // WAVE_THETA_SERIAL_HPP
