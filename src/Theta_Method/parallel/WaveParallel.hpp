#ifndef WAVE_PARALLEL_HPP
#define WAVE_PARALLEL_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <functional>

using namespace dealii;

/**
 * Parallel class managing the wave equation problem using MPI.
 * Solves: u_tt = c^2 * Laplacian(u) + f
 * Using the theta-method time discretization.
 *
 * Supports two test cases:
 * EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
 *      f = (π²/2 - 1) * sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
 *
 * EX2: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(π/√2 * t)
 *      f = 0 (homogeneous)
 */
class WaveParallel
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Test case selector: 1 for EX1, 2 for EX2
  enum TestCase { EX1 = 1, EX2 = 2 };

  // Initial displacement condition.
  // EX1 = EX2: u0 = sin(π(x+1)/2) * sin(π(y+1)/2)
  class InitialValuesU : public Function<dim>
  {
  public:
    InitialValuesU() = default;

    virtual double value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0);
    }
  };

  // Initial velocity condition.
  // EX1 = EX2: v0 = 0
  class InitialValuesV : public Function<dim>
  {
  public:
    InitialValuesV() = default;

    virtual double value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      (void)p;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return 0.0;
    }
  };

  // Forcing term for EX1: f = (π²/2 - 1) * sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
  class RightHandSideEX1 : public Function<dim>
  {
  public:
    RightHandSideEX1() = default;

    virtual double value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return (numbers::PI * numbers::PI / 2.0 - 1.0) *
             std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
             std::cos(this->get_time());
    }
  };

  // Forcing term for EX2: f = 0 (homogeneous wave equation)
  class RightHandSideEX2 : public Function<dim>
  {
  public:
    RightHandSideEX2() = default;

    virtual double value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0.0;
    }
  };

  // Boundary values for displacement u.
  class BoundaryValuesU : public Function<dim>
  {
  public:
    BoundaryValuesU() = default;

    virtual double value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0.0;  // Zero displacement at boundaries
    }
  };

  // Boundary values for velocity v.
  class BoundaryValuesV : public Function<dim>
  {
  public:
    BoundaryValuesV() = default;

    virtual double value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0.0;  // Zero velocity at boundaries
    }
  };

  // Exact solution for EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
  class ExactSolutionEX1 : public Function<dim>
  {
  public:
    ExactSolutionEX1() = default;

    virtual double value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
             std::cos(this->get_time());
    }

    virtual Tensor<1, dim> gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      const double time_factor = std::cos(this->get_time());

      // ∂u/∂x = (π/2) cos(π(x+1)/2) sin(π(y+1)/2) cos(t)
      result[0] = numbers::PI * 0.5 *
                  std::cos(numbers::PI * (p[0] + 1.0) / 2.0) *
                  std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
                  time_factor;

      // ∂u/∂y = (π/2) sin(π(x+1)/2) cos(π(y+1)/2) cos(t)
      result[1] = numbers::PI * 0.5 *
                  std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
                  std::cos(numbers::PI * (p[1] + 1.0) / 2.0) *
                  time_factor;

      return result;
    }
  };

  // Exact solution for EX2: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(π/√2 * t)
  class ExactSolutionEX2 : public Function<dim>
  {
  public:
    ExactSolutionEX2() = default;

    virtual double value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      const double omega = numbers::PI / std::sqrt(2.0);
      return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
             std::cos(omega * this->get_time());
    }

    virtual Tensor<1, dim> gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
      Tensor<1, dim> result;
      const double omega = numbers::PI / std::sqrt(2.0);
      const double time_factor = std::cos(omega * this->get_time());

      // ∂u/∂x = (π/2) cos(π(x+1)/2) sin(π(y+1)/2) cos(π/√2 * t)
      result[0] = numbers::PI * 0.5 *
                  std::cos(numbers::PI * (p[0] + 1.0) / 2.0) *
                  std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
                  time_factor;

      // ∂u/∂y = (π/2) sin(π(x+1)/2) cos(π(y+1)/2) cos(π/√2 * t)
      result[1] = numbers::PI * 0.5 *
                  std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
                  std::cos(numbers::PI * (p[1] + 1.0) / 2.0) *
                  time_factor;

      return result;
    }
  };

  // Constructor.
  WaveParallel(const unsigned int degree_,
               const double       T_,
               const double       theta_,
               const double       delta_t_,
               const double       domain_left_  = -1.0,
               const double       domain_right_ = 1.0,
               const unsigned int n_refine_     = 5,
               const TestCase     test_case_    = EX1);

  // Run the time-dependent simulation.
  void run(Function<dim> *exact_solution = nullptr);

  // Compute error against exact solution.
  double compute_error(const VectorTools::NormType &norm_type,
                Function<dim>               &exact_solution) const;

  // Compute total energy: E = 0.5 * (v^T M v + u^T A u)
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

  // ----- Dispersion analysis:  center point tracking -----

  // File stream for center point time series
  std:: ofstream center_point_file;

  // DoF index corresponding to the center point (0,0)
  types::global_dof_index center_dof_index;

  // Flag indicating if center point is owned by this MPI process
  bool center_point_is_local;

  // Find the DoF closest to a given point
  void find_center_point_dof();

  // Record solution value at center point
  void record_center_point_value();


  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif // WAVE_PARALLEL_HPP