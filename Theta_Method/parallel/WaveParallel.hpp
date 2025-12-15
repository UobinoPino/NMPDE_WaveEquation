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
 */
class WaveParallel
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Initial displacement condition.
  class InitialValuesU : public Function<dim>
  {
  public:
    InitialValuesU(const double amplitude_ = 10.0, const double sigma_ = 0.5)
      : amplitude(amplitude_)
      , sigma(sigma_)
    {}

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      // Radial distance from origin
      const double r = p.norm();

      // Gaussian bump centered at origin
      return amplitude * std::exp(-r * r / (2.0 * sigma * sigma));
    }

  private:
    const double amplitude;
    const double sigma;
  };

  // Initial velocity condition.
  class InitialValuesV : public Function<dim>
  {
  public:
    InitialValuesV() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      (void)p;
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      // Zero initial velocity - wave will spread naturally
      return 0.0;
    }
  };

  // Forcing term (right-hand side).
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide() = default;

    virtual double
    value(const Point<dim> & /*p*/,
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

    virtual double
    value(const Point<dim> & /*p*/,
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

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      return 0.0;  // Zero velocity at boundaries
    }
  };

  // Constructor.
  WaveParallel(const unsigned int degree_,
               const double       T_,
               const double       theta_,
               const double       delta_t_,
               const double       domain_left_  = -5.0,
               const double       domain_right_ = 5.0,
               const unsigned int n_refine_     = 7);

  // Run the time-dependent simulation.
  void
  run();

protected:
  // Initialization.
  void
  setup_system();

  // Assemble mass and laplace matrices.
  void
  assemble_matrices();

  // Assemble system for u equation.
  void
  assemble_system_u();

  // Assemble system for v equation.
  void
  assemble_system_v();

  // Solve for displacement u.
  void
  solve_u();

  // Solve for velocity v.
  void
  solve_v();

  // Output results.
  void
  output_results() const;

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

  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif // WAVE_PARALLEL_HPP