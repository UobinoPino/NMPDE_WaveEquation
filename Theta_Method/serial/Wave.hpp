#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_creator.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <functional>

using namespace dealii;

/**
 * Class managing the wave equation problem.
 * Solves: u_tt = c^2 * Laplacian(u) + f
 * Using the theta-method time discretization.
 */
class Wave
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  class InitialValuesU : public Function<dim>
  {
  public:
    InitialValuesU() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0);
    }
  };

  // Initial velocity: v(x,y,0) = 0 (since d/dt[cos(t)]|_{t=0} = 0)
  class InitialValuesV : public Function<dim>
  {
  public:
    InitialValuesV() = default;

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int component = 0) const override
    {
      (void)component;
      return 0.0;
    }
  };

  // Forcing term: f = (π²/2 - 1) * sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int component = 0) const override
    {
      (void)component;
      return (numbers::PI * numbers::PI / 2.0 - 1.0) *
             std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
             std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
             std::cos(this->get_time());
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
  Wave(const unsigned int degree_,
       const double       T_,
       const double       theta_,
       const double       delta_t_,
       const double       domain_left_  = -5.0,
       const double       domain_right_ = 5.0,
       const unsigned int n_refine_     = 7);

  // Run the time-dependent simulation.
  void
  run(Function<dim> *exact_solution = nullptr);

  double
  compute_error(const VectorTools::NormType &norm_type,
                Function<dim>               &exact_solution) const;

protected:
  // Initialization.
  void
  setup_system();

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

#endif // WAVE_HPP