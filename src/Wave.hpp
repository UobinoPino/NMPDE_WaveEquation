#ifndef WAVE_HPP
#define WAVE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

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
 * Class managing the differential problem.
 */
class Wave
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 2;

  // Initial condition. u0​=sin( pi*(x+1)/2 ) * sin( pi*(y+1)/2 )
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0() = default;

    // Evaluation of the function.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // const double r = p.norm();
      // const double amplitude = 10.0;
      // const double sigma = 0.5;

      // return amplitude * std::exp(-(r*r)/ (2.0 * sigma * sigma));

      return std::sin(numbers::PI * (p[0] + 1) / 2) *
             std::sin(numbers::PI * (p[1] + 1) / 2);
    }
  };

    // Initial condition. u1​=0
  class FunctionU1 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU1() = default;

    // Evaluation of the function.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

      // Initial condition. u2​=(1/(βΔt^2))(u0​)- (1/(βΔt))u1​
  class FunctionU2 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU2() = default;

    // Evaluation of the function.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
      // return (1/(0.25 * 0.01 * 0.01)) * // beta=0.25, delta_t=0.01
      //        std::sin(numbers::PI * p[0]) *
      //        std::sin(numbers::PI * p[1]);
    }
  };

  // Dirichlet boundary function.
  //
  // This is implemented as a dealii::Function<dim>, instead of e.g. a lambda
  // function, because this allows to use dealii boundary utilities directly.
  class FunctionG : public Function<dim>
  {
  public:
    // Constructor.
    FunctionG()
    {}

    // Evaluation.
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // // Forcing term. f=(2π2-1) cos(t) sin(πx)sin(πy) 
  // class FunctionF : public Function<dim>
  // {
  // public:
  //   // Constructor.
  //   FunctionF()
  //   {}

  //   // Evaluation.
  //   virtual double
  //   value(const Point<dim> &p,
  //         const unsigned int /*component*/ = 0) const override
  //   {
  //     return (2 * numbers::PI * numbers::PI - 1) *
  //            std::cos(numbers::PI * this->get_time()) *
  //            std::sin(numbers::PI * p[0]) *
  //            std::sin(numbers::PI * p[1]);
  //   }
  // };

  // Constructor.
  Wave(const std::string                               &mesh_file_name_,
       const unsigned int                              &r_,
       const double                                    &T_,
       const double                                    &beta_,
       const double                                    &gamma_,
       const double                                    &delta_t_,
       const std::function<double(const Point<dim> &, const double &)> &f_)
    : mesh_file_name(mesh_file_name_)
    , r(r_)
    , T(T_)
    , beta(beta_)
    , gamma(gamma_)
    , delta_t(delta_t_)
    , f(f_) // qui voglio usare FunctionF
    , mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Run the time-dependent simulation.
  void
  run(Function<dim> *exact_solution = nullptr);

  // Compute the error against a given exact solution.
  double
  compute_error(const VectorTools::NormType &norm_type,
                Function<dim>         &exact_solution) const;

protected:
  // Initialization.
  void
  setup();

  // System assembly.
  void
  assemble();

  // System solution.
  void
  solve_linear_system();

  // Output.
  void
  output() const;

  // Name of the mesh.
  const std::string mesh_file_name;

  // Polynomial degree.
  const unsigned int r;

  // Final time.
  const double T;

  // Newmark parameters
  const double beta; // 0.25
  const double gamma; // 0.5

  // Time step.
  const double delta_t;

  // Current time.
  double time = 0.0;

  // Current timestep number.
  unsigned int timestep_number = 0;

  // Forcing term. NON LO VOGLIO - forse si, cosa cambia tra lui e FunctionU0
  std::function<double(const Point<dim> &, const double &)> f;

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

  // no quadrature boundary perchè non abbiamo condizioni di neumann

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

  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif