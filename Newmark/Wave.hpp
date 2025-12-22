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
#include <deal.II/fe/fe_q.h>

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

  // Test case selector: 1 for EX1, 2 for EX2
  enum TestCase { EX1 = 1, EX2 = 2 };

  // Initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU0() = default;

    // Evaluation of the function.
    virtual double
    value([[maybe_unused]] const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // const double r = p.norm();
      // const double amplitude = 10.0;
      // const double sigma = 0.5;

      // return amplitude * std::exp(-(r*r)/ (2.0 * sigma * sigma));

      // EX1 = EX2: u0​=sin( pi*(x+1)/2 ) * sin( pi*(y+1)/2 ) 
      return std::sin(numbers::PI * (p[0] + 1) / 2) *
             std::sin(numbers::PI * (p[1] + 1) / 2);
    }
  };

    // Initial condition
  class FunctionU1 : public Function<dim>
  {
  public:
    // Constructor.
    FunctionU1() = default;

    // Evaluation of the function.
    virtual double
    value([[maybe_unused]] const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // EX1 = EX2: u1​=0.0
      return 0.0;
    }
  };

  // Initial acceleration for EX1: a0 = -sin(π(x+1)/2) * sin(π(y+1)/2)
  class FunctionU2_EX1 : public Function<dim>
  {
  public:
    FunctionU2_EX1() = default;

    virtual double value([[maybe_unused]] const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // EX1: u_tt(0) = -cos(0) * φ(x,y) = -φ(x,y)
      return -(std::sin(numbers::PI * (p[0] + 1) / 2) *
               std::sin(numbers::PI * (p[1] + 1) / 2));
    }
  };

  // Initial acceleration for EX2: a0 = -π²/2 * sin(π(x+1)/2) * sin(π(y+1)/2)
  class FunctionU2_EX2 : public Function<dim>
  {
  public:
    FunctionU2_EX2() = default;

    virtual double value([[maybe_unused]] const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // EX2: u_tt(0) = -(π/√2)² * φ(x,y) = -π²/2 * φ(x,y)
      return -(numbers::PI * numbers::PI * 0.5) *
             (std::sin(numbers::PI * (p[0] + 1) / 2) *
              std::sin(numbers::PI * (p[1] + 1) / 2));
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
    value([[maybe_unused]] const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      // EX1 = EX2: g=0.0
      return 0.0;
    }
  };

  // Forcing term for EX1: f = (π²/2 - 1) * sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
  class RightHandSideEX1 : public Function<dim>
  {
  public:
    RightHandSideEX1() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
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

    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Exact solution for EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
class ExactSolutionEX1 : public Function<dim>
{
public:
  ExactSolutionEX1() = default;

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
           std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
           std::cos(this->get_time());
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
    const double time_factor = std::cos(this->get_time());

    result[0] = numbers::PI * 0.5 *
                std::cos(numbers::PI * (p[0] + 1.0) / 2.0) *
                std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
                time_factor;

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

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    const double omega = numbers::PI / std::sqrt(2.0);
    return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
           std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
           std::cos(omega * this->get_time());
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
    const double omega = numbers::PI / std::sqrt(2.0);
    const double time_factor = std::cos(omega * this->get_time());

    result[0] = numbers::PI * 0.5 *
                std::cos(numbers::PI * (p[0] + 1.0) / 2.0) *
                std::sin(numbers::PI * (p[1] + 1.0) / 2.0) *
                time_factor;

    result[1] = numbers::PI * 0.5 *
                std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
                std::cos(numbers::PI * (p[1] + 1.0) / 2.0) *
                time_factor;

    return result;
  }
};



  Wave(const double                                   &domain_left_,
      const double                                    &domain_right_,
      const unsigned int                              &n_refine_,
      const unsigned int                              &r_,
      const double                                    &T_,
      const double                                    &beta_,
      const double                                    &gamma_,
      const double                                    &delta_t_,
      const TestCase                                  test_case_ = EX1)
    : domain_left(domain_left_)
    , domain_right(domain_right_)
    , n_refine(n_refine_)
    , r(r_)
    , T(T_)
    , beta(beta_)
    , gamma(gamma_)
    , delta_t(delta_t_)
    , test_case(test_case_)
    , mpi_size(Utilities:: MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , mesh(MPI_COMM_WORLD)
    , pcout(std::cout, mpi_rank == 0)
  {}

  // Run the time-dependent simulation.
  void run(Function<dim> *exact_solution = nullptr);

  // Compute the error against a given exact solution.
  double compute_error(const VectorTools::NormType &norm_type,
                Function<dim>         &exact_solution) const;

protected:
  // Initialization.
  void setup();

  // System assembly.
  void assemble();

  // System solution.
  void solve_linear_system();

  // Output.
  void output() const;

  // Compute total energy:  E = 0.5 * (v^T M v + u^T A u)
  double compute_total_energy() const;

  // // Name of the mesh.
  // const std::string mesh_file_name;

  // Domain boundaries for hyper cube. 
  const double domain_left;
  const double domain_right;

  // Number of global refinements.
  const unsigned int n_refine;

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

  // Mass and stiffness matrices for energy computation
  TrilinosWrappers::SparseMatrix mass_matrix;
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Output stream for process 0.
  ConditionalOStream pcout;
};

#endif