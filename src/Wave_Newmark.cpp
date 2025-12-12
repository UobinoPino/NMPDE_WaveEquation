#include "Wave.hpp"

static constexpr unsigned int dim = Wave::dim;


// Exact solution.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation. u(x,y,t)=sin( pi*(x+1)/2 ) * sin( pi*(y+1)/2 ) * cos(t)
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return std::sin(M_PI * (p[0] + 1) / 2) * std::sin(M_PI * (p[1] + 1) / 2) *
           std::cos(this->get_time());
  }

  // Gradient evaluation.
  virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
           const unsigned int /*component*/ = 0) const override
  {
    Tensor<1, dim> result;
    
    const double time_factor = std::cos(this->get_time());

    // ∂u/∂x = π cos(πx) sin(πy) cos(t)
    result[0] = M_PI * std::cos(M_PI * p[0]) * std::sin(M_PI * p[1]) * time_factor;

    // ∂u/∂y = π sin(πx) cos(πy) cos(t)
    result[1] = M_PI * std::sin(M_PI * p[0]) * std::cos(M_PI * p[1]) * time_factor;

    return result;
  }
};

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // f(x,y,t) = (pi^2/2 - 1) * phi(x,y) * cos(t)
  const auto f  = [](const Point<dim>  &p, const double  &t) {
    return (2 * numbers::PI * numbers::PI - 1) *
             std::cos(numbers::PI * t) *
             std::sin(numbers::PI * (p[0] + 1) / 2) *
             std::sin(numbers::PI * (p[1] + 1) / 2);
    return 0.0;
  };

  ExactSolution exact_solution;

  Wave problem(/*mesh_filename = */ "../mesh/mesh-square-10.msh",
               /* degree = */ 1,
               /* T = */ 2.0,
               /* beta = */ 0.25,
               /* gamma = */ 0.5,
               /* delta_t = */ 0.01,
               f);

  problem.run(&exact_solution);

  return 0;
}