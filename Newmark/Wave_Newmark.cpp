#include "Wave.hpp"

static constexpr unsigned int dim = Wave::dim;


// Exact solution.
class ExactSolution : public Function<dim>
{
public:
  // Constructor.
  ExactSolution()
  {}

  // Evaluation.
  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    // // EX1: u(x,y,t) = sin( pi*(x+1)/2 ) * sin( pi*(y+1)/2 ) * cos(t)
    // return std::sin(M_PI * (p[0] + 1) / 2) * std::sin(M_PI * (p[1] + 1) / 2) *
    //        std::cos(this->get_time());
          
    // EX2: sin( pi*(x+1)/2 ) * sin( pi*(y+1)/2 ) * cos(pi / sqrt(2) * t)
    const double time_factor = std::cos(numbers::PI / std::sqrt(2) * this->get_time());
    return std::sin(numbers::PI * (p[0] + 1) / 2) *
           std::sin(numbers::PI * (p[1] + 1) / 2) *
           time_factor;

    // // EX?: u(x,y,t) = exp( -r(x,y)^2 ) * cos(t - r(x,y))
    // return std::exp( - (p[0]*p[0] + p[1]*p[1]) ) *
    //        std::cos( this->get_time() - std::sqrt(p[0]*p[0] + p[1]*p[1]) );
  }



    virtual Tensor<1, dim>
  gradient(const Point<dim> &p,
          const unsigned int = 0) const override
  {
    Tensor<1, dim> result;

    // // EX1: ∂u/∂x = π*0.5 cos(pi*(x+1)/2) sin(pi*(y+1)/2) cos(t)
    // const double time_factor = std::cos(this->get_time());
    // result[0] = M_PI * 0.5 * std::cos(M_PI * (p[0] + 1) / 2) * std::sin(M_PI * (p[1] + 1) / 2) * time_factor;
    // // EX1: ∂u/∂y = π*0.5 sin(pi*(x+1)/2) cos(pi*(y+1)/2) cos(t)
    // result[1] = M_PI * 0.5 * std::sin(M_PI * (p[0] + 1) / 2) * std::cos(M_PI * (p[1] + 1) / 2) * time_factor;

    // EX2: ∂u/∂x = π*0.5 cos(pi*(x+1)/2) sin(pi*(y+1)/2) * cos(pi/sqrt(2)*t)
    const double time_factor = std::cos(numbers::PI / std::sqrt(2) * this->get_time());
    result[0] = numbers::PI * 0.5 * std::cos(numbers::PI * (p[0] + 1) / 2) *
                std::sin(numbers::PI * (p[1] + 1) / 2) * time_factor;
    // EX2: ∂u/∂y = π*0.5 sin(pi*(x+1)/2) cos(pi*(y+1)/2) * cos(pi/sqrt(2)*t)
    result[1] = numbers::PI * 0.5 * std::sin(numbers::PI * (p[0] + 1) / 2) *
                std::cos(numbers::PI * (p[1] + 1) / 2) * time_factor;

    // // EX?: u_x(x,y,t) = ( x / r ) * exp( -r^2 ) * [ sin(t - r) - 2*r*cos(t - r) ]
    // // EX?: u_y(x,y,t) = ( y / r ) * exp( -r^2 ) * [ sin(t - r) - 2*r*cos(t - r) ]

    // const double r2 = p[0]*p[0] + p[1]*p[1];
    // const double r  = std::sqrt(r2);

    // if (r < 1e-12)
    //   {
    //     // Limite per r -> 0
    //     result[0] = 0.0;
    //     result[1] = 0.0;
    //   }
    // else
    //   {
    //     const double common_factor =
    //       std::exp(-r2) *
    //       ( std::sin(this->get_time() - r)
    //         - 2.0 * r * std::cos(this->get_time() - r) );

    //     result[0] = (p[0] / r) * common_factor;
    //     result[1] = (p[1] / r) * common_factor;
    //   }

    return result;
  }
};

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const auto f  = [](const Point<dim>  &p, const double  &t) {

    // // EX1: f(x,y,t) = (π²/2 - 1) * φ(x,y) * cos(t)
    // // where φ(x,y) = sin(π(x+1)/2) * sin(π(y+1)/2)
    
    // const double pi = numbers::PI;
    // const double phi = std::sin(pi * (p[0] + 1) / 2) *
    //                    std::sin(pi * (p[1] + 1) / 2);

    // return (pi * pi / 2.0 - 1.0) * phi * std::cos(t);

    // EX2: f(x,y,t) = (π²/2 - π²/2) * φ(x,y) * cos(π/√2 * t) = 0
    // where φ(x,y) = sin(π(x+1)/2) * sin(π(y+1)/2)
    return 0.0;

  //   // EX?: f(x,y,t) = [ 4*r^2 - 4 - 1 ] * exp( -r^2 ) * cos(t - r) + [ 4*r - 2/r ] * exp( -r^2 ) * sin(t - r)
  //   const double r2 = p[0]*p[0] + p[1]*p[1];
  //   const double r  = std::sqrt(r2);
  //   const double exp_factor = std::exp(-r2);
  //   const double cos_factor = std::cos(t - r);
  //   const double sin_factor = std::sin(t - r);

  //   if (r < 1e-5)
  //     {
  //       // limite per r -> 0
  //       return -5.0 * exp_factor * std::cos(t);
  //     }

  //   return (4.0*r*r - 5.0) * exp_factor * cos_factor
  //       + (4.0*r - 2.0/r) * exp_factor * sin_factor;
  };

  ExactSolution exact_solution;

  Wave problem( -1.0,    // domain_left
                1.0,     // domain_right  
                5,       // n_refine (number of global refinements - controls mesh precision)
               /* degree = */ 1,
               /* T = */ 1.0,
               /* beta = */ 0.25,
               /* gamma = */ 0.5,
               /* delta_t = */ 0.01,
               f);

  problem.run(&exact_solution);

  return 0;
}