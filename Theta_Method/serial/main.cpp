#include "Wave.hpp"
static constexpr unsigned int dim = Wave::dim;

// Exact solution: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
class ExactSolution : public Function<dim>
{
public:
    ExactSolution() = default;

    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
        return std::sin(M_PI * (p[0] + 1.0) / 2.0) *
               std::sin(M_PI * (p[1] + 1.0) / 2.0) *
               std::cos(this->get_time());
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p,
             const unsigned int /*component*/ = 0) const override
    {
        Tensor<1, dim> result;
        const double time_factor = std::cos(this->get_time());

        // ∂u/∂x = (π/2) cos(π(x+1)/2) sin(π(y+1)/2) cos(t)
        result[0] = M_PI * 0.5 *
                    std::cos(M_PI * (p[0] + 1.0) / 2.0) *
                    std::sin(M_PI * (p[1] + 1.0) / 2.0) *
                    time_factor;

        // ∂u/∂y = (π/2) sin(π(x+1)/2) cos(π(y+1)/2) cos(t)
        result[1] = M_PI * 0.5 *
                    std::sin(M_PI * (p[0] + 1.0) / 2.0) *
                    std::cos(M_PI * (p[1] + 1.0) / 2.0) *
                    time_factor;

        return result;
    }
};


int
main()
{
    try
    {

        ExactSolution exact_solution;
        Wave wave_equation(/* degree = */ 1,
                           /* T = */ 2.0,
                           /* theta = */ 0.5,
                           /* delta_t = */ 0.01,
                           /* domain_left = */ -1.0,
                           /* domain_right = */ 1.0,
                           /* n_refine = */ 5);

        wave_equation.run(&exact_solution);
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}