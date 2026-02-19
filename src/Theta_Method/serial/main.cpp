#include "Wave.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

using TestCase = WaveEquation::TestCase;

int
main(int argc, char *argv[])
{
  try
    {
      // ============================================================
      // Declare parameters with defaults.
      // ============================================================
      ParameterHandler prm;

      prm.declare_entry("Test case",
                         "EX2",
                         Patterns::Selection("EX1|EX2|EX3|EX4"),
                         "Test case selector: EX1 (forced) or EX2 (free) or EX3 (non-homogeneous) or EX4 (square wave)");

      prm.declare_entry("Refinement",
                         "7",
                         Patterns::Integer(1),
                         "Number of global mesh refinements");

      prm.declare_entry("Degree",
                         "1",
                         Patterns::Integer(1),
                         "Polynomial degree of finite elements");

      prm.declare_entry("Final time",
                         "2.0",
                         Patterns::Double(0.0),
                         "Final simulation time T");

      prm.declare_entry("Time step",
                         "0.01",
                         Patterns::Double(0.0),
                         "Time step size delta_t");

      prm.declare_entry("Theta",
                         "0.5",
                         Patterns::Double(0.0, 1.0),
                         "Theta parameter (0.5 = Crank-Nicolson)");

      prm.declare_entry("Domain left",
                         "-1.0",
                         Patterns::Double(),
                         "Left boundary of the square domain");

      prm.declare_entry("Domain right",
                         "1.0",
                         Patterns::Double(),
                         "Right boundary of the square domain");

      // ============================================================
      // Parse .prm file if provided as command line argument.
      // ============================================================
      if (argc > 1)
        {
          std::cout << "Reading parameters from: " << argv[1] << std::endl;
          prm.parse_input(argv[1]);
        }
      else
        {
          std::cout << "No parameter file provided, using defaults."
                    << std::endl;
        }

      // ============================================================
      // Extract parameters.
      // ============================================================
      const TestCase     test_case =
        WaveEquation::parse_test_case(prm.get("Test case"));
      const unsigned int n_refine     = prm.get_integer("Refinement");
      const unsigned int degree       = prm.get_integer("Degree");
      const double       T            = prm.get_double("Final time");
      const double       delta_t      = prm.get_double("Time step");
      const double       theta        = prm.get_double("Theta");
      const double       domain_left  = prm.get_double("Domain left");
      const double       domain_right = prm.get_double("Domain right");

      // ============================================================
      // Print configuration.
      // ============================================================
      WaveEquation::print_test_case_info(test_case);
      std::cout << "Parameters: n_refine=" << n_refine
                << ", degree=" << degree
                << ", T=" << T
                << ", dt=" << delta_t
                << ", theta=" << theta
                << std::endl << std::endl;

      auto exact_solution = WaveEquation::create_exact_solution(test_case);

      WaveThetaSerial wave_equation(degree,
                                    T,
                                    theta,
                                    delta_t,
                                    domain_left,
                                    domain_right,
                                    n_refine,
                                    test_case);

      // ============================================================
      // Run with TimerOutput.
      // ============================================================
      Timer wall_clock;

      TimerOutput timer(std::cout,
                        TimerOutput::summary,
                        TimerOutput::wall_times);

      {
        TimerOutput::Scope t(timer, "Total simulation");
        wall_clock.start();
        wave_equation.run(exact_solution.get());
        wall_clock.stop();
      }


    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}