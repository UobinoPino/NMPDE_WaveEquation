#include "Wave.hpp"

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/timer.h>

using TestCase = WaveEquation::TestCase;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  try
    {
      const unsigned int mpi_rank =
        Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const unsigned int mpi_size =
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

      ConditionalOStream pcout(std::cout, mpi_rank == 0);

      // ============================================================
      // Declare parameters with defaults.
      // ============================================================
      ParameterHandler prm;

      prm.declare_entry("Test case",
                         "EX2",
                         Patterns::Selection("EX1|EX2"),
                         "Test case selector: EX1 (forced) or EX2 (free)");

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

      prm.declare_entry("Beta",
                         "0.25",
                         Patterns::Double(0.0),
                         "Newmark beta parameter");

      prm.declare_entry("Gamma",
                         "0.5",
                         Patterns::Double(0.0),
                         "Newmark gamma parameter");

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
          pcout << "Reading parameters from: " << argv[1] << std::endl;
          prm.parse_input(argv[1]);
        }
      else
        {
          pcout << "No parameter file provided, using defaults." << std::endl;
        }

      // ============================================================
      // Extract parameters.
      // ============================================================
      const TestCase     test_case =
        WaveEquation::parse_test_case(prm.get("Test case"));
      const unsigned int n_refine    = prm.get_integer("Refinement");
      const unsigned int degree      = prm.get_integer("Degree");
      const double       T           = prm.get_double("Final time");
      const double       delta_t     = prm.get_double("Time step");
      const double       beta        = prm.get_double("Beta");
      const double       gamma       = prm.get_double("Gamma");
      const double       domain_left = prm.get_double("Domain left");
      const double       domain_right = prm.get_double("Domain right");

      // ============================================================
      // Print configuration.
      // ============================================================
      WaveEquation::print_test_case_info(test_case, pcout.get_stream());
      pcout << "Parameters: n_refine=" << n_refine
            << ", degree=" << degree
            << ", T=" << T
            << ", dt=" << delta_t
            << ", beta=" << beta
            << ", gamma=" << gamma
            << std::endl << std::endl;

      auto exact_solution = WaveEquation::create_exact_solution(test_case);

      WaveNewmark problem(domain_left,
                          domain_right,
                          n_refine,
                          degree,
                          T,
                          beta,
                          gamma,
                          delta_t,
                          test_case);

      // ============================================================
      // Run with TimerOutput.
      // ============================================================
      Timer wall_clock;

      TimerOutput timer(MPI_COMM_WORLD,
                        pcout,
                        TimerOutput::summary,
                        TimerOutput::wall_times);

      {
        TimerOutput::Scope t(timer, "Total simulation");
        wall_clock.start();
        problem.run(exact_solution.get());
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