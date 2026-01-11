#include "WaveParallel.hpp"
#include <chrono>

using TestCase = WaveEquation::TestCase;

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  try
    {
      // ============================================================
      // SELECT TEST CASE: TestCase::EX1 or TestCase::EX2
      // ============================================================
      const TestCase test_case = TestCase::EX2;

      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const unsigned int mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

      // Create appropriate exact solution using factory function
      auto exact_solution = WaveEquation::create_exact_solution(test_case);

      if (mpi_rank == 0)
        {
          WaveEquation::print_test_case_info(test_case);
          std::cout << std::endl;
        }

      WaveThetaParallel wave_equation(/* degree = */ 1,
                                      /* T = */ 2.0,
                                      /* theta = */ 0.5,
                                      /* delta_t = */ 0.01,
                                      /* domain_left = */ -1.0,
                                      /* domain_right = */ 1.0,
                                      /* n_refine = */ 7,
                                      /* test_case = */ test_case);

      // Synchronize before timing
      MPI_Barrier(MPI_COMM_WORLD);
      auto start = std::chrono::high_resolution_clock::now();

      wave_equation.run(exact_solution.get());

      // Synchronize after computation
      MPI_Barrier(MPI_COMM_WORLD);
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed = end - start;

      if (mpi_rank == 0)
        {
          std::cout << "Total execution time: " << elapsed.count() << " seconds"
                    << std::endl;
          // Output for scalability script parsing
          std::cout << "SCALABILITY_RESULT,parallel," << mpi_size << ","
                    << elapsed.count() << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Fix your code please!" << std::endl
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
                << "Fix your code please!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
