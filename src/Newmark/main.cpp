#include "Wave.hpp"
#include <chrono>

static constexpr unsigned int dim = Wave::dim;

// Main function.
int main(int argc, char *argv[]) {
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);


    try
    {
      // ============================================================
      // SELECT TEST CASE: Wave::EX1 or Wave::EX2
      // ============================================================
      const Wave::TestCase test_case = Wave::EX2;  // Change to Wave::EX1 for first test case

      const unsigned int mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
      const unsigned int mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

      // Create appropriate exact solution based on test case
      std::unique_ptr<Function<dim>> exact_solution;

      if (test_case == Wave::EX1)
        {
          exact_solution = std::make_unique<Wave::ExactSolutionEX1>();
          if (mpi_rank == 0)
            {
              std::cout << "Running EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)" << std::endl;
              std::cout << "             f = (π²/2 - 1) * φ(x,y) * cos(t)" << std::endl;
            }
        }
      else
        {
          exact_solution = std::make_unique<Wave::ExactSolutionEX2>();
          if (mpi_rank == 0)
            {
              std::cout << "Running EX2: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(π/√2 * t)" << std::endl;
              std::cout << "             f = 0 (homogeneous)" << std::endl;
            }
        }

      if (mpi_rank == 0)
        std::cout << std::endl;

      Wave problem(/* domain_left = */ -1.0,
                   /* domain_right = */ 1.0,
                   /* n_refine = */ 1,
                   /* degree = */ 1,
                   /* T = */ 2.0,
                   /* beta = */ 0.25, //0.25
                   /* gamma = */ 0.5,
                   /* delta_t = */ 0.01,
                   /* test_case = */ test_case);

      // Synchronize before timing
      MPI_Barrier(MPI_COMM_WORLD);
      auto start = std::chrono::high_resolution_clock::now();

      problem.run(exact_solution.get());

      // Synchronize after computation
      MPI_Barrier(MPI_COMM_WORLD);
      auto end = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double> elapsed = end - start;

      if (mpi_rank == 0)
        {
          std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
          std::cout << "SCALABILITY_RESULT,newmark," << mpi_size << "," << elapsed.count() << std::endl;
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Fix your code please!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Fix your code please!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;



}