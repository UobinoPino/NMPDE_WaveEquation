#include "Wave.hpp"
#include <chrono>

static constexpr unsigned int dim = Wave::dim;

int
main()
{
    try
    {
        // ============================================================
        // SELECT TEST CASE: Wave::EX1 or Wave::EX2
        // ============================================================
        // EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
        //      f = (π²/2 - 1) * sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)
        //
        // EX2: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(π/√2 * t)
        //      f = 0 (homogeneous wave equation)
        // ============================================================

        const Wave::TestCase test_case = Wave::EX2;  // Change to Wave::EX2 for first test case

        // Create appropriate exact solution based on test case
        std::unique_ptr<Function<dim>> exact_solution;
        if (test_case == Wave::EX1)
        {
            exact_solution = std::make_unique<Wave::ExactSolutionEX1>();
            std::cout << "Running EX1: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(t)" << std::endl;
            std::cout << "             f = (π²/2 - 1) * φ(x,y) * cos(t)" << std::endl;
        }
        else
        {
            exact_solution = std::make_unique<Wave::ExactSolutionEX2>();
            std::cout << "Running EX2: u(x,y,t) = sin(π(x+1)/2) * sin(π(y+1)/2) * cos(π/√2 * t)" << std::endl;
            std::cout << "             f = 0 (homogeneous)" << std::endl;
        }
        std::cout << std::endl;

        Wave wave_equation(/* degree = */ 1,
                           /* T = */ 2.0,
                           /* theta = */ 0.5,
                           /* delta_t = */ 0.01,
                           /* domain_left = */ -1.0,
                           /* domain_right = */ 1.0,
                           /* n_refine = */ 5,
                           /* test_case = */ test_case);

        auto start = std::chrono::high_resolution_clock::now();
        wave_equation.run(exact_solution.get());
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Ehi, fix your code please!" << std::endl
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
                  << "Ehi, fix your code please!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}