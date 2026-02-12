#include "Wave.hpp"
#include <deal.II/base/timer.h>

using TestCase = WaveEquation::TestCase;

int main()
{
    try
    {
      // ============================================================
      // SELECT TEST CASE: TestCase::EX1 or TestCase::EX2
      // ============================================================
      const TestCase test_case = TestCase::EX2;

        auto exact_solution = WaveEquation::create_exact_solution(test_case);

        WaveEquation::print_test_case_info(test_case);
        std::cout << std::endl;

        WaveThetaSerial wave_equation(/* degree = */ 1,
                                      /* T = */ 2.0,
                                      /* theta = */ 0.5,
                                      /* delta_t = */ 0.01,
                                      /* domain_left = */ -1.0,
                                      /* domain_right = */ 1.0,
                                      /* n_refine = */ 7,
                                      /* test_case = */ test_case);

        TimerOutput timer(std::cout,
                          TimerOutput::summary,
                          TimerOutput::wall_times);

        {
            TimerOutput::Scope t(timer, "Total simulation");
            wave_equation.run(exact_solution.get());
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