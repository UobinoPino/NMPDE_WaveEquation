#include "WaveParallel.hpp"

int
main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    try
    {
        WaveParallel wave_equation(/* degree = */ 1,
                                   /* T = */ 2.0,
                                   /* theta = */ 0.5,
                                   /* delta_t = */ 1.0 / 64.0,
                                   /* domain_left = */ -5.0,
                                   /* domain_right = */ 5.0,
                                   /* n_refine = */ 7);

        wave_equation.run();
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