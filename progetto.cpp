#include "wave.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  constexpr unsigned int dim = wave::dim;

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);


  // chat dice che per usare davvero la f devo fare così
  wave::FunctionF forcing;
  const auto f = [&](const Point<dim> &p, const double &t)
  {
      forcing.set_time(t);
      return forcing.value(p);
  };


  wave problem(/*mesh_filename = */ "../mesh/mesh-square-10.msh",
               /* degree = */ 1,
               /* T = */ 1.0,
               /* delta_t = */ 0.05,
               f);

  problem.run();

  return 0;
}