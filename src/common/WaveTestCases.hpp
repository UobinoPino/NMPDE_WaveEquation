#ifndef WAVE_TEST_CASES_HPP
#define WAVE_TEST_CASES_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/numbers.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace dealii;

/**
 * Common namespace for Wave Equation test cases and functions.
 *
 * This header centralizes all test case definitions, initial conditions,
 * boundary conditions, forcing terms, and exact solutions used by both
 * the Newmark and Theta-method solvers.
 *
 * Mathematical Problem:
 *   ∂²u/∂t² = Δu + f(x,y,t)    in Ω × (0, T]
 *   u = 0                       on ∂Ω (Dirichlet BC)
 *   u(x,y,0) = u₀(x,y)         (initial displacement)
 *   ∂u/∂t(x,y,0) = v₀(x,y)     (initial velocity)
 *
 * Domain: Ω = [-1, 1]²
 *
 * Test Cases:
 *   EX1: u(x,y,t) = sin(π(x+1)/2) · sin(π(y+1)/2) · cos(t)
 *        f = (π²/2 - 1) · φ(x,y) · cos(t)
 *
 *   EX2: u(x,y,t) = sin(π(x+1)/2) · sin(π(y+1)/2) · cos(π/√2 · t)
 *        f = 0 (homogeneous)
 */


namespace WaveEquation
{

// Physical dimension
static constexpr unsigned int dim = 2;

/**
 * Test case selector enumeration.
 */
enum class TestCase
{
  EX1 = 1,  // Forced vibration with cos(t)
  EX2 = 2   // Free vibration (homogeneous) with cos(π/√2 · t)
};

/**
 * Returns the test case name as a string.
 */
inline std::string test_case_name(TestCase test_case)
{
  return (test_case == TestCase::EX1) ? "EX1" : "EX2";
}

/**
 * Parses a test case from a string ("EX1" or "EX2").
 * Throws std::runtime_error if the string is not recognized.
 */
inline TestCase parse_test_case(const std::string &name)
{
  if (name == "EX1")
    return TestCase::EX1;
  else if (name == "EX2")
    return TestCase::EX2;
  else
    throw std::runtime_error("Unknown test case: '" + name +
                             "'. Valid options are: EX1, EX2");
}

// ============================================================================
// Spatial Mode Function (common to all test cases)
// ============================================================================

/**
 * Computes the spatial mode φ(x,y) = sin(π(x+1)/2) · sin(π(y+1)/2)
 * This is the fundamental eigenfunction for the domain [-1,1]² with
 * homogeneous Dirichlet boundary conditions.
 */
inline double spatial_mode(const Point<dim> &p)
{
  return std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
         std::sin(numbers::PI * (p[1] + 1.0) / 2.0);
}

/**
 * Computes the gradient of the spatial mode φ(x,y).
 */
inline Tensor<1, dim> spatial_mode_gradient(const Point<dim> &p)
{
  Tensor<1, dim> result;

  // ∂φ/∂x = (π/2) cos(π(x+1)/2) sin(π(y+1)/2)
  result[0] = numbers::PI * 0.5 *
              std::cos(numbers::PI * (p[0] + 1.0) / 2.0) *
              std::sin(numbers::PI * (p[1] + 1.0) / 2.0);

  // ∂φ/∂y = (π/2) sin(π(x+1)/2) cos(π(y+1)/2)
  result[1] = numbers::PI * 0.5 *
              std::sin(numbers::PI * (p[0] + 1.0) / 2.0) *
              std::cos(numbers::PI * (p[1] + 1.0) / 2.0);

  return result;
}

// ============================================================================
// Initial Conditions
// ============================================================================

/**
 * Initial displacement: u₀(x,y) = sin(π(x+1)/2) · sin(π(y+1)/2)
 * Common to both EX1 and EX2.
 */
class InitialDisplacement : public Function<dim>
{
public:
  InitialDisplacement() = default;

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return spatial_mode(p);
  }
};

/**
 * Initial velocity: v₀(x,y) = 0
 * Common to both EX1 and EX2 (since d/dt[cos(ωt)]|_{t=0} = 0 for any ω).
 */
class InitialVelocity : public Function<dim>
{
public:
  InitialVelocity() = default;

  virtual double value(const Point<dim> & /*p*/,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0.0;
  }
};

/**
 * Initial acceleration for EX1: a₀(x,y) = -φ(x,y)
 * Derived from u_tt(x,y,0) = -cos(0) · φ(x,y) = -φ(x,y)
 */
class InitialAccelerationEX1 : public Function<dim>
{
public:
  InitialAccelerationEX1() = default;

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return -spatial_mode(p);
  }
};

/**
 * Initial acceleration for EX2: a₀(x,y) = -π²/2 · φ(x,y)
 * Derived from u_tt(x,y,0) = -(π/√2)² · φ(x,y) = -π²/2 · φ(x,y)
 */
class InitialAccelerationEX2 : public Function<dim>
{
public:
  InitialAccelerationEX2() = default;

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return -(numbers::PI * numbers::PI * 0.5) * spatial_mode(p);
  }
};

// ============================================================================
// Boundary Conditions
// ============================================================================

/**
 * Homogeneous Dirichlet boundary condition: g(x,y,t) = 0
 * Used for both displacement and velocity at boundaries.
 */
class HomogeneousDirichletBC : public Function<dim>
{
public:
  HomogeneousDirichletBC() = default;

  virtual double value(const Point<dim> & /*p*/,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0.0;
  }
};

// ============================================================================
// Forcing Terms
// ============================================================================

/**
 * Forcing term for EX1: f(x,y,t) = (π²/2 - 1) · φ(x,y) · cos(t)
 */
class ForcingTermEX1 : public Function<dim>
{
public:
  ForcingTermEX1() = default;

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return (numbers::PI * numbers::PI / 2.0 - 1.0) *
           spatial_mode(p) *
           std::cos(this->get_time());
  }
};

/**
 * Forcing term for EX2: f(x,y,t) = 0 (homogeneous wave equation)
 */
class ForcingTermEX2 : public Function<dim>
{
public:
  ForcingTermEX2() = default;

  virtual double value(const Point<dim> & /*p*/,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0.0;
  }
};

// ============================================================================
// Exact Solutions
// ============================================================================

/**
 * Exact solution for EX1: u(x,y,t) = φ(x,y) · cos(t)
 */
class ExactSolutionEX1 : public Function<dim>
{
public:
  ExactSolutionEX1() = default;

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return spatial_mode(p) * std::cos(this->get_time());
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
           const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const double time_factor = std::cos(this->get_time());
    return time_factor * spatial_mode_gradient(p);
  }
};

/**
 * Exact solution for EX2: u(x,y,t) = φ(x,y) · cos(π/√2 · t)
 */
class ExactSolutionEX2 : public Function<dim>
{
public:
  ExactSolutionEX2() = default;

  // Angular frequency for EX2
  static constexpr double omega = numbers::PI / 1.4142135623730951; // π/√2

  virtual double value(const Point<dim> &p,
        const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return spatial_mode(p) * std::cos(omega * this->get_time());
  }

  virtual Tensor<1, dim> gradient(const Point<dim> &p,
           const unsigned int component = 0) const override
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const double time_factor = std::cos(omega * this->get_time());
    return time_factor * spatial_mode_gradient(p);
  }
};

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Creates the appropriate exact solution for the given test case.
 */
inline std::unique_ptr<Function<dim>> create_exact_solution(TestCase test_case)
{
  if (test_case == TestCase::EX1)
    return std::make_unique<ExactSolutionEX1>();
  else
    return std::make_unique<ExactSolutionEX2>();
}

/**
 * Creates the appropriate forcing term for the given test case.
 */
inline std::unique_ptr<Function<dim>> create_forcing_term(TestCase test_case)
{
  if (test_case == TestCase::EX1)
    return std::make_unique<ForcingTermEX1>();
  else
    return std::make_unique<ForcingTermEX2>();
}

/**
 * Creates the appropriate initial acceleration for the given test case.
 * (Only needed for Newmark method)
 */
inline std::unique_ptr<Function<dim>> create_initial_acceleration(TestCase test_case)
{
  if (test_case == TestCase::EX1)
    return std::make_unique<InitialAccelerationEX1>();
  else
    return std::make_unique<InitialAccelerationEX2>();
}

/**
 * Prints information about the selected test case.
 */
inline void print_test_case_info(TestCase test_case, std::ostream &out = std::cout)
{
  if (test_case == TestCase::EX1)
    {
      out << "Running EX1: u(x,y,t) = sin(pi(x+1)/2) * sin(pi(y+1)/2) * cos(t)\n"
          << "             f = (pi^2/2 - 1) * phi(x,y) * cos(t)\n";
    }
  else
    {
      out << "Running EX2: u(x,y,t) = sin(pi(x+1)/2) * sin(pi(y+1)/2) * cos(pi/sqrt(2) * t)\n"
          << "             f = 0 (homogeneous)\n";
    }
}

} // namespace WaveEquation

#endif // WAVE_TEST_CASES_HPP