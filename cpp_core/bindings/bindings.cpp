#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/black_scholes.hpp"
#include "../include/monte_carlo.hpp"
#include "../include/portfolio.hpp"

namespace py = pybind11;
using namespace quant;

PYBIND11_MODULE(quant_cpp, m) {
    m.doc() = "High-performance C++ quantitative finance library";

    // Black-Scholes module
    py::class_<BlackScholes>(m, "BlackScholes")
        .def_static("call_price", &BlackScholes::call_price,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("q") = 0.0,
                   "Calculate European call option price")
        .def_static("put_price", &BlackScholes::put_price,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("q") = 0.0,
                   "Calculate European put option price")
        .def_static("delta", &BlackScholes::delta,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("is_call") = true, py::arg("q") = 0.0,
                   "Calculate option delta")
        .def_static("gamma", &BlackScholes::gamma,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("q") = 0.0,
                   "Calculate option gamma")
        .def_static("vega", &BlackScholes::vega,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("q") = 0.0,
                   "Calculate option vega")
        .def_static("theta", &BlackScholes::theta,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("is_call") = true, py::arg("q") = 0.0,
                   "Calculate option theta")
        .def_static("rho", &BlackScholes::rho,
                   py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), 
                   py::arg("sigma"), py::arg("is_call") = true, py::arg("q") = 0.0,
                   "Calculate option rho")
        .def_static("implied_volatility", &BlackScholes::implied_volatility,
                   py::arg("market_price"), py::arg("S"), py::arg("K"), 
                   py::arg("T"), py::arg("r"), py::arg("is_call") = true,
                   py::arg("q") = 0.0, py::arg("initial_guess") = 0.2,
                   py::arg("tolerance") = 1e-6, py::arg("max_iterations") = 100,
                   "Calculate implied volatility");

    // Monte Carlo Engine
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<unsigned int>(), py::arg("seed") = 42)
        .def("price_european_option", &MonteCarloEngine::price_european_option,
             py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"),
             py::arg("sigma"), py::arg("is_call"), py::arg("n_simulations"),
             py::arg("q") = 0.0,
             "Price European option using Monte Carlo simulation")
        .def("price_asian_option", &MonteCarloEngine::price_asian_option,
             py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"),
             py::arg("sigma"), py::arg("is_call"), py::arg("n_simulations"),
             py::arg("n_steps"),
             "Price Asian option using Monte Carlo simulation")
        .def("simulate_gbm_path", &MonteCarloEngine::simulate_gbm_path,
             py::arg("S0"), py::arg("mu"), py::arg("sigma"), 
             py::arg("T"), py::arg("steps"),
             "Simulate geometric Brownian motion path")
        .def("calculate_var", &MonteCarloEngine::calculate_var,
             py::arg("returns"), py::arg("confidence_level"),
             "Calculate Value at Risk")
        .def("calculate_cvar", &MonteCarloEngine::calculate_cvar,
             py::arg("returns"), py::arg("confidence_level"),
             "Calculate Conditional Value at Risk")
        .def("simulate_portfolio_returns", &MonteCarloEngine::simulate_portfolio_returns,
             py::arg("weights"), py::arg("expected_returns"), 
             py::arg("cov_matrix"), py::arg("n_simulations"),
             "Simulate portfolio returns for VaR/CVaR calculation");

    // Portfolio Analytics
    py::class_<Portfolio>(m, "Portfolio")
        .def_static("portfolio_return", &Portfolio::portfolio_return,
                   py::arg("weights"), py::arg("expected_returns"),
                   "Calculate portfolio expected return")
        .def_static("portfolio_variance", &Portfolio::portfolio_variance,
                   py::arg("weights"), py::arg("cov_matrix"),
                   "Calculate portfolio variance")
        .def_static("portfolio_volatility", &Portfolio::portfolio_volatility,
                   py::arg("weights"), py::arg("cov_matrix"),
                   "Calculate portfolio volatility")
        .def_static("sharpe_ratio", &Portfolio::sharpe_ratio,
                   py::arg("weights"), py::arg("expected_returns"),
                   py::arg("cov_matrix"), py::arg("risk_free_rate"),
                   "Calculate Sharpe ratio")
        .def_static("portfolio_beta", &Portfolio::portfolio_beta,
                   py::arg("weights"), py::arg("asset_betas"),
                   "Calculate portfolio beta")
        .def_static("tracking_error", &Portfolio::tracking_error,
                   py::arg("portfolio_returns"), py::arg("benchmark_returns"),
                   "Calculate tracking error")
        .def_static("information_ratio", &Portfolio::information_ratio,
                   py::arg("portfolio_returns"), py::arg("benchmark_returns"),
                   "Calculate information ratio")
        .def_static("sortino_ratio", &Portfolio::sortino_ratio,
                   py::arg("returns"), py::arg("risk_free_rate"),
                   py::arg("target_return") = 0.0,
                   "Calculate Sortino ratio")
        .def_static("calmar_ratio", &Portfolio::calmar_ratio,
                   py::arg("cumulative_returns"),
                   "Calculate Calmar ratio")
        .def_static("max_drawdown", &Portfolio::max_drawdown,
                   py::arg("cumulative_returns"),
                   "Calculate maximum drawdown")
        .def_static("historical_var", &Portfolio::historical_var,
                   py::arg("returns"), py::arg("confidence_level"),
                   "Calculate Value at Risk (historical method)")
        .def_static("conditional_var", &Portfolio::conditional_var,
                   py::arg("returns"), py::arg("confidence_level"),
                   "Calculate Conditional Value at Risk");
}
