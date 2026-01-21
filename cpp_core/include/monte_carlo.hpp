#ifndef MONTE_CARLO_HPP
#define MONTE_CARLO_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace quant {

/**
 * Monte Carlo Simulation Engine
 * High-performance simulations for options pricing and risk analysis
 */
class MonteCarloEngine {
private:
    std::mt19937_64 rng;
    std::normal_distribution<double> normal_dist;

public:
    MonteCarloEngine(unsigned int seed = 42) 
        : rng(seed), normal_dist(0.0, 1.0) {}

    /**
     * Generate standard normal random numbers
     */
    void generate_normals(std::vector<double>& output, size_t n) {
        output.resize(n);
        for (size_t i = 0; i < n; ++i) {
            output[i] = normal_dist(rng);
        }
    }

    /**
     * Simulate geometric Brownian motion path
     */
    std::vector<double> simulate_gbm_path(double S0, double mu, double sigma, 
                                          double T, int steps) {
        std::vector<double> path(steps + 1);
        path[0] = S0;
        
        double dt = T / steps;
        double drift = (mu - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * std::sqrt(dt);

        for (int i = 1; i <= steps; ++i) {
            double z = normal_dist(rng);
            path[i] = path[i-1] * std::exp(drift + diffusion * z);
        }

        return path;
    }

    /**
     * European option pricing via Monte Carlo
     */
    double price_european_option(double S0, double K, double T, double r, 
                                 double sigma, bool is_call, int n_simulations,
                                 double q = 0.0) {
        double sum_payoffs = 0.0;
        
        for (int i = 0; i < n_simulations; ++i) {
            double z = normal_dist(rng);
            double ST = S0 * std::exp((r - q - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);
            
            double payoff = is_call ? std::max(ST - K, 0.0) : std::max(K - ST, 0.0);
            sum_payoffs += payoff;
        }

        return std::exp(-r * T) * (sum_payoffs / n_simulations);
    }

    /**
     * Asian option pricing via Monte Carlo
     */
    double price_asian_option(double S0, double K, double T, double r, 
                              double sigma, bool is_call, int n_simulations,
                              int n_steps) {
        double sum_payoffs = 0.0;
        
        for (int i = 0; i < n_simulations; ++i) {
            auto path = simulate_gbm_path(S0, r, sigma, T, n_steps);
            double avg_price = std::accumulate(path.begin(), path.end(), 0.0) / path.size();
            
            double payoff = is_call ? std::max(avg_price - K, 0.0) : std::max(K - avg_price, 0.0);
            sum_payoffs += payoff;
        }

        return std::exp(-r * T) * (sum_payoffs / n_simulations);
    }

    /**
     * Calculate Value at Risk (VaR) using Monte Carlo
     */
    double calculate_var(const std::vector<double>& returns, double confidence_level) {
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t index = static_cast<size_t>((1.0 - confidence_level) * sorted_returns.size());
        return -sorted_returns[index];
    }

    /**
     * Calculate Conditional Value at Risk (CVaR)
     */
    double calculate_cvar(const std::vector<double>& returns, double confidence_level) {
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t index = static_cast<size_t>((1.0 - confidence_level) * sorted_returns.size());
        double sum = 0.0;
        for (size_t i = 0; i <= index; ++i) {
            sum += sorted_returns[i];
        }
        
        return -sum / (index + 1);
    }

    /**
     * Simulate portfolio returns for VaR/CVaR calculation
     */
    std::vector<double> simulate_portfolio_returns(const std::vector<double>& weights,
                                                   const std::vector<double>& expected_returns,
                                                   const std::vector<std::vector<double>>& cov_matrix,
                                                   int n_simulations) {
        size_t n_assets = weights.size();
        std::vector<double> portfolio_returns(n_simulations);

        // Cholesky decomposition for correlated random numbers
        auto L = cholesky_decomposition(cov_matrix);

        for (int sim = 0; sim < n_simulations; ++sim) {
            std::vector<double> z(n_assets);
            for (size_t i = 0; i < n_assets; ++i) {
                z[i] = normal_dist(rng);
            }

            // Apply Cholesky to get correlated normals
            std::vector<double> correlated_z(n_assets, 0.0);
            for (size_t i = 0; i < n_assets; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    correlated_z[i] += L[i][j] * z[j];
                }
            }

            // Calculate portfolio return
            double portfolio_return = 0.0;
            for (size_t i = 0; i < n_assets; ++i) {
                portfolio_return += weights[i] * (expected_returns[i] + correlated_z[i]);
            }
            portfolio_returns[sim] = portfolio_return;
        }

        return portfolio_returns;
    }

private:
    /**
     * Cholesky decomposition for covariance matrix
     */
    std::vector<std::vector<double>> cholesky_decomposition(const std::vector<std::vector<double>>& matrix) {
        size_t n = matrix.size();
        std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < j; ++k) {
                    sum += L[i][k] * L[j][k];
                }

                if (i == j) {
                    L[i][j] = std::sqrt(std::max(matrix[i][i] - sum, 0.0));
                } else {
                    if (L[j][j] > 1e-10) {
                        L[i][j] = (matrix[i][j] - sum) / L[j][j];
                    }
                }
            }
        }

        return L;
    }
};

} // namespace quant

#endif // MONTE_CARLO_HPP
