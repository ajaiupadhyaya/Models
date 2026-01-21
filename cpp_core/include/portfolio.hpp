#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace quant {

/**
 * Portfolio Optimization and Analytics
 * High-performance C++ implementations for portfolio calculations
 */
class Portfolio {
public:
    /**
     * Calculate portfolio expected return
     */
    static double portfolio_return(const std::vector<double>& weights,
                                   const std::vector<double>& expected_returns) {
        double ret = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            ret += weights[i] * expected_returns[i];
        }
        return ret;
    }

    /**
     * Calculate portfolio variance
     */
    static double portfolio_variance(const std::vector<double>& weights,
                                     const std::vector<std::vector<double>>& cov_matrix) {
        size_t n = weights.size();
        double variance = 0.0;

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                variance += weights[i] * weights[j] * cov_matrix[i][j];
            }
        }

        return variance;
    }

    /**
     * Calculate portfolio volatility (standard deviation)
     */
    static double portfolio_volatility(const std::vector<double>& weights,
                                       const std::vector<std::vector<double>>& cov_matrix) {
        return std::sqrt(portfolio_variance(weights, cov_matrix));
    }

    /**
     * Calculate Sharpe ratio
     */
    static double sharpe_ratio(const std::vector<double>& weights,
                              const std::vector<double>& expected_returns,
                              const std::vector<std::vector<double>>& cov_matrix,
                              double risk_free_rate) {
        double ret = portfolio_return(weights, expected_returns);
        double vol = portfolio_volatility(weights, cov_matrix);
        
        if (vol < 1e-10) return 0.0;
        return (ret - risk_free_rate) / vol;
    }

    /**
     * Calculate portfolio beta
     */
    static double portfolio_beta(const std::vector<double>& weights,
                                const std::vector<double>& asset_betas) {
        double beta = 0.0;
        for (size_t i = 0; i < weights.size(); ++i) {
            beta += weights[i] * asset_betas[i];
        }
        return beta;
    }

    /**
     * Calculate tracking error
     */
    static double tracking_error(const std::vector<double>& portfolio_returns,
                                 const std::vector<double>& benchmark_returns) {
        if (portfolio_returns.size() != benchmark_returns.size()) {
            return 0.0;
        }

        std::vector<double> tracking_diff(portfolio_returns.size());
        for (size_t i = 0; i < portfolio_returns.size(); ++i) {
            tracking_diff[i] = portfolio_returns[i] - benchmark_returns[i];
        }

        return standard_deviation(tracking_diff);
    }

    /**
     * Calculate information ratio
     */
    static double information_ratio(const std::vector<double>& portfolio_returns,
                                   const std::vector<double>& benchmark_returns) {
        double te = tracking_error(portfolio_returns, benchmark_returns);
        if (te < 1e-10) return 0.0;

        double excess_return = mean(portfolio_returns) - mean(benchmark_returns);
        return excess_return / te;
    }

    /**
     * Calculate Sortino ratio
     */
    static double sortino_ratio(const std::vector<double>& returns,
                               double risk_free_rate,
                               double target_return = 0.0) {
        double avg_return = mean(returns);
        double downside_dev = downside_deviation(returns, target_return);
        
        if (downside_dev < 1e-10) return 0.0;
        return (avg_return - risk_free_rate) / downside_dev;
    }

    /**
     * Calculate Calmar ratio (return / max drawdown)
     */
    static double calmar_ratio(const std::vector<double>& cumulative_returns) {
        if (cumulative_returns.empty()) return 0.0;

        double total_return = cumulative_returns.back();
        double max_dd = max_drawdown(cumulative_returns);
        
        if (std::abs(max_dd) < 1e-10) return 0.0;
        return total_return / std::abs(max_dd);
    }

    /**
     * Calculate maximum drawdown
     */
    static double max_drawdown(const std::vector<double>& cumulative_returns) {
        if (cumulative_returns.empty()) return 0.0;

        double max_dd = 0.0;
        double peak = cumulative_returns[0];

        for (double value : cumulative_returns) {
            if (value > peak) {
                peak = value;
            }
            double dd = (peak - value) / peak;
            if (dd > max_dd) {
                max_dd = dd;
            }
        }

        return max_dd;
    }

    /**
     * Calculate Value at Risk (historical method)
     */
    static double historical_var(const std::vector<double>& returns, double confidence_level) {
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t index = static_cast<size_t>((1.0 - confidence_level) * sorted_returns.size());
        return -sorted_returns[index];
    }

    /**
     * Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
     */
    static double conditional_var(const std::vector<double>& returns, double confidence_level) {
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t index = static_cast<size_t>((1.0 - confidence_level) * sorted_returns.size());
        double sum = 0.0;
        for (size_t i = 0; i <= index; ++i) {
            sum += sorted_returns[i];
        }
        
        return -sum / (index + 1);
    }

private:
    /**
     * Calculate mean of a vector
     */
    static double mean(const std::vector<double>& data) {
        if (data.empty()) return 0.0;
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    /**
     * Calculate standard deviation
     */
    static double standard_deviation(const std::vector<double>& data) {
        if (data.size() < 2) return 0.0;

        double m = mean(data);
        double sum_sq_diff = 0.0;
        for (double value : data) {
            double diff = value - m;
            sum_sq_diff += diff * diff;
        }

        return std::sqrt(sum_sq_diff / (data.size() - 1));
    }

    /**
     * Calculate downside deviation
     */
    static double downside_deviation(const std::vector<double>& returns, double target) {
        if (returns.empty()) return 0.0;

        double sum_sq_diff = 0.0;
        int count = 0;
        for (double ret : returns) {
            if (ret < target) {
                double diff = ret - target;
                sum_sq_diff += diff * diff;
                count++;
            }
        }

        if (count == 0) return 0.0;
        return std::sqrt(sum_sq_diff / count);
    }
};

} // namespace quant

#endif // PORTFOLIO_HPP
