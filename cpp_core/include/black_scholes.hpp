#ifndef BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_HPP

#include <cmath>
#include <algorithm>

// Ensure math constants are available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

namespace quant {

/**
 * Black-Scholes-Merton Options Pricing
 * High-performance C++ implementation for quantitative finance
 */
class BlackScholes {
public:
    /**
     * Calculate standard normal CDF using approximation
     */
    static double norm_cdf(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }

    /**
     * Calculate standard normal PDF
     */
    static double norm_pdf(double x) {
        return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
    }

    /**
     * Calculate d1 parameter
     */
    static double calculate_d1(double S, double K, double T, double r, double sigma) {
        return (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    }

    /**
     * Calculate d2 parameter
     */
    static double calculate_d2(double d1, double sigma, double T) {
        return d1 - sigma * std::sqrt(T);
    }

    /**
     * European call option price
     */
    static double call_price(double S, double K, double T, double r, double sigma, double q = 0.0) {
        if (T <= 0.0) {
            return std::max(S - K, 0.0);
        }

        double d1 = calculate_d1(S, K, T, r - q, sigma);
        double d2 = calculate_d2(d1, sigma, T);

        return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    }

    /**
     * European put option price
     */
    static double put_price(double S, double K, double T, double r, double sigma, double q = 0.0) {
        if (T <= 0.0) {
            return std::max(K - S, 0.0);
        }

        double d1 = calculate_d1(S, K, T, r - q, sigma);
        double d2 = calculate_d2(d1, sigma, T);

        return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
    }

    /**
     * Option delta
     */
    static double delta(double S, double K, double T, double r, double sigma, bool is_call = true, double q = 0.0) {
        double d1 = calculate_d1(S, K, T, r - q, sigma);
        double exp_q = std::exp(-q * T);

        if (is_call) {
            return exp_q * norm_cdf(d1);
        } else {
            return -exp_q * norm_cdf(-d1);
        }
    }

    /**
     * Option gamma
     */
    static double gamma(double S, double K, double T, double r, double sigma, double q = 0.0) {
        double d1 = calculate_d1(S, K, T, r - q, sigma);
        return (std::exp(-q * T) * norm_pdf(d1)) / (S * sigma * std::sqrt(T));
    }

    /**
     * Option vega (per 1% change in volatility)
     */
    static double vega(double S, double K, double T, double r, double sigma, double q = 0.0) {
        double d1 = calculate_d1(S, K, T, r - q, sigma);
        return S * std::exp(-q * T) * norm_pdf(d1) * std::sqrt(T) / 100.0;
    }

    /**
     * Option theta (per day)
     */
    static double theta(double S, double K, double T, double r, double sigma, bool is_call = true, double q = 0.0) {
        double d1 = calculate_d1(S, K, T, r - q, sigma);
        double d2 = calculate_d2(d1, sigma, T);

        double term1 = -(S * std::exp(-q * T) * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T));

        if (is_call) {
            return (term1 - r * K * std::exp(-r * T) * norm_cdf(d2) + q * S * std::exp(-q * T) * norm_cdf(d1)) / 365.0;
        } else {
            return (term1 + r * K * std::exp(-r * T) * norm_cdf(-d2) - q * S * std::exp(-q * T) * norm_cdf(-d1)) / 365.0;
        }
    }

    /**
     * Option rho (per 1% change in rate)
     */
    static double rho(double S, double K, double T, double r, double sigma, bool is_call = true, double q = 0.0) {
        double d1 = calculate_d1(S, K, T, r - q, sigma);
        double d2 = calculate_d2(d1, sigma, T);

        if (is_call) {
            return K * T * std::exp(-r * T) * norm_cdf(d2) / 100.0;
        } else {
            return -K * T * std::exp(-r * T) * norm_cdf(-d2) / 100.0;
        }
    }

    /**
     * Implied volatility using Newton-Raphson method
     */
    static double implied_volatility(double market_price, double S, double K, double T, 
                                    double r, bool is_call = true, double q = 0.0,
                                    double initial_guess = 0.2, double tolerance = 1e-6,
                                    int max_iterations = 100) {
        double sigma = initial_guess;

        for (int i = 0; i < max_iterations; ++i) {
            double price = is_call ? call_price(S, K, T, r, sigma, q) : put_price(S, K, T, r, sigma, q);
            double diff = price - market_price;

            if (std::abs(diff) < tolerance) {
                return sigma;
            }

            double vega_val = vega(S, K, T, r, sigma, q) * 100.0; // Convert back to per-unit
            if (std::abs(vega_val) < 1e-10) {
                break; // Avoid division by zero
            }

            sigma = sigma - diff / vega_val;

            // Keep sigma in reasonable bounds
            if (sigma < 0.001) sigma = 0.001;
            if (sigma > 5.0) sigma = 5.0;
        }

        return sigma;
    }
};

} // namespace quant

#endif // BLACK_SCHOLES_HPP
