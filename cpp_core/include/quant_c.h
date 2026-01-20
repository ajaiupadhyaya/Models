#ifndef QUANT_C_H
#define QUANT_C_H

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Black-Scholes Options Pricing in Pure C
 * Provides C interface for legacy systems and maximum portability
 */

/* Error function approximation for normal CDF */
static inline double erf_approx(double x) {
    /* Abramowitz and Stegun approximation */
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;
    
    int sign = (x < 0) ? -1 : 1;
    x = fabs(x);
    
    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    
    return sign * y;
}

/* Standard normal CDF */
static inline double norm_cdf_c(double x) {
    return 0.5 * (1.0 + erf_approx(x / sqrt(2.0)));
}

/* Standard normal PDF */
static inline double norm_pdf_c(double x) {
    return (1.0 / sqrt(2.0 * M_PI)) * exp(-0.5 * x * x);
}

/* Calculate d1 parameter */
static inline double calculate_d1_c(double S, double K, double T, double r, double sigma) {
    return (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
}

/* Calculate d2 parameter */
static inline double calculate_d2_c(double d1, double sigma, double T) {
    return d1 - sigma * sqrt(T);
}

/* European call option price */
static inline double call_price_c(double S, double K, double T, double r, double sigma, double q) {
    if (T <= 0.0) {
        return (S - K > 0.0) ? (S - K) : 0.0;
    }
    
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    double d2 = calculate_d2_c(d1, sigma, T);
    
    return S * exp(-q * T) * norm_cdf_c(d1) - K * exp(-r * T) * norm_cdf_c(d2);
}

/* European put option price */
static inline double put_price_c(double S, double K, double T, double r, double sigma, double q) {
    if (T <= 0.0) {
        return (K - S > 0.0) ? (K - S) : 0.0;
    }
    
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    double d2 = calculate_d2_c(d1, sigma, T);
    
    return K * exp(-r * T) * norm_cdf_c(-d2) - S * exp(-q * T) * norm_cdf_c(-d1);
}

/* Option delta */
static inline double delta_c(double S, double K, double T, double r, double sigma, int is_call, double q) {
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    double exp_q = exp(-q * T);
    
    if (is_call) {
        return exp_q * norm_cdf_c(d1);
    } else {
        return -exp_q * norm_cdf_c(-d1);
    }
}

/* Option gamma */
static inline double gamma_c(double S, double K, double T, double r, double sigma, double q) {
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    return (exp(-q * T) * norm_pdf_c(d1)) / (S * sigma * sqrt(T));
}

/* Option vega (per 1% change) */
static inline double vega_c(double S, double K, double T, double r, double sigma, double q) {
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    return S * exp(-q * T) * norm_pdf_c(d1) * sqrt(T) / 100.0;
}

/* Option theta (per day) */
static inline double theta_c(double S, double K, double T, double r, double sigma, int is_call, double q) {
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    double d2 = calculate_d2_c(d1, sigma, T);
    
    double term1 = -(S * exp(-q * T) * norm_pdf_c(d1) * sigma) / (2.0 * sqrt(T));
    
    if (is_call) {
        return (term1 - r * K * exp(-r * T) * norm_cdf_c(d2) + 
                q * S * exp(-q * T) * norm_cdf_c(d1)) / 365.0;
    } else {
        return (term1 + r * K * exp(-r * T) * norm_cdf_c(-d2) - 
                q * S * exp(-q * T) * norm_cdf_c(-d1)) / 365.0;
    }
}

/* Option rho (per 1% change) */
static inline double rho_c(double S, double K, double T, double r, double sigma, int is_call, double q) {
    double d1 = calculate_d1_c(S, K, T, r - q, sigma);
    double d2 = calculate_d2_c(d1, sigma, T);
    
    if (is_call) {
        return K * T * exp(-r * T) * norm_cdf_c(d2) / 100.0;
    } else {
        return -K * T * exp(-r * T) * norm_cdf_c(-d2) / 100.0;
    }
}

/**
 * Simple portfolio return calculation
 */
static inline double portfolio_return_c(const double* weights, const double* returns, int n) {
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += weights[i] * returns[i];
    }
    return result;
}

/**
 * Simple variance calculation
 */
static inline double variance_c(const double* data, int n) {
    if (n < 2) return 0.0;
    
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    double mean = sum / n;
    
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    
    return sum_sq / (n - 1);
}

/**
 * Maximum drawdown calculation
 */
static inline double max_drawdown_c(const double* cumulative_returns, int n) {
    if (n == 0) return 0.0;
    
    double max_dd = 0.0;
    double peak = cumulative_returns[0];
    
    for (int i = 0; i < n; i++) {
        if (cumulative_returns[i] > peak) {
            peak = cumulative_returns[i];
        }
        double dd = (peak - cumulative_returns[i]) / peak;
        if (dd > max_dd) {
            max_dd = dd;
        }
    }
    
    return max_dd;
}

#ifdef __cplusplus
}
#endif

#endif /* QUANT_C_H */
