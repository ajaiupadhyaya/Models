/**
 * Example usage of the Pure C quantitative finance library
 * Demonstrates portability and performance for legacy systems
 */

#include <stdio.h>
#include <stdlib.h>
#include "../include/quant_c.h"

int main() {
    printf("========================================\n");
    printf("Pure C Quantitative Finance Library\n");
    printf("========================================\n\n");
    
    /* Black-Scholes parameters */
    double S = 100.0;  /* Stock price */
    double K = 100.0;  /* Strike price */
    double T = 1.0;    /* Time to maturity (years) */
    double r = 0.05;   /* Risk-free rate */
    double sigma = 0.2; /* Volatility */
    double q = 0.0;    /* Dividend yield */
    
    /* Calculate option prices */
    double call = call_price_c(S, K, T, r, sigma, q);
    double put = put_price_c(S, K, T, r, sigma, q);
    
    printf("Black-Scholes European Options:\n");
    printf("  Spot: $%.2f, Strike: $%.2f\n", S, K);
    printf("  Time: %.2f years, Vol: %.1f%%, Rate: %.1f%%\n", 
           T, sigma * 100, r * 100);
    printf("\n");
    printf("  Call Price: $%.4f\n", call);
    printf("  Put Price:  $%.4f\n", put);
    printf("\n");
    
    /* Calculate Greeks */
    printf("Greeks (Call Option):\n");
    printf("  Delta: %.4f\n", delta_c(S, K, T, r, sigma, 1, q));
    printf("  Gamma: %.6f\n", gamma_c(S, K, T, r, sigma, q));
    printf("  Vega:  %.4f\n", vega_c(S, K, T, r, sigma, q));
    printf("  Theta: %.4f (per day)\n", theta_c(S, K, T, r, sigma, 1, q));
    printf("  Rho:   %.4f (per 1%% rate change)\n", rho_c(S, K, T, r, sigma, 1, q));
    printf("\n");
    
    /* Portfolio calculations */
    printf("Portfolio Analytics:\n");
    double weights[] = {0.3, 0.3, 0.4};
    double returns[] = {0.10, 0.12, 0.08};
    int n_assets = 3;
    
    double port_return = portfolio_return_c(weights, returns, n_assets);
    printf("  Portfolio Return: %.2f%%\n", port_return * 100);
    
    /* Risk metrics */
    double cumulative[] = {1.0, 1.1, 1.05, 1.15, 1.08, 1.20};
    int n_points = 6;
    double max_dd = max_drawdown_c(cumulative, n_points);
    printf("  Maximum Drawdown: %.2f%%\n", max_dd * 100);
    printf("\n");
    
    /* Performance test */
    printf("Performance Test (10,000 iterations):\n");
    int n_iter = 10000;
    double sum = 0.0;
    
    for (int i = 0; i < n_iter; i++) {
        sum += call_price_c(S, K + i * 0.01, T, r, sigma, q);
    }
    
    printf("  Average call price: $%.4f\n", sum / n_iter);
    printf("  (Pure C implementation, highly optimized)\n");
    printf("\n");
    
    printf("========================================\n");
    printf("Pure C library test completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}
