export const TERMINAL_API_ENDPOINTS = {
  models: "/api/v1/models",
  trainModel: "/api/v1/models/train",
  backtestRun: "/api/v1/backtest/run",
  backtestCompare: "/api/v1/backtest/compare",
  backtestWalkForward: "/api/v1/backtest/walk-forward",
  backtestTechnical: "/api/v1/backtest/technical",
  quantBacktest: "/api/v1/quant/backtest",
  quantFactorRank: "/api/v1/quant/factor-rank",
  quantPairs: "/api/v1/quant/pairs",
  quantOptionsChain: "/api/v1/quant/options-chain",
} as const;
