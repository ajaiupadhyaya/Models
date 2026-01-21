#!/usr/bin/env python3
"""
🚀 EXAMPLE: Full Automated Trading Loop

Demonstrates the complete end-to-end trading system:
1. Fetch macro data
2. Predict with ML models
3. Get AI trading recommendations
4. Execute trades on Alpaca
5. Generate OpenAI narrative
"""

import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datetime import datetime
from core.data_fetcher import DataFetcher
from models.ml.advanced_trading import EnsemblePredictor, LSTMPredictor
from core.ai_analysis import get_ai_service
from core.paper_trading import AlpacaAdapter
from core.investor_reports import InvestorReportGenerator


def main():
    """Run the full trading loop."""
    
    logger.info("=" * 70)
    logger.info("🚀 AUTOMATED TRADING LOOP - EXAMPLE")
    logger.info("=" * 70)
    
    # Configuration
    symbols = ["AAPL", "MSFT"]
    lookback_days = 90
    
    # Initialize services
    fetcher = DataFetcher()
    ai_service = get_ai_service()
    
    try:
        alpaca_key = os.getenv("ALPACA_API_KEY")
        alpaca_secret = os.getenv("ALPACA_API_SECRET")
        alpaca = AlpacaAdapter(alpaca_key, alpaca_secret) if (alpaca_key and alpaca_secret) else None
    except:
        alpaca = None
    
    # Step 1: Fetch macro context
    logger.info("\n📊 STEP 1: Fetching macro data...")
    try:
        unemployment = fetcher.get_economic_indicator("UNRATE")
        gdp = fetcher.get_economic_indicator("GDP")
        logger.info(f"  ✓ Unemployment rate: {unemployment}%")
        logger.info(f"  ✓ GDP: ${gdp}B")
    except Exception as e:
        logger.warning(f"  ⚠ Macro fetch failed: {e}")
        unemployment = None
        gdp = None
    
    # Step 2: Process each symbol
    trading_plan = {}
    
    for symbol in symbols:
        logger.info(f"\n📈 STEP 2: Analyzing {symbol}...")
        
        try:
            # Fetch stock data
            df = fetcher.get_stock_data(symbol, period=f"{lookback_days}d")
            if df is None or len(df) < 20:
                logger.warning(f"  ⚠ Insufficient data for {symbol}")
                continue
            
            current_price = df['Close'].iloc[-1]
            logger.info(f"  ✓ Current price: ${current_price:.2f}")
            
            # Ensemble prediction
            logger.info(f"  • Training ensemble model...")
            ensemble = EnsemblePredictor(lookback_window=20)
            ensemble.train(df)
            ensemble_pred = ensemble.predict(df)
            if hasattr(ensemble_pred, 'iloc'):
                ensemble_pred = ensemble_pred.iloc[-1]
            else:
                ensemble_pred = ensemble_pred[-1]
            logger.info(f"    ✓ Ensemble prediction: ${ensemble_pred:.2f}")
            
            # LSTM prediction
            logger.info(f"  • Training LSTM model...")
            try:
                lstm = LSTMPredictor(lookback_window=20, hidden_units=16)
                lstm.train(df, epochs=2, batch_size=32)
                lstm_pred = lstm.predict(df)
                if hasattr(lstm_pred, 'iloc'):
                    lstm_pred = lstm_pred.iloc[-1]
                else:
                    lstm_pred = lstm_pred[-1]
                logger.info(f"    ✓ LSTM prediction: ${lstm_pred:.2f}")
            except Exception as e:
                logger.warning(f"    ⚠ LSTM failed: {str(e)[:40]}")
                lstm_pred = None
            
            # Average predictions
            predictions = [p for p in [ensemble_pred, lstm_pred] if p is not None]
            avg_pred = sum(predictions) / len(predictions) if predictions else current_price
            confidence = 0.75 if len(predictions) > 1 else 0.5
            
            logger.info(f"  ✓ Average prediction: ${avg_pred:.2f}")
            logger.info(f"  ✓ Implied move: {((avg_pred - current_price) / current_price * 100):+.2f}%")
            
            # Get AI insight
            logger.info(f"  • Getting AI trading recommendation...")
            market_context = f"Unemployment: {unemployment}%, GDP: ${gdp}B" if unemployment else "Normal conditions"
            
            insight = ai_service.generate_trading_insight(
                symbol=symbol,
                current_price=current_price,
                prediction=avg_pred,
                confidence=confidence,
                market_context=market_context
            )
            
            action = insight.get("action", "HOLD")
            reasoning = insight.get("reasoning", "")
            risk_level = insight.get("risk_level", "medium")
            
            logger.info(f"  ✓ AI Action: {action}")
            logger.info(f"  ✓ Reasoning: {reasoning[:60]}...")
            logger.info(f"  ✓ Risk Level: {risk_level}")
            
            trading_plan[symbol] = {
                "action": action,
                "current_price": current_price,
                "predicted_price": avg_pred,
                "confidence": confidence,
                "reasoning": reasoning,
                "risk_level": risk_level
            }
        
        except Exception as e:
            logger.error(f"  ✗ Error processing {symbol}: {e}")
            continue
    
    # Step 3: Execute trades (if Alpaca is configured)
    logger.info("\n💰 STEP 3: Trade Execution...")
    
    executed_trades = []
    
    if alpaca and alpaca.is_authenticated():
        logger.info("  ✓ Alpaca authenticated")
        
        for symbol, plan in trading_plan.items():
            action = plan["action"]
            
            try:
                if action == "BUY":
                    logger.info(f"  → Placing BUY order for {symbol}...")
                    order = alpaca.submit_order(
                        symbol=symbol,
                        qty=10,
                        side="buy",
                        type="market"
                    )
                    if order:
                        logger.info(f"    ✓ Order ID: {order.get('id', 'unknown')}")
                        executed_trades.append({
                            "symbol": symbol,
                            "action": "BUY",
                            "qty": 10,
                            "order_id": order.get("id")
                        })
                
                elif action == "SELL":
                    logger.info(f"  → Placing SELL order for {symbol}...")
                    positions = alpaca.get_positions()
                    if positions and symbol in positions:
                        qty = positions[symbol]["qty"]
                        order = alpaca.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="sell",
                            type="market"
                        )
                        if order:
                            logger.info(f"    ✓ Order ID: {order.get('id', 'unknown')}")
                            executed_trades.append({
                                "symbol": symbol,
                                "action": "SELL",
                                "qty": qty,
                                "order_id": order.get("id")
                            })
                    else:
                        logger.info(f"    ⚠ No position to sell")
            
            except Exception as e:
                logger.error(f"  ✗ Trade execution failed: {e}")
    
    else:
        logger.info("  ⚠ Alpaca not configured (dry run)")
    
    # Step 4: Generate report
    logger.info("\n📄 STEP 4: Generating Report...")
    
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "symbols_analyzed": len(trading_plan),
        "trading_plan": trading_plan,
        "trades_executed": len(executed_trades),
        "executed_trades": executed_trades,
        "macro_context": {
            "unemployment": unemployment,
            "gdp": gdp
        }
    }
    
    try:
        reporter = InvestorReportGenerator()
        summary = reporter.generate_executive_summary(
            symbol=",".join(symbols),
            metrics={
                "symbols_analyzed": len(trading_plan),
                "trades_planned": len(trading_plan),
                "trades_executed": len(executed_trades),
                "avg_confidence": sum(p["confidence"] for p in trading_plan.values()) / len(trading_plan) if trading_plan else 0
            },
            ai_enabled=True
        )
        logger.info(f"  ✓ Report generated")
    except Exception as e:
        logger.warning(f"  ⚠ Report generation failed: {e}")
        summary = ""
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ TRADING LOOP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Symbols Analyzed: {len(trading_plan)}")
    logger.info(f"Trading Plans: {len(trading_plan)}")
    logger.info(f"Trades Executed: {len(executed_trades)}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 70)
    
    if trading_plan:
        logger.info("\n📊 TRADING PLAN:")
        for symbol, plan in trading_plan.items():
            logger.info(f"  {symbol}: {plan['action']} @ ${plan['current_price']:.2f} (predict: ${plan['predicted_price']:.2f})")
    
    if executed_trades:
        logger.info("\n💳 EXECUTED TRADES:")
        for trade in executed_trades:
            logger.info(f"  {trade['action']} {trade['qty']} {trade['symbol']}")
    
    logger.info("\n✨ Ready for next iteration!")
    logger.info("💡 Tip: Run this script on a schedule (e.g., daily with cron) for automated trading")
    
    return report_data


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
