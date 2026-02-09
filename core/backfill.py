"""
Historical Data Backfill Script

Populates cold storage with historical OHLCV data for a list of symbols.

Features:
- Parallel processing with rate limiting
- Progress tracking with checkpoints
- Resume capability after failures
- Validation and error handling

Usage:
    python -m core.backfill --symbols AAPL,GOOGL,MSFT --years 10
    python -m core.backfill --file symbols.txt --years 5 --workers 4
"""

import argparse
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .unified_fetcher import UnifiedDataFetcher, AssetType
from .cold_storage import ColdStorageManager
from .data_providers import OHLCV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BackfillManager:
    """
    Manages historical data backfill process.
    
    Features:
    - Downloads and stores historical data for multiple symbols
    - Progress tracking and checkpoint/resume
    - Parallel execution with rate limiting
    - Validation and error handling
    """
    
    def __init__(
        self,
        fetcher: Optional[UnifiedDataFetcher] = None,
        storage: Optional[ColdStorageManager] = None,
        checkpoint_path: Path = Path("data/backfill_checkpoint.json"),
    ):
        """
        Initialize backfill manager.
        
        Args:
            fetcher: Unified data fetcher (creates default if None)
            storage: Cold storage manager (creates default if None)
            checkpoint_path: Path to checkpoint file
        """
        self.fetcher = fetcher or UnifiedDataFetcher()
        self.storage = storage or ColdStorageManager()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                return json.load(f)
        return {"completed": [], "failed": [], "in_progress": None}
    
    def save_checkpoint(self, checkpoint: Dict):
        """Save checkpoint to disk."""
        with open(self.checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
    
    def backfill_symbol(
        self,
        symbol: str,
        asset_type: AssetType,
        years: int = 10,
        interval: str = "1d",
    ) -> bool:
        """
        Backfill historical data for a single symbol.
        
        Args:
            symbol: Symbol to backfill
            asset_type: Asset type (EQUITY, CRYPTO, etc.)
            years: Number of years of history
            interval: Bar interval
            
        Returns:
            True if successful, False otherwise
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            logger.info(f"Starting backfill: {symbol} ({start_date.date()} to {end_date.date()})")
            
            # Fetch data
            result = self.fetcher.fetch_ohlcv(
                symbol=symbol,
                asset_type=asset_type,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            
            if not result.data:
                logger.warning(f"No data fetched for {symbol}")
                return False
            
            # Group by year/month and write to partitions
            monthly_data: Dict[tuple[int, int], List[OHLCV]] = {}
            
            for bar in result.data:
                key = (bar.date.year, bar.date.month)
                monthly_data.setdefault(key, []).append(bar)
            
            # Write partitions
            for (year, month), bars in monthly_data.items():
                self.storage.write_partition(
                    symbol=symbol,
                    data=bars,
                    year=year,
                    month=month,
                )
            
            logger.info(
                f"✓ {symbol}: {len(result.data)} bars written to "
                f"{len(monthly_data)} partitions"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"✗ {symbol} backfill failed: {e}")
            return False
    
    def backfill_batch(
        self,
        symbols: List[str],
        asset_type: AssetType = AssetType.EQUITY,
        years: int = 10,
        workers: int = 4,
        resume: bool = True,
    ) -> Dict[str, any]:
        """
        Backfill multiple symbols in parallel.
        
        Args:
            symbols: List of symbols to backfill
            asset_type: Asset type for all symbols
            years: Number of years of history
            workers: Number of parallel workers
            resume: Resume from checkpoint if available
            
        Returns:
            Dict with summary stats
        """
        checkpoint = self.load_checkpoint() if resume else {"completed": [], "failed": []}
        
        # Filter out already completed symbols
        remaining = [s for s in symbols if s not in checkpoint["completed"]]
        
        if not remaining:
            logger.info("All symbols already completed!")
            return checkpoint
        
        logger.info(
            f"Starting batch backfill: {len(remaining)} symbols "
            f"({workers} workers, {years} years)"
        )
        
        start_time = time.time()
        completed = checkpoint["completed"].copy()
        failed = checkpoint["failed"].copy()
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self.backfill_symbol,
                    symbol,
                    asset_type,
                    years,
                ): symbol
                for symbol in remaining
            }
            
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed.append(symbol)
                    else:
                        failed.append(symbol)
                    
                    # Save checkpoint after each symbol
                    checkpoint["completed"] = completed
                    checkpoint["failed"] = failed
                    self.save_checkpoint(checkpoint)
                    
                    progress = len(completed) / len(symbols) * 100
                    logger.info(
                        f"Progress: {len(completed)}/{len(symbols)} "
                        f"({progress:.1f}%)"
                    )
                
                except Exception as e:
                    logger.error(f"✗ {symbol} execution error: {e}")
                    failed.append(symbol)
        
        elapsed = time.time() - start_time
        
        summary = {
            "total": len(symbols),
            "completed": len(completed),
            "failed": len(failed),
            "elapsed_minutes": elapsed / 60,
            "symbols_per_minute": len(completed) / (elapsed / 60) if elapsed > 0 else 0,
        }
        
        logger.info(
            f"\n✓ Backfill complete:\n"
            f"  Total: {summary['total']}\n"
            f"  Completed: {summary['completed']}\n"
            f"  Failed: {summary['failed']}\n"
            f"  Time: {summary['elapsed_minutes']:.1f} minutes\n"
            f"  Rate: {summary['symbols_per_minute']:.1f} symbols/min"
        )
        
        return summary


def main():
    """CLI entry point for backfill script."""
    parser = argparse.ArgumentParser(description="Historical data backfill")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., AAPL,GOOGL,MSFT)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to file containing symbols (one per line)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years of history (default: 10)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--asset-type",
        type=str,
        default="equity",
        choices=["equity", "crypto", "forex"],
        help="Asset type (default: equity)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (don't resume from checkpoint)",
    )
    
    args = parser.parse_args()
    
    # Get symbol list
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.file:
        with open(args.file) as f:
            symbols = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Must provide --symbols or --file")
    
    # Map asset type
    asset_type_map = {
        "equity": AssetType.EQUITY,
        "crypto": AssetType.CRYPTO,
        "forex": AssetType.FOREX,
    }
    asset_type = asset_type_map[args.asset_type]
    
    # Run backfill
    manager = BackfillManager()
    summary = manager.backfill_batch(
        symbols=symbols,
        asset_type=asset_type,
        years=args.years,
        workers=args.workers,
        resume=not args.no_resume,
    )
    
    return summary


if __name__ == "__main__":
    main()
