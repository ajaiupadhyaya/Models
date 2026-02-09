"""
Cold Storage & Audit Trail Management

Provides utilities for:
- Partitioned Parquet storage (by symbol/year/month)
- Audit log compression and archival
- Data retention policy management
- Hot/warm/cold tier management

Storage Structure:
cold_storage/
├── data/
│   ├── AAPL/
│   │   ├── 2020/
│   │   │   ├── 01/
│   │   │   │   └── data.parquet
│   │   │   ├── 02/
│   │   │   └── ...
│   │   └── 2021/
│   └── GOOGL/
└── audit/
    ├── 2025-01.jsonl.gz
    ├── 2025-02.jsonl.gz
    └── current.jsonl
"""

import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_providers import OHLCV

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    
    hot_days: int = 30  # Keep in fast storage for N days
    warm_days: int = 180  # Keep in medium storage for N days
    cold_days: int = 3650  # Keep in cold storage for N days (10 years)
    
    def get_tier(self, age_days: int) -> str:
        """Get storage tier based on data age."""
        if age_days <= self.hot_days:
            return "hot"
        elif age_days <= self.warm_days:
            return "warm"
        else:
            return "cold"


class ColdStorageManager:
    """
    Manages partitioned cold storage and audit log archival.
    
    Features:
    - Partitioned Parquet writes (symbol/year/month)
    - Audit log compression (monthly archives)
    - Retention policy enforcement
    - Storage tier management (hot/warm/cold)
    """
    
    def __init__(
        self,
        base_path: Path = Path("data/cold_storage"),
        retention_policy: Optional[RetentionPolicy] = None,
    ):
        """
        Initialize cold storage manager.
        
        Args:
            base_path: Root directory for cold storage
            retention_policy: Data retention configuration
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.audit_path = self.base_path / "audit"
        
        # Create directories
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.audit_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_policy = retention_policy or RetentionPolicy()
        
        logger.info(f"Cold storage initialized: {self.base_path}")
    
    def write_partition(
        self,
        symbol: str,
        data: List[OHLCV],
        year: Optional[int] = None,
        month: Optional[int] = None,
    ):
        """
        Write data to partitioned Parquet storage.
        
        Partition structure: data/{symbol}/{year}/{month}/data.parquet
        
        Args:
            symbol: Symbol to store
            data: List of OHLCV bars
            year: Year partition (defaults to current year)
            month: Month partition (defaults to current month)
        """
        if not data:
            logger.warning(f"No data to write for {symbol}")
            return
        
        year = year or datetime.now().year
        month = month or datetime.now().month
        
        # Create partition directory
        partition_dir = self.data_path / symbol / str(year) / f"{month:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame
        rows = []
        for bar in data:
            rows.append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "adjusted_close": bar.adjusted_close,
            })
        
        df = pd.DataFrame(rows)
        
        # Write to Parquet (append if exists)
        parquet_path = partition_dir / "data.parquet"
        
        if parquet_path.exists():
            # Load existing data and append
            existing_df = pd.read_parquet(parquet_path)
            df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates based on date
            df = df.drop_duplicates(subset=["date"], keep="last")
            df = df.sort_values("date")
        
        # Write to Parquet
        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        
        file_size = parquet_path.stat().st_size / 1024  # KB
        logger.info(
            f"✓ Wrote {len(data)} bars to {symbol}/{year}/{month:02d} "
            f"({file_size:.1f}KB)"
        )
    
    def read_partition(
        self,
        symbol: str,
        year: Optional[int] = None,
        month: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[OHLCV]:
        """
        Read data from partitioned storage.
        
        Args:
            symbol: Symbol to read
            year: Year partition (if None, reads all years)
            month: Month partition (if None, reads all months in year)
            start_date: Optional date filter
            end_date: Optional date filter
            
        Returns:
            List of OHLCV bars
        """
        symbol_dir = self.data_path / symbol
        
        if not symbol_dir.exists():
            logger.warning(f"No data found for {symbol}")
            return []
        
        # Collect all matching partitions
        partitions = []
        
        if year and month:
            # Single partition
            partitions.append(symbol_dir / str(year) / f"{month:02d}")
        elif year:
            # All months in year
            year_dir = symbol_dir / str(year)
            if year_dir.exists():
                partitions.extend(year_dir.iterdir())
        else:
            # All years and months
            for year_dir in symbol_dir.iterdir():
                if year_dir.is_dir():
                    partitions.extend(year_dir.iterdir())
        
        # Read data from all partitions
        dfs = []
        for partition in partitions:
            parquet_path = partition / "data.parquet"
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                dfs.append(df)
        
        if not dfs:
            logger.warning(f"No data files found for {symbol}")
            return []
        
        # Combine all partitions
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=["date"], keep="last")
        combined_df = combined_df.sort_values("date")
        
        # Apply date filters
        if start_date:
            combined_df = combined_df[combined_df["date"] >= start_date]
        if end_date:
            combined_df = combined_df[combined_df["date"] <= end_date]
        
        # Convert to OHLCV objects
        bars = []
        for _, row in combined_df.iterrows():
            bars.append(OHLCV(
                date=row["date"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                adjusted_close=row.get("adjusted_close"),
            ))
        
        logger.info(f"✓ Read {len(bars)} bars for {symbol}")
        return bars
    
    def compress_audit_logs(
        self,
        source_path: Path,
        archive_before: Optional[datetime] = None,
    ):
        """
        Compress old audit logs to monthly archives.
        
        Reads from source_path (e.g., logs/fetch_audit.jsonl),
        archives entries older than archive_before to monthly .gz files.
        
        Args:
            source_path: Path to current audit log file
            archive_before: Archive entries before this date (default: 30 days ago)
        """
        if not source_path.exists():
            logger.warning(f"Audit log not found: {source_path}")
            return
        
        archive_before = archive_before or datetime.now() - timedelta(days=30)
        
        # Read current log
        entries = []
        with open(source_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in audit log: {line[:100]}")
        
        # Group entries by month
        monthly_entries: Dict[str, List] = {}
        current_entries = []
        
        for entry in entries:
            try:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp < archive_before:
                    month_key = timestamp.strftime("%Y-%m")
                    if month_key not in monthly_entries:
                        monthly_entries[month_key] = []
                    monthly_entries[month_key].append(entry)
                else:
                    current_entries.append(entry)
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid entry timestamp: {e}")
                current_entries.append(entry)
        
        # Write monthly archives
        for month_key, month_entries in monthly_entries.items():
            archive_path = self.audit_path / f"{month_key}.jsonl.gz"
            
            # Append to existing archive if it exists
            existing_entries = []
            if archive_path.exists():
                with gzip.open(archive_path, "rt") as f:
                    for line in f:
                        try:
                            existing_entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            
            all_entries = existing_entries + month_entries
            
            with gzip.open(archive_path, "wt") as f:
                for entry in all_entries:
                    f.write(json.dumps(entry) + "\n")
            
            logger.info(f"✓ Archived {len(month_entries)} entries to {month_key}.jsonl.gz")
        
        # Rewrite current log with only recent entries
        with open(source_path, "w") as f:
            for entry in current_entries:
                f.write(json.dumps(entry) + "\n")
        
        logger.info(f"✓ Kept {len(current_entries)} current entries in {source_path}")
    
    def get_storage_stats(self) -> Dict[str, any]:
        """
        Get cold storage statistics.
        
        Returns:
            Dict with storage stats (symbols, total size, partition count, etc.)
        """
        stats = {
            "symbols": [],
            "total_size_mb": 0.0,
            "partition_count": 0,
            "audit_archives": 0,
            "audit_size_mb": 0.0,
        }
        
        # Data stats
        for symbol_dir in self.data_path.iterdir():
            if symbol_dir.is_dir():
                stats["symbols"].append(symbol_dir.name)
                
                for year_dir in symbol_dir.iterdir():
                    if year_dir.is_dir():
                        for month_dir in year_dir.iterdir():
                            if month_dir.is_dir():
                                parquet_path = month_dir / "data.parquet"
                                if parquet_path.exists():
                                    stats["partition_count"] += 1
                                    stats["total_size_mb"] += parquet_path.stat().st_size / 1024 / 1024
        
        # Audit stats
        for audit_file in self.audit_path.iterdir():
            if audit_file.suffix in [".gz", ".jsonl"]:
                stats["audit_archives"] += 1
                stats["audit_size_mb"] += audit_file.stat().st_size / 1024 / 1024
        
        return stats
    
    def cleanup_old_data(self, days_threshold: int = 3650):
        """
        Delete data older than threshold (default: 10 years).
        
        Args:
            days_threshold: Delete data older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        deleted_count = 0
        
        for symbol_dir in self.data_path.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            for year_dir in symbol_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    parquet_path = month_dir / "data.parquet"
                    if not parquet_path.exists():
                        continue
                    
                    # Check if all data in partition is older than cutoff
                    df = pd.read_parquet(parquet_path)
                    if df["date"].max() < cutoff_date:
                        shutil.rmtree(month_dir)
                        deleted_count += 1
                        logger.info(f"✗ Deleted old partition: {symbol_dir.name}/{year_dir.name}/{month_dir.name}")
        
        logger.info(f"✓ Cleanup complete: deleted {deleted_count} old partitions")
        return deleted_count
