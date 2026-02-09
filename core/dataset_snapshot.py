"""
Point-in-Time Dataset Snapshots

Provides versioned, immutable snapshots of financial datasets for:
- Backtesting with exact historical data states
- Reproducible research and analysis
- Data lineage tracking
- Compliance and auditing

Features:
- Save/load snapshots to Parquet with compression
- Metadata tracking (ID, timestamp, symbols, date ranges)
- Query API (list snapshots, get by ID, search by metadata)
- Snapshot versioning and immutability guarantees
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data_providers import OHLCV, AssetType

logger = logging.getLogger(__name__)


@dataclass
class SnapshotMetadata:
    """Metadata for a dataset snapshot."""
    
    snapshot_id: str
    created_at: datetime
    symbols: List[str]
    asset_types: List[str]
    start_date: datetime
    end_date: datetime
    interval: str
    row_count: int
    providers_used: List[str]
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at.isoformat(),
            "symbols": self.symbols,
            "asset_types": self.asset_types,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "interval": self.interval,
            "row_count": self.row_count,
            "providers_used": self.providers_used,
            "description": self.description,
            "tags": self.tags or [],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            symbols=data["symbols"],
            asset_types=data["asset_types"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            interval=data["interval"],
            row_count=data["row_count"],
            providers_used=data["providers_used"],
            description=data.get("description"),
            tags=data.get("tags", []),
        )


class DatasetSnapshot:
    """
    Point-in-time snapshot of financial data.
    
    Provides versioned, immutable storage of OHLCV data with full metadata
    tracking for reproducible backtesting and analysis.
    
    Storage Format:
    - Data: Parquet files (compressed with snappy)
    - Metadata: JSON sidecar files
    - Structure: snapshots/{snapshot_id}/data.parquet + metadata.json
    """
    
    def __init__(self, base_path: Path = Path("data/snapshots")):
        """
        Initialize snapshot manager.
        
        Args:
            base_path: Root directory for snapshot storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Snapshot manager initialized: {self.base_path}")
    
    def save_snapshot(
        self,
        data: Dict[str, List[OHLCV]],  # Changed: symbol -> List[OHLCV]
        snapshot_id: str,
        asset_types: List[AssetType],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        providers_used: Optional[List[str]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> SnapshotMetadata:
        """
        Save a dataset snapshot to Parquet.
        
        Args:
            data: Dict mapping symbol -> List[OHLCV bars]
            snapshot_id: Unique identifier for snapshot
            asset_types: Asset types included
            start_date: Data start date
            end_date: Data end date
            interval: Bar interval (e.g., "1d", "1h")
            providers_used: Data providers used to fetch data
            description: Human-readable description
            tags: Tags for categorization
            
        Returns:
            SnapshotMetadata object
        """
        snapshot_dir = self.base_path / snapshot_id
        
        # Check if snapshot already exists
        if snapshot_dir.exists():
            raise ValueError(f"Snapshot {snapshot_id} already exists (immutable)")
        
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert OHLCV to DataFrame
        rows = []
        for symbol, bars in data.items():
            for bar in bars:
                rows.append({
                    "symbol": symbol,
                    "date": bar.date,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "adjusted_close": bar.adjusted_close,
                })
        
        df = pd.DataFrame(rows)
        
        # Save to Parquet with compression
        parquet_path = snapshot_dir / "data.parquet"
        df.to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )
        
        # Extract symbols from data
        symbols = list(data.keys())
        
        # Create metadata
        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            created_at=datetime.now(),
            symbols=symbols,
            asset_types=[at.value for at in asset_types],
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            row_count=len(df),
            providers_used=providers_used or [],
            description=description,
            tags=tags,
        )
        
        # Save metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        file_size = parquet_path.stat().st_size / 1024 / 1024  # MB
        logger.info(
            f"✓ Snapshot saved: {snapshot_id} "
            f"({len(df):,} rows, {len(symbols)} symbols, {file_size:.2f}MB)"
        )
        
        return metadata
    
    def load_snapshot(
        self,
        snapshot_id: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[Dict[str, List[OHLCV]], SnapshotMetadata]:
        """
        Load a snapshot from Parquet.
        
        Args:
            snapshot_id: Snapshot identifier
            symbols: Optional list of symbols to filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            (Dict of symbol -> List[OHLCV bars], metadata)
        """
        snapshot_dir = self.base_path / snapshot_id
        
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        # Load metadata
        metadata_path = snapshot_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = SnapshotMetadata.from_dict(json.load(f))
        
        # Load Parquet
        parquet_path = snapshot_dir / "data.parquet"
        df = pd.read_parquet(parquet_path, engine="pyarrow")
        
        # Apply filters
        if symbols:
            df = df[df["symbol"].isin(symbols)]
        
        if start_date:
            df = df[df["date"] >= start_date]
        
        if end_date:
            df = df[df["date"] <= end_date]
        
        # Convert to Dict[symbol -> List[OHLCV]]
        result = {}
        for symbol in df["symbol"].unique():
            symbol_df = df[df["symbol"] == symbol]
            bars = []
            for _, row in symbol_df.iterrows():
                bars.append(OHLCV(
                    date=row["date"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    adjusted_close=row.get("adjusted_close"),
                ))
            result[symbol] = bars
        
        total_bars = sum(len(bars) for bars in result.values())
        logger.info(f"✓ Snapshot loaded: {snapshot_id} ({total_bars:,} bars, {len(result)} symbols)")
        return result, metadata
    
    def list_snapshots(
        self,
        tags: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        start_after: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SnapshotMetadata]:
        """
        List available snapshots with optional filtering.
        
        Args:
            tags: Filter by tags (snapshots with ANY of these tags)
            symbols: Filter by symbols (snapshots containing ANY of these symbols)
            start_after: Filter by creation date (snapshots created after this date)
            limit: Maximum number of results
            
        Returns:
            List of SnapshotMetadata objects (sorted by created_at desc)
        """
        snapshots = []
        
        for snapshot_dir in self.base_path.iterdir():
            if not snapshot_dir.is_dir():
                continue
            
            metadata_path = snapshot_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path) as f:
                    metadata = SnapshotMetadata.from_dict(json.load(f))
                
                # Apply filters
                if tags and not any(tag in (metadata.tags or []) for tag in tags):
                    continue
                
                if symbols and not any(sym in metadata.symbols for sym in symbols):
                    continue
                
                if start_after and metadata.created_at <= start_after:
                    continue
                
                snapshots.append(metadata)
            
            except Exception as e:
                logger.warning(f"Failed to load metadata for {snapshot_dir.name}: {e}")
                continue
        
        # Sort by created_at desc
        snapshots.sort(key=lambda m: m.created_at, reverse=True)
        
        return snapshots[:limit]
    
    def get_metadata(self, snapshot_id: str) -> SnapshotMetadata:
        """
        Get metadata for a snapshot without loading data.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            SnapshotMetadata object
        """
        snapshot_dir = self.base_path / snapshot_id
        metadata_path = snapshot_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        with open(metadata_path) as f:
            return SnapshotMetadata.from_dict(json.load(f))
    
    def delete_snapshot(self, snapshot_id: str):
        """
        Delete a snapshot (use with caution - violates immutability).
        
        Args:
            snapshot_id: Snapshot identifier
        """
        import shutil
        
        snapshot_dir = self.base_path / snapshot_id
        
        if not snapshot_dir.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        shutil.rmtree(snapshot_dir)
        logger.warning(f"✗ Snapshot deleted: {snapshot_id}")
    
    def get_snapshot_size(self, snapshot_id: str) -> float:
        """
        Get size of snapshot in MB.
        
        Args:
            snapshot_id: Snapshot identifier
            
        Returns:
            Size in MB
        """
        snapshot_dir = self.base_path / snapshot_id
        parquet_path = snapshot_dir / "data.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        return parquet_path.stat().st_size / 1024 / 1024
