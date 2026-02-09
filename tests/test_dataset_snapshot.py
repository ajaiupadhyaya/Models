"""
Tests for Point-in-Time Dataset Snapshots

Tests snapshot creation, loading, querying, and metadata management.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pandas as pd

from core.dataset_snapshot import DatasetSnapshot, SnapshotMetadata
from core.data_providers import OHLCV, AssetType


class TestSnapshotMetadata:
    """Test metadata creation and serialization."""
    
    def test_metadata_creation(self):
        """Test creating metadata."""
        metadata = SnapshotMetadata(
            snapshot_id="test_001",
            created_at=datetime.now(),
            symbols=["AAPL", "GOOGL"],
            asset_types=["EQUITY", "EQUITY"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            interval="1d",
            row_count=500,
            providers_used=["polygon", "iex"],
            description="Test snapshot",
            tags=["test", "equities"],
        )
        
        assert metadata.snapshot_id == "test_001"
        assert len(metadata.symbols) == 2
        assert metadata.row_count == 500
    
    def test_metadata_to_dict(self):
        """Test metadata serialization to dict."""
        metadata = SnapshotMetadata(
            snapshot_id="test_002",
            created_at=datetime(2025, 1, 1, 12, 0),
            symbols=["BTC-USD"],
            asset_types=["CRYPTO"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            interval="1h",
            row_count=8760,
            providers_used=["coingecko"],
        )
        
        data = metadata.to_dict()
        
        assert data["snapshot_id"] == "test_002"
        assert data["created_at"] == "2025-01-01T12:00:00"
        assert data["symbols"] == ["BTC-USD"]
    
    def test_metadata_from_dict(self):
        """Test metadata deserialization from dict."""
        data = {
            "snapshot_id": "test_003",
            "created_at": "2025-01-01T12:00:00",
            "symbols": ["AAPL"],
            "asset_types": ["EQUITY"],
            "start_date": "2020-01-01T00:00:00",
            "end_date": "2020-12-31T00:00:00",
            "interval": "1d",
            "row_count": 252,
            "providers_used": ["polygon"],
            "description": "Test",
            "tags": ["test"],
        }
        
        metadata = SnapshotMetadata.from_dict(data)
        
        assert metadata.snapshot_id == "test_003"
        assert metadata.created_at == datetime(2025, 1, 1, 12, 0)
        assert metadata.symbols == ["AAPL"]


class TestDatasetSnapshot:
    """Test snapshot save/load operations."""
    
    def test_snapshot_creation(self):
        """Test creating snapshot manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            assert manager.base_path.exists()
    
    def test_save_snapshot(self):
        """Test saving a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            # Create sample data
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, i),
                        open=100.0 + i,
                        high=105.0 + i,
                        low=95.0 + i,
                        close=102.0 + i,
                        volume=1000000,
                    )
                    for i in range(1, 11)
                ]
            }
            
            metadata = manager.save_snapshot(
                data=data,
                snapshot_id="test_save",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 10),
                interval="1d",
                providers_used=["polygon"],
                description="Test snapshot",
                tags=["test"],
            )
            
            assert metadata.snapshot_id == "test_save"
            assert metadata.row_count == 10
            assert len(metadata.symbols) == 1
            
            # Verify files exist
            snapshot_dir = Path(tmpdir) / "test_save"
            assert (snapshot_dir / "data.parquet").exists()
            assert (snapshot_dir / "metadata.json").exists()
    
    def test_save_duplicate_snapshot_fails(self):
        """Test that saving duplicate snapshot raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                ]
            }
            
            # First save should succeed
            manager.save_snapshot(
                data=data,
                snapshot_id="dup_test",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 1),
                interval="1d",
            )
            
            # Second save should fail (immutability)
            with pytest.raises(ValueError, match="already exists"):
                manager.save_snapshot(
                    data=data,
                    snapshot_id="dup_test",
                    asset_types=[AssetType.EQUITY],
                    start_date=datetime(2020, 1, 1),
                    end_date=datetime(2020, 1, 1),
                    interval="1d",
                )
    
    def test_load_snapshot(self):
        """Test loading a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            # Create and save
            original_data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, i),
                        open=100.0 + i,
                        high=105.0 + i,
                        low=95.0 + i,
                        close=102.0 + i,
                        volume=1000000,
                    )
                    for i in range(1, 6)
                ]
            }
            
            manager.save_snapshot(
                data=original_data,
                snapshot_id="test_load",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 5),
                interval="1d",
            )
            
            # Load and verify
            loaded_data, metadata = manager.load_snapshot("test_load")
            
            assert len(loaded_data["AAPL"]) == 5
            assert loaded_data["AAPL"][0].close == 103.0  # close = 102.0 + i where i=1
            assert metadata.snapshot_id == "test_load"
    
    def test_load_snapshot_with_filters(self):
        """Test loading snapshot with date/symbol filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            # Create multi-symbol data
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, i),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                    for i in range(1, 11)
                ],
                "GOOGL": [
                    OHLCV(
                        date=datetime(2020, 1, i),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                    for i in range(1, 11)
                ]
            }
            
            manager.save_snapshot(
                data=data,
                snapshot_id="test_filter",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 10),
                interval="1d",
            )
            
            # Load with symbol filter
            loaded, _ = manager.load_snapshot(
                "test_filter",
                symbols=["AAPL"],
            )
            assert len(loaded["AAPL"]) == 10
            assert "GOOGL" not in loaded
            
            # Load with date filter
            loaded, _ = manager.load_snapshot(
                "test_filter",
                start_date=datetime(2020, 1, 5),
                end_date=datetime(2020, 1, 7),
            )
            total_bars = sum(len(bars) for bars in loaded.values())
            assert total_bars == 6  # 3 days * 2 symbols
    
    def test_list_snapshots(self):
        """Test listing snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            # Create multiple snapshots
            data_equity = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                ]
            }
            
            data_crypto = {
                "BTC-USD": [
                    OHLCV(
                        date=datetime(2020, 1, 1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                ]
            }
            
            manager.save_snapshot(
                data=data_equity,
                snapshot_id="snap_1",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 1),
                interval="1d",
                tags=["test", "equities"],
            )
            
            manager.save_snapshot(
                data=data_crypto,
                snapshot_id="snap_2",
                asset_types=[AssetType.CRYPTO],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 1),
                interval="1h",
                tags=["test", "crypto"],
            )
            
            # List all
            snapshots = manager.list_snapshots()
            assert len(snapshots) == 2
            
            # Filter by tags
            snapshots = manager.list_snapshots(tags=["crypto"])
            assert len(snapshots) == 1
            assert snapshots[0].snapshot_id == "snap_2"
            
            # Filter by symbols
            snapshots = manager.list_snapshots(symbols=["AAPL"])
            assert len(snapshots) == 1
            assert snapshots[0].snapshot_id == "snap_1"
    
    def test_get_metadata(self):
        """Test getting metadata without loading data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                ]
            }
            
            manager.save_snapshot(
                data=data,
                snapshot_id="meta_test",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 1),
                interval="1d",
                description="Metadata test",
            )
            
            metadata = manager.get_metadata("meta_test")
            
            assert metadata.snapshot_id == "meta_test"
            assert metadata.description == "Metadata test"
            assert metadata.row_count == 1
    
    def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                ]
            }
            
            manager.save_snapshot(
                data=data,
                snapshot_id="delete_test",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 1, 1),
                interval="1d",
            )
            
            # Verify exists
            assert (Path(tmpdir) / "delete_test").exists()
            
            # Delete
            manager.delete_snapshot("delete_test")
            
            # Verify deleted
            assert not (Path(tmpdir) / "delete_test").exists()
    
    def test_get_snapshot_size(self):
        """Test getting snapshot size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1) + timedelta(days=i),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=102.0,
                        volume=1000000,
                    )
                    for i in range(100)
                ]
            }
            
            manager.save_snapshot(
                data=data,
                snapshot_id="size_test",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 4, 10),
                interval="1d",
            )
            
            size_mb = manager.get_snapshot_size("size_test")
            
            assert size_mb > 0
            assert size_mb < 1.0  # Should be small for 100 rows


class TestSnapshotIntegration:
    """Integration tests for snapshot workflows."""
    
    def test_roundtrip_snapshot(self):
        """Test complete save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DatasetSnapshot(base_path=Path(tmpdir))
            
            # Create realistic data
            original_data = {
                "AAPL": [
                    OHLCV(
                        date=datetime(2020, 1, 1) + timedelta(days=i),
                        open=150.0 + i * 0.5,
                        high=155.0 + i * 0.5,
                        low=145.0 + i * 0.5,
                        close=152.0 + i * 0.5,
                        volume=10000000,
                        adjusted_close=152.0 + i * 0.5,
                    )
                    for i in range(252)  # 1 year of daily data
                ]
            }
            
            # Save
            save_metadata = manager.save_snapshot(
                data=original_data,
                snapshot_id="roundtrip_test",
                asset_types=[AssetType.EQUITY],
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31),
                interval="1d",
                providers_used=["polygon"],
                description="Full year AAPL data",
                tags=["backtesting", "equities", "2020"],
            )
            
            # Load
            loaded_data, load_metadata = manager.load_snapshot("roundtrip_test")
            
            # Verify data integrity
            assert len(loaded_data["AAPL"]) == len(original_data["AAPL"])
            assert loaded_data["AAPL"][0].close == original_data["AAPL"][0].close
            assert loaded_data["AAPL"][-1].close == original_data["AAPL"][-1].close
            
            # Verify metadata
            assert load_metadata.snapshot_id == save_metadata.snapshot_id
            assert load_metadata.row_count == save_metadata.row_count
            assert load_metadata.description == save_metadata.description
