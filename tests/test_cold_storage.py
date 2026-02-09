"""
Tests for Cold Storage & Audit Trail Management

Tests partitioned Parquet storage, audit log compression, and retention policies.
"""

import gzip
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.cold_storage import ColdStorageManager, RetentionPolicy
from core.data_providers import OHLCV


class TestRetentionPolicy:
    """Test retention policy configuration."""
    
    def test_policy_creation(self):
        """Test creating retention policy."""
        policy = RetentionPolicy(hot_days=7, warm_days=90, cold_days=365)
        
        assert policy.hot_days == 7
        assert policy.warm_days == 90
        assert policy.cold_days == 365
    
    def test_get_tier(self):
        """Test tier classification."""
        policy = RetentionPolicy(hot_days=30, warm_days=180, cold_days=3650)
        
        assert policy.get_tier(15) == "hot"
        assert policy.get_tier(100) == "warm"
        assert policy.get_tier(500) == "cold"


class TestColdStorageManager:
    """Test cold storage operations."""
    
    def test_manager_creation(self):
        """Test creating storage manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            assert manager.data_path.exists()
            assert manager.audit_path.exists()
    
    def test_write_partition(self):
        """Test writing partitioned data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            data = [
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
            
            manager.write_partition("AAPL", data, year=2020, month=1)
            
            # Verify partition exists
            partition_path = Path(tmpdir) / "data" / "AAPL" / "2020" / "01" / "data.parquet"
            assert partition_path.exists()
    
    def test_read_partition(self):
        """Test reading partitioned data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            original_data = [
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
            
            manager.write_partition("AAPL", original_data, year=2020, month=1)
            
            loaded_data = manager.read_partition("AAPL", year=2020, month=1)
            
            assert len(loaded_data) == 10
            assert loaded_data[0].close == 103.0  # 102.0 + 1
    
    def test_read_multiple_partitions(self):
        """Test reading across multiple partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            # Write January data
            jan_data = [
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
            
            # Write February data
            feb_data = [
                OHLCV(
                    date=datetime(2020, 2, i),
                    open=110.0,
                    high=115.0,
                    low=105.0,
                    close=112.0,
                    volume=1000000,
                )
                for i in range(1, 11)
            ]
            
            manager.write_partition("AAPL", jan_data, year=2020, month=1)
            manager.write_partition("AAPL", feb_data, year=2020, month=2)
            
            # Read all 2020 data
            all_data = manager.read_partition("AAPL", year=2020)
            
            assert len(all_data) == 20
    
    def test_compress_audit_logs(self):
        """Test audit log compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            # Create mock audit log
            log_path = Path(tmpdir) / "test_audit.jsonl"
            entries = []
            
            # Old entries (2 months ago)
            for i in range(10):
                entries.append({
                    "timestamp": (datetime.now() - timedelta(days=60 + i)).isoformat(),
                    "symbol": f"TEST{i}",
                    "provider": "test",
                })
            
            # Recent entries (10 days ago)
            for i in range(5):
                entries.append({
                    "timestamp": (datetime.now() - timedelta(days=10 + i)).isoformat(),
                    "symbol": f"RECENT{i}",
                    "provider": "test",
                })
            
            with open(log_path, "w") as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
            
            # Compress logs
            manager.compress_audit_logs(log_path)
            
            # Verify current log only has recent entries
            with open(log_path) as f:
                current_entries = [json.loads(line) for line in f]
            
            assert len(current_entries) == 5
    
    def test_get_storage_stats(self):
        """Test storage statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            data = [
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
            
            manager.write_partition("AAPL", data, year=2020, month=1)
            manager.write_partition("GOOGL", data, year=2020, month=1)
            
            stats = manager.get_storage_stats()
            
            assert len(stats["symbols"]) == 2
            assert "AAPL" in stats["symbols"]
            assert "GOOGL" in stats["symbols"]
            assert stats["partition_count"] == 2
            assert stats["total_size_mb"] > 0


class TestIntegration:
    """Integration tests for cold storage workflows."""
    
    def test_write_read_roundtrip(self):
        """Test complete write/read cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            # Create year of daily data
            original_data = [
                OHLCV(
                    date=datetime(2020, 1, 1) + timedelta(days=i),
                    open=150.0 + i * 0.1,
                    high=155.0 + i * 0.1,
                    low=145.0 + i * 0.1,
                    close=152.0 + i * 0.1,
                    volume=10000000,
                )
                for i in range(365)
            ]
            
            # Write to monthly partitions
            for month in range(1, 13):
                month_data = [
                    bar for bar in original_data
                    if bar.date.month == month
                ]
                manager.write_partition("AAPL", month_data, year=2020, month=month)
            
            # Read all 2020 data
            loaded_data = manager.read_partition("AAPL", year=2020)
            
            assert len(loaded_data) == 365
            assert loaded_data[0].date == original_data[0].date
            assert loaded_data[-1].date == original_data[-1].date
    
    def test_date_filters(self):
        """Test reading with date filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ColdStorageManager(base_path=Path(tmpdir))
            
            data = [
                OHLCV(
                    date=datetime(2020, 1, 1) + timedelta(days=i),
                    open=100.0,
                    high=105.0,
                    low=95.0,
                    close=102.0,
                    volume=1000000,
                )
                for i in range(365)
            ]
            
            # Write all data
            for month in range(1, 13):
                month_data = [bar for bar in data if bar.date.month == month]
                manager.write_partition("AAPL", month_data, year=2020, month=month)
            
            # Read Q1 only
            q1_data = manager.read_partition(
                "AAPL",
                year=2020,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 3, 31),
            )
            
            assert len(q1_data) == 91  # Jan (31) + Feb (29 in 2020) + Mar (31)
