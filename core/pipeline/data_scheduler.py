"""
Data Scheduler for Automated Updates
Scheduled data fetching and caching
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum
import schedule
import time
import threading
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')


class UpdateFrequency(Enum):
    """Update frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    INTRADAY = "intraday"
    REALTIME = "realtime"


class UpdateJob:
    """
    Represents a scheduled update job.
    """
    
    def __init__(self,
                 job_id: str,
                 name: str,
                 function: Callable,
                 frequency: UpdateFrequency,
                 time_of_day: Optional[str] = None,
                 **kwargs):
        """
        Initialize update job.
        
        Args:
            job_id: Unique job identifier
            name: Human-readable job name
            function: Function to execute
            frequency: Update frequency
            time_of_day: Time to execute (HH:MM format)
            **kwargs: Arguments to pass to function
        """
        self.job_id = job_id
        self.name = name
        self.function = function
        self.frequency = frequency
        self.time_of_day = time_of_day
        self.kwargs = kwargs
        
        self.last_run = None
        self.next_run = None
        self.status = "pending"
        self.error_count = 0
        self.success_count = 0
        
        self.logger = logging.getLogger(f"Job-{job_id}")
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the job.
        
        Returns:
            Dictionary with execution results
        """
        try:
            start_time = datetime.now()
            
            # Execute function
            result = self.function(**self.kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            self.last_run = datetime.now()
            self.status = "success"
            self.success_count += 1
            
            self.logger.info(f"Job {self.name} executed successfully in {duration:.2f}s")
            
            return {
                'job_id': self.job_id,
                'status': 'success',
                'timestamp': start_time,
                'duration': duration,
                'result': result
            }
            
        except Exception as e:
            self.status = "error"
            self.error_count += 1
            
            self.logger.error(f"Job {self.name} failed: {str(e)}")
            
            return {
                'job_id': self.job_id,
                'status': 'error',
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def get_status(self) -> Dict:
        """
        Get job status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'job_id': self.job_id,
            'name': self.name,
            'status': self.status,
            'frequency': self.frequency.value,
            'last_run': self.last_run,
            'next_run': self.next_run,
            'success_count': self.success_count,
            'error_count': self.error_count
        }


class DataScheduler:
    """
    Manages scheduled data updates.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize data scheduler.
        
        Args:
            max_workers: Maximum concurrent jobs
        """
        self.jobs = {}
        self.max_workers = max_workers
        self.is_running = False
        self.scheduler_thread = None
        
        self.logger = logging.getLogger("DataScheduler")
        
        # Setup logging
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def add_job(self,
               job: UpdateJob) -> bool:
        """
        Add a job to the scheduler.
        
        Args:
            job: UpdateJob instance
        
        Returns:
            True if added successfully
        """
        try:
            self.jobs[job.job_id] = job
            
            # Schedule the job
            self._schedule_job(job)
            
            self.logger.info(f"Added job: {job.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add job: {str(e)}")
            return False
    
    def _schedule_job(self, job: UpdateJob):
        """
        Schedule a job with the scheduler.
        
        Args:
            job: UpdateJob to schedule
        """
        if job.frequency == UpdateFrequency.DAILY:
            if job.time_of_day:
                schedule.every().day.at(job.time_of_day).do(job.execute)
            else:
                schedule.every().day.do(job.execute)
        
        elif job.frequency == UpdateFrequency.WEEKLY:
            schedule.every().week.do(job.execute)
        
        elif job.frequency == UpdateFrequency.MONTHLY:
            schedule.every(30).days.do(job.execute)
        
        elif job.frequency == UpdateFrequency.QUARTERLY:
            schedule.every(90).days.do(job.execute)
        
        elif job.frequency == UpdateFrequency.INTRADAY:
            schedule.every(4).hours.do(job.execute)
    
    def remove_job(self, job_id: str) -> bool:
        """
        Remove a job from the scheduler.
        
        Args:
            job_id: ID of job to remove
        
        Returns:
            True if removed
        """
        if job_id in self.jobs:
            del self.jobs[job_id]
            self.logger.info(f"Removed job: {job_id}")
            return True
        return False
    
    def start(self):
        """Start the scheduler."""
        if self.is_running:
            self.logger.warning("Scheduler already running")
            return
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self.is_running = False
        schedule.clear()
        
        self.logger.info("Scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop."""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def get_status(self) -> Dict:
        """
        Get scheduler status.
        
        Returns:
            Dictionary with scheduler info
        """
        return {
            'is_running': self.is_running,
            'total_jobs': len(self.jobs),
            'jobs': [job.get_status() for job in self.jobs.values()]
        }
    
    def manual_run(self, job_id: str) -> Dict:
        """
        Manually run a job.
        
        Args:
            job_id: ID of job to run
        
        Returns:
            Execution result
        """
        if job_id not in self.jobs:
            return {'error': f'Job {job_id} not found'}
        
        job = self.jobs[job_id]
        return job.execute()


class UpdateJobBuilder:
    """
    Builder for creating update jobs easily.
    """
    
    @staticmethod
    def stock_data_update(tickers: List[str],
                         frequency: UpdateFrequency = UpdateFrequency.DAILY) -> UpdateJob:
        """
        Create a stock data update job.
        
        Args:
            tickers: List of stock tickers
            frequency: Update frequency
        
        Returns:
            UpdateJob instance
        """
        def fetch_stock_data(**kwargs):
            import yfinance as yf
            results = {}
            for ticker in kwargs['tickers']:
                try:
                    data = yf.download(ticker, period='1d', progress=False)
                    results[ticker] = {
                        'price': data['Close'].iloc[-1],
                        'timestamp': datetime.now()
                    }
                except Exception as e:
                    results[ticker] = {'error': str(e)}
            return results
        
        return UpdateJob(
            job_id=f"stock_data_{'-'.join(tickers)}",
            name=f"Update stock data: {', '.join(tickers)}",
            function=fetch_stock_data,
            frequency=frequency,
            time_of_day="16:30",  # After market close
            tickers=tickers
        )
    
    @staticmethod
    def economic_data_update(indicators: List[str],
                            fred_key: Optional[str] = None,
                            frequency: UpdateFrequency = UpdateFrequency.WEEKLY) -> UpdateJob:
        """
        Create an economic data update job.
        
        Args:
            indicators: List of FRED series IDs
            fred_key: FRED API key
            frequency: Update frequency
        
        Returns:
            UpdateJob instance
        """
        def fetch_economic_data(**kwargs):
            import os
            try:
                from fredapi import Fred
                
                api_key = kwargs['fred_key'] or os.getenv('FRED_API_KEY')
                if not api_key:
                    return {'error': 'FRED API key not provided'}
                
                fred = Fred(api_key=api_key)
                results = {}
                
                for series_id in kwargs['indicators']:
                    try:
                        data = fred.get_series(series_id)
                        results[series_id] = {
                            'value': data.iloc[-1],
                            'date': data.index[-1]
                        }
                    except Exception as e:
                        results[series_id] = {'error': str(e)}
                
                return results
                
            except ImportError:
                return {'error': 'fredapi not installed'}
        
        return UpdateJob(
            job_id=f"econ_data_{len(indicators)}_indicators",
            name=f"Update {len(indicators)} economic indicators",
            function=fetch_economic_data,
            frequency=frequency,
            time_of_day="08:00",  # After data release
            indicators=indicators,
            fred_key=fred_key
        )
    
    @staticmethod
    def portfolio_rebalance(portfolio_tickers: List[str],
                           rebalance_frequency: UpdateFrequency = UpdateFrequency.MONTHLY) -> UpdateJob:
        """
        Create a portfolio rebalancing job.
        
        Args:
            portfolio_tickers: List of portfolio holdings
            rebalance_frequency: Rebalancing frequency
        
        Returns:
            UpdateJob instance
        """
        def rebalance_portfolio(**kwargs):
            import yfinance as yf
            
            tickers = kwargs['portfolio_tickers']
            
            try:
                # Get current prices
                data = yf.download(tickers, period='1d', progress=False)['Close']
                
                # Calculate current weights (equal-weight for simplicity)
                current_prices = data.iloc[-1]
                market_values = current_prices  # Simplified
                total_value = market_values.sum()
                current_weights = market_values / total_value
                
                # Target weights (equal-weight)
                target_weights = pd.Series(1/len(tickers), index=tickers)
                
                # Calculate rebalancing needed
                rebalance_amounts = target_weights - current_weights
                
                return {
                    'current_weights': current_weights.to_dict(),
                    'target_weights': target_weights.to_dict(),
                    'rebalance_amounts': rebalance_amounts.to_dict(),
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                return {'error': str(e)}
        
        return UpdateJob(
            job_id=f"portfolio_rebalance_{len(portfolio_tickers)}",
            name=f"Rebalance portfolio: {len(portfolio_tickers)} holdings",
            function=rebalance_portfolio,
            frequency=rebalance_frequency,
            time_of_day="09:00",
            portfolio_tickers=portfolio_tickers
        )
