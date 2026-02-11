#!/usr/bin/env python3
"""
Final Project Audit and Cleanup
Ensures everything is tidy, clean, and production-ready
"""

import sys
import os
from pathlib import Path
import logging
import json
import shutil
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent


class ProjectAuditor:
    """Comprehensive project audit and cleanup."""
    
    def __init__(self):
        """Initialize auditor."""
        self.issues_found = []
        self.fixes_applied = []
        self.cleanup_items = []
    
    def check_duplicate_files(self) -> List[str]:
        """Check for duplicate or unnecessary files."""
        duplicates = []
        
        # Check for duplicate test/validation files
        test_files = list(project_root.glob("*test*.py")) + list(project_root.glob("*audit*.py")) + list(project_root.glob("*validate*.py"))
        
        # Keep essential ones, flag others
        essential = {'test_all_components.py', 'validate_publication_ready.py', 'test_integration.py'}
        
        for test_file in test_files:
            if test_file.name not in essential and test_file.name.startswith('test_'):
                # Check if it's a duplicate
                duplicates.append(str(test_file))
        
        return duplicates
    
    def check_unused_imports(self) -> List[str]:
        """Check for common unused imports (basic check)."""
        # This is a simplified check - full analysis would require AST parsing
        return []
    
    def check_missing_init_files(self) -> List[str]:
        """Check for missing __init__.py files."""
        missing = []
        
        directories_that_need_init = [
            'core',
            'models',
            'api',
            'automation',
            'core/advanced_viz',
            'core/pipeline',
            'models/ml',
            'models/macro',
            'models/risk',
            'models/trading',
            'models/portfolio',
            'models/valuation',
            'models/options',
            'models/sentiment',
            'models/fundamental',
            'models/fixed_income',
        ]
        
        for dir_path in directories_that_need_init:
            init_file = project_root / dir_path / '__init__.py'
            if not init_file.exists():
                missing.append(str(init_file))
                # Create it
                init_file.write_text('"""Module initialization."""\n')
                self.fixes_applied.append(f"Created {init_file}")
        
        return missing
    
    def check_temporary_files(self) -> List[str]:
        """Check for temporary or cache files to clean."""
        temp_files = []
        
        # Python cache
        for pycache in project_root.rglob('__pycache__'):
            temp_files.append(str(pycache))
        
        # .pyc files
        for pyc in project_root.rglob('*.pyc'):
            temp_files.append(str(pyc))
        
        # .pyo files
        for pyo in project_root.rglob('*.pyo'):
            temp_files.append(str(pyo))
        
        return temp_files
    
    def check_log_files(self) -> List[str]:
        """Check for log files that should be cleaned or gitignored."""
        log_files = []
        
        for log_file in project_root.glob('*.log'):
            log_files.append(str(log_file))
        
        log_dir = project_root / 'logs'
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                log_files.append(str(log_file))
        
        return log_files
    
    def check_gitignore(self) -> bool:
        """Check if .gitignore is comprehensive."""
        gitignore_path = project_root / '.gitignore'
        
        if not gitignore_path.exists():
            # Create comprehensive .gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
logs/
data/pipeline/
data/models/
reports/
.env
.env.local

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Documentation
*.pdf
*.docx
"""
            gitignore_path.write_text(gitignore_content)
            self.fixes_applied.append("Created .gitignore")
            return True
        
        return True
    
    def cleanup_files(self, files_to_remove: List[str], dry_run: bool = False):
        """Clean up files."""
        for file_path in files_to_remove:
            path = Path(file_path)
            if path.exists():
                if dry_run:
                    logger.info(f"Would remove: {file_path}")
                else:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    self.cleanup_items.append(f"Removed {file_path}")
                    logger.info(f"Removed: {file_path}")
    
    def check_code_quality(self) -> List[str]:
        """Basic code quality checks."""
        issues = []
        
        # Check for TODO/FIXME comments that might indicate incomplete work
        for py_file in project_root.rglob('*.py'):
            try:
                content = py_file.read_text()
                if 'TODO:' in content or 'FIXME:' in content:
                    # Count them
                    todo_count = content.count('TODO:')
                    fixme_count = content.count('FIXME:')
                    if todo_count + fixme_count > 0:
                        issues.append(f"{py_file}: {todo_count} TODO, {fixme_count} FIXME")
            except Exception as e:
                pass  # Skip binary or problematic files
        
        return issues
    
    def generate_cleanup_report(self) -> Dict:
        """Generate cleanup report."""
        return {
            'issues_found': len(self.issues_found),
            'fixes_applied': len(self.fixes_applied),
            'cleanup_items': len(self.cleanup_items),
            'details': {
                'issues': self.issues_found,
                'fixes': self.fixes_applied,
                'cleanup': self.cleanup_items
            }
        }
    
    def run_full_audit(self, cleanup: bool = True):
        """Run full audit and cleanup."""
        logger.info("="*80)
        logger.info("FINAL PROJECT AUDIT AND CLEANUP")
        logger.info("="*80)
        
        # Check missing init files
        logger.info("\n[1] Checking for missing __init__.py files...")
        missing_init = self.check_missing_init_files()
        if missing_init:
            logger.info(f"Found {len(missing_init)} missing init files - fixed")
        
        # Check .gitignore
        logger.info("\n[2] Checking .gitignore...")
        self.check_gitignore()
        logger.info("✓ .gitignore checked")
        
        # Check temporary files
        logger.info("\n[3] Checking for temporary files...")
        temp_files = self.check_temporary_files()
        if temp_files:
            logger.info(f"Found {len(temp_files)} temporary files")
            if cleanup:
                self.cleanup_files(temp_files, dry_run=False)
        
        # Check log files (keep but ensure gitignored)
        logger.info("\n[4] Checking log files...")
        log_files = self.check_log_files()
        if log_files:
            logger.info(f"Found {len(log_files)} log files (should be gitignored)")
        
        # Check code quality
        logger.info("\n[5] Checking code quality...")
        quality_issues = self.check_code_quality()
        if quality_issues:
            logger.info(f"Found {len(quality_issues)} code quality notes")
            for issue in quality_issues[:5]:  # Show first 5
                logger.info(f"  {issue}")
        
        # Check duplicates
        logger.info("\n[6] Checking for duplicate files...")
        duplicates = self.check_duplicate_files()
        if duplicates:
            logger.info(f"Found {len(duplicates)} potential duplicate files")
            for dup in duplicates[:5]:
                logger.info(f"  {dup}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("AUDIT SUMMARY")
        logger.info("="*80)
        logger.info(f"Issues Found: {len(self.issues_found)}")
        logger.info(f"Fixes Applied: {len(self.fixes_applied)}")
        logger.info(f"Cleanup Items: {len(self.cleanup_items)}")
        
        report = self.generate_cleanup_report()
        
        # Save report
        report_path = project_root / 'audit_report_final.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved to: {report_path}")
        logger.info("\n✅ AUDIT COMPLETE")


if __name__ == "__main__":
    auditor = ProjectAuditor()
    auditor.run_full_audit(cleanup=True)
