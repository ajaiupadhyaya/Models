"""
Report generation utility for financial analysis reports.
"""

import os
from datetime import datetime
from jinja2 import Template
from typing import Dict, Optional
import pandas as pd


class ReportGenerator:
    """
    Generate financial analysis reports from templates.
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize report generator.
        
        Args:
            template_path: Path to markdown template
        """
        if template_path is None:
            template_path = os.path.join(
                os.path.dirname(__file__),
                'template_markdown.md'
            )
        
        with open(template_path, 'r') as f:
            self.template = Template(f.read())
    
    def generate_report(self,
                       data: Dict,
                       output_path: str,
                       format: str = 'markdown') -> str:
        """
        Generate report from template and data.
        
        Args:
            data: Dictionary with report data
            output_path: Output file path
            format: Output format ('markdown', 'html')
        
        Returns:
            Generated report content
        """
        # Add default values
        if 'report_date' not in data:
            data['report_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Render template
        content = self.template.render(**data)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(content)
        
        # Convert to HTML if requested
        if format == 'html':
            html_path = output_path.replace('.md', '.html')
            self._convert_to_html(content, html_path)
            return html_path
        
        return output_path
    
    def _convert_to_html(self, markdown_content: str, output_path: str):
        """
        Convert markdown to HTML.
        
        Args:
            markdown_content: Markdown content
            output_path: Output HTML path
        """
        try:
            import markdown
            html = markdown.markdown(markdown_content, extensions=['tables', 'fenced_code'])
            
            # Wrap in HTML template
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Financial Analysis Report</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        line-height: 1.6;
                        color: #333;
                    }}
                    h1, h2, h3 {{
                        color: #2c3e50;
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid #ddd;
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #3498db;
                        color: white;
                    }}
                    code {{
                        background-color: #f4f4f4;
                        padding: 2px 6px;
                        border-radius: 3px;
                    }}
                </style>
            </head>
            <body>
                {html}
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(html_template)
        except ImportError:
            print("Markdown library not available. Install with: pip install markdown")


def create_sample_report(output_path: str = 'sample_report.md'):
    """
    Create a sample financial analysis report.
    
    Args:
        output_path: Output file path
    """
    generator = ReportGenerator()
    
    sample_data = {
        'analyst_name': 'Financial Analyst',
        'project_name': 'Sample Analysis',
        'executive_summary': 'This is a sample financial analysis report demonstrating the template structure.',
        'market_conditions': 'Current market conditions are stable with moderate volatility.',
        'economic_indicators': 'Key indicators show positive trends in GDP and employment.',
        'methodology': 'Analysis conducted using DCF valuation and comparable company analysis.',
        'assumptions': 'Key assumptions include WACC of 10% and terminal growth rate of 3%.',
        'data_sources': 'Data sourced from FRED, Alpha Vantage, and company filings.',
        'primary_findings': 'The analysis indicates fair value in the range of $X to $Y.',
        'supporting_analysis': 'Supporting analysis includes sensitivity analysis and scenario modeling.',
        'valuation_summary': 'Base case valuation: $Z per share.',
        'sensitivity_analysis': 'Sensitivity analysis shows valuation is most sensitive to WACC assumptions.',
        'risk_factors': 'Key risks include market volatility and regulatory changes.',
        'risk_metrics': 'VaR (95%): -X%, CVaR (95%): -Y%',
        'recommendations': 'Recommendation: [Buy/Hold/Sell] based on current valuation and risk profile.',
        'data_tables': 'See appendices for detailed data tables.',
        'charts': 'Charts and visualizations are included in the appendices.',
        'references': '1. Company 10-K filing\n2. Industry research reports'
    }
    
    return generator.generate_report(sample_data, output_path)
