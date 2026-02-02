#!/usr/bin/env python3
"""
Aiperf Benchmark Analysis Script

Analyzes CSV output files from Aiperf benchmarks and generates performance reports.

Usage:
    python analyze_benchmark.py <csv_file> [--output <report.md>] [--compare <csv2>]

Example:
    python analyze_benchmark.py profile_export_aiperf.csv
    python analyze_benchmark.py run1.csv --compare run2.csv --output comparison.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas and numpy required. Install with:")
    print("  pip install pandas numpy --break-system-packages")
    sys.exit(1)


def load_benchmark_csv(filepath: str) -> pd.DataFrame:
    """Load an Aiperf benchmark CSV file."""
    df = pd.read_csv(filepath)
    return df


def filter_successful_requests(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only successful requests."""
    # Check for error column - may be named differently
    error_cols = [c for c in df.columns if 'error' in c.lower()]
    if error_cols:
        # Filter where error column is null/empty
        for col in error_cols:
            df = df[df[col].isna() | (df[col] == '')]
    return df


def calculate_percentiles(series: pd.Series, percentiles: list = [50, 90, 95, 99]) -> dict:
    """Calculate percentiles for a metric."""
    result = {}
    for p in percentiles:
        val = series.quantile(p / 100)
        result[f"p{p}"] = val
    result["mean"] = series.mean()
    result["min"] = series.min()
    result["max"] = series.max()
    result["std"] = series.std()
    return result


def analyze_latency_metrics(df: pd.DataFrame) -> dict:
    """Analyze latency-related metrics."""
    results = {}
    
    # Time to First Token (TTFT)
    ttft_cols = [c for c in df.columns if 'time_to_first_token' in c.lower() or 'ttft' in c.lower()]
    if ttft_cols:
        col = ttft_cols[0]
        if df[col].notna().any():
            results['ttft'] = calculate_percentiles(df[col].dropna())
    
    # Inter-Token Latency (ITL)
    itl_cols = [c for c in df.columns if 'inter_token_latency' in c.lower() or 'itl' in c.lower()]
    if itl_cols:
        col = itl_cols[0]
        if df[col].notna().any():
            results['itl'] = calculate_percentiles(df[col].dropna())
    
    # Request Latency (end-to-end)
    latency_cols = [c for c in df.columns if 'request_latency' in c.lower() and 'token' not in c.lower()]
    if latency_cols:
        col = latency_cols[0]
        if df[col].notna().any():
            results['request_latency'] = calculate_percentiles(df[col].dropna())
    
    return results


def analyze_throughput_metrics(df: pd.DataFrame) -> dict:
    """Analyze throughput-related metrics."""
    results = {}
    
    # Token throughput per request
    throughput_cols = [c for c in df.columns if 'throughput' in c.lower()]
    for col in throughput_cols:
        if df[col].notna().any():
            name = col.replace('_', ' ').title()
            results[col] = {
                'mean': df[col].mean(),
                'min': df[col].min(),
                'max': df[col].max(),
                'std': df[col].std()
            }
    
    # Token counts
    input_cols = [c for c in df.columns if 'input_token' in c.lower() and 'throughput' not in c.lower()]
    output_cols = [c for c in df.columns if 'output_token' in c.lower() and 'throughput' not in c.lower()]
    
    if input_cols:
        col = input_cols[0]
        results['input_tokens'] = {
            'total': df[col].sum(),
            'mean': df[col].mean(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    if output_cols:
        col = output_cols[0]
        results['output_tokens'] = {
            'total': df[col].sum(),
            'mean': df[col].mean(),
            'min': df[col].min(),
            'max': df[col].max()
        }
    
    return results


def analyze_request_stats(df: pd.DataFrame, df_all: pd.DataFrame) -> dict:
    """Analyze request statistics."""
    total = len(df_all)
    successful = len(df)
    failed = total - successful
    
    return {
        'total_requests': total,
        'successful_requests': successful,
        'failed_requests': failed,
        'success_rate': (successful / total * 100) if total > 0 else 0
    }


def generate_report(filepath: str, df: pd.DataFrame, df_success: pd.DataFrame) -> str:
    """Generate a markdown performance report."""
    lines = []
    lines.append(f"# Aiperf Benchmark Analysis Report")
    lines.append(f"\n**File:** `{filepath}`")
    lines.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Request stats
    stats = analyze_request_stats(df_success, df)
    lines.append("\n## Request Summary")
    lines.append(f"- Total Requests: {stats['total_requests']}")
    lines.append(f"- Successful: {stats['successful_requests']}")
    lines.append(f"- Failed: {stats['failed_requests']}")
    lines.append(f"- Success Rate: {stats['success_rate']:.1f}%")
    
    # Latency metrics
    latency = analyze_latency_metrics(df_success)
    if latency:
        lines.append("\n## Latency Metrics")
        
        if 'ttft' in latency:
            lines.append("\n### Time to First Token (TTFT)")
            ttft = latency['ttft']
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| p50 | {ttft['p50']*1000:.2f} ms |")
            lines.append(f"| p90 | {ttft['p90']*1000:.2f} ms |")
            lines.append(f"| p95 | {ttft['p95']*1000:.2f} ms |")
            lines.append(f"| p99 | {ttft['p99']*1000:.2f} ms |")
            lines.append(f"| Mean | {ttft['mean']*1000:.2f} ms |")
        
        if 'itl' in latency:
            lines.append("\n### Inter-Token Latency (ITL)")
            itl = latency['itl']
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| p50 | {itl['p50']*1000:.2f} ms |")
            lines.append(f"| p90 | {itl['p90']*1000:.2f} ms |")
            lines.append(f"| p95 | {itl['p95']*1000:.2f} ms |")
            lines.append(f"| p99 | {itl['p99']*1000:.2f} ms |")
            lines.append(f"| Mean | {itl['mean']*1000:.2f} ms |")
        
        if 'request_latency' in latency:
            lines.append("\n### End-to-End Request Latency")
            rl = latency['request_latency']
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| p50 | {rl['p50']:.3f} s |")
            lines.append(f"| p90 | {rl['p90']:.3f} s |")
            lines.append(f"| p95 | {rl['p95']:.3f} s |")
            lines.append(f"| p99 | {rl['p99']:.3f} s |")
            lines.append(f"| Mean | {rl['mean']:.3f} s |")
    
    # Throughput metrics
    throughput = analyze_throughput_metrics(df_success)
    if throughput:
        lines.append("\n## Throughput Metrics")
        
        if 'input_tokens' in throughput:
            lines.append("\n### Input Tokens")
            it = throughput['input_tokens']
            lines.append(f"- Total: {it['total']:,.0f}")
            lines.append(f"- Mean per request: {it['mean']:.1f}")
        
        if 'output_tokens' in throughput:
            lines.append("\n### Output Tokens")
            ot = throughput['output_tokens']
            lines.append(f"- Total: {ot['total']:,.0f}")
            lines.append(f"- Mean per request: {ot['mean']:.1f}")
        
        for key, val in throughput.items():
            if 'throughput' in key:
                name = key.replace('_', ' ').replace('per request', '').strip()
                lines.append(f"\n### {name.title()}")
                lines.append(f"- Mean: {val['mean']:.1f} tokens/s")
                lines.append(f"- Min: {val['min']:.1f} tokens/s")
                lines.append(f"- Max: {val['max']:.1f} tokens/s")
    
    return "\n".join(lines)


def compare_benchmarks(file1: str, file2: str) -> str:
    """Compare two benchmark runs."""
    df1 = load_benchmark_csv(file1)
    df2 = load_benchmark_csv(file2)
    df1_success = filter_successful_requests(df1)
    df2_success = filter_successful_requests(df2)
    
    lines = []
    lines.append("# Benchmark Comparison Report")
    lines.append(f"\n**Run 1:** `{file1}`")
    lines.append(f"**Run 2:** `{file2}`")
    
    # Compare latencies
    lat1 = analyze_latency_metrics(df1_success)
    lat2 = analyze_latency_metrics(df2_success)
    
    lines.append("\n## Latency Comparison")
    
    if 'ttft' in lat1 and 'ttft' in lat2:
        lines.append("\n### TTFT (ms)")
        lines.append("| Percentile | Run 1 | Run 2 | Diff |")
        lines.append("|------------|-------|-------|------|")
        for p in ['p50', 'p90', 'p95', 'p99']:
            v1 = lat1['ttft'][p] * 1000
            v2 = lat2['ttft'][p] * 1000
            diff = ((v2 - v1) / v1 * 100) if v1 > 0 else 0
            sign = "+" if diff > 0 else ""
            lines.append(f"| {p} | {v1:.2f} | {v2:.2f} | {sign}{diff:.1f}% |")
    
    if 'itl' in lat1 and 'itl' in lat2:
        lines.append("\n### ITL (ms)")
        lines.append("| Percentile | Run 1 | Run 2 | Diff |")
        lines.append("|------------|-------|-------|------|")
        for p in ['p50', 'p90', 'p95', 'p99']:
            v1 = lat1['itl'][p] * 1000
            v2 = lat2['itl'][p] * 1000
            diff = ((v2 - v1) / v1 * 100) if v1 > 0 else 0
            sign = "+" if diff > 0 else ""
            lines.append(f"| {p} | {v1:.2f} | {v2:.2f} | {sign}{diff:.1f}% |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Aiperf benchmark CSV output files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s profile_export_aiperf.csv
  %(prog)s results.csv --output report.md
  %(prog)s baseline.csv --compare optimized.csv
        """
    )
    parser.add_argument("csv_file", help="Primary benchmark CSV file to analyze")
    parser.add_argument("--output", "-o", help="Output report file (markdown)")
    parser.add_argument("--compare", "-c", help="Second CSV file for comparison")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of markdown")
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)
    
    if args.compare:
        if not Path(args.compare).exists():
            print(f"Error: Comparison file not found: {args.compare}")
            sys.exit(1)
        report = compare_benchmarks(args.csv_file, args.compare)
    else:
        df = load_benchmark_csv(args.csv_file)
        df_success = filter_successful_requests(df)
        
        if args.json:
            results = {
                'file': args.csv_file,
                'request_stats': analyze_request_stats(df_success, df),
                'latency': analyze_latency_metrics(df_success),
                'throughput': analyze_throughput_metrics(df_success)
            }
            report = json.dumps(results, indent=2, default=str)
        else:
            report = generate_report(args.csv_file, df, df_success)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()