"""corpus_analysis_report.py.

Last Updated: November 19, 2025
Last Updated: November 19, 2025
"""

import json
import time
from pathlib import Path

import pandas as pd
from html_to_markdown import convert
from wasabi import msg

from lexos.corpus import Corpus


def create_corpus_analysis_report(
    corpus: Corpus,
    output_dir: str = None,
    console_output=True,
    html=False,
) -> str:
    """Create a comprehensive report of the corpus analysis results.

    This function exports various statistics and summaries of the corpus
    analysis to CSV files and generates a text report.

    Args:
        corpus (Corpus): The corpus object containing documents and metadata.
        output_dir (str): The directory path where the report files will be saved.
        console_output (bool): Whether to print progress messages to the console.
        html (bool): Whether to generate an HTML report (not implemented yet).

    Returns:
        str: The generated report as a string in either HTML or Markdown format.
    """
    # Get stats object once to avoid multiple calls
    stats = corpus.get_stats(active_only=True)

    # Only create output directory and save files if output_dir is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 1. Export corpus data to CSV
        corpus_df = corpus.to_df()
        csv_path = output_dir / "corpus_overview.csv"
        corpus_df.to_csv(csv_path, index=False)
        if console_output:
            msg.good(f"✓ Exported corpus overview to {csv_path}")
            msg.info(f"   - {len(corpus_df)} documents")
            msg.info(f"   - {len(corpus_df.columns)} data columns")

        # 2. Export detailed statistics
        stats_df = stats.doc_stats_df
        stats_path = output_dir / "document_statistics.csv"
        stats_df.to_csv(stats_path)
        if console_output:
            msg.good(f"✓ Exported detailed statistics to {stats_path}")

        # 3. Export analysis results summary
        summary_data = {
            "corpus_name": corpus.name,
            "total_documents": corpus.num_docs,
            "active_documents": corpus.num_active_docs,
            "total_tokens": corpus.num_tokens,
            "mean_document_length": stats.mean,
            "std_document_length": stats.standard_deviation,
            "iqr_outliers_count": len(stats.iqr_outliers),
            "corpus_fingerprint": corpus._generate_corpus_fingerprint(),
        }

        # Add quality metrics
        quality = stats.corpus_quality_metrics
        summary_data.update(
            {
                "length_balance_classification": quality["document_length_balance"][
                    "classification"
                ],
                "sampling_adequacy": quality["vocabulary_richness"][
                    "sampling_adequacy"
                ],
                "size_adequacy": quality["corpus_size_metrics"]["size_adequacy"],
            }
        )

        # Add Zipf analysis
        zipf = stats.zipf_analysis
        summary_data.update(
            {
                "zipf_follows_law": zipf["follows_zipf"],
                "zipf_goodness_of_fit": zipf["zipf_goodness_of_fit"],
                "zipf_r_squared": zipf["r_squared"],
            }
        )

        summary_df = pd.DataFrame([summary_data])
        summary_path = output_dir / "corpus_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        if console_output:
            msg.good(f"✓ Exported corpus summary to {summary_path}")

        # 4. Export analysis results from modules
        if corpus.analysis_results:
            results_path = output_dir / "module_analysis_results.json"
            with open(results_path, "w") as f:
                json.dump(corpus.analysis_results, f, indent=2, default=str)
            if console_output:
                msg.good(f"✓ Exported module results to {results_path}")

    # Get quality and zipf metrics from stats object
    quality = stats.corpus_quality_metrics
    zipf = stats.zipf_analysis

    # Always generate HTML first
    html_report = "<html><head><title>Corpus Analysis Report</title></head><body>"
    html_report += "<h1>Corpus Analysis Report</h1>"
    html_report += f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
    html_report += "<hr>"
    html_report += f"<h2>Corpus Overview</h2><ul>"
    html_report += f"<li>Name: {corpus.name}</li>"
    html_report += f"<li>Total Documents: {corpus.num_docs}</li>"
    html_report += f"<li>Active Documents: {corpus.num_active_docs}</li>"
    html_report += f"<li>Total Tokens: {corpus.num_tokens}</li>"
    html_report += "</ul>"
    html_report += f"<h2>Statistical Summary</h2><ul>"
    html_report += f"<li>Mean Length: {stats.mean:.1f} tokens</li>"
    html_report += f"<li>Standard Deviation: {stats.standard_deviation:.1f} tokens</li>"
    html_report += (
        f"<li>Shortest Document: {stats.doc_stats_df['total_tokens'].min()} tokens</li>"
    )
    html_report += (
        f"<li>Longest Document: {stats.doc_stats_df['total_tokens'].max()} tokens</li>"
    )
    html_report += f"<li>IQR Outliers: {len(stats.iqr_outliers)}</li>"
    html_report += "</ul>"
    html_report += f"<h2>Quality Assessment</h2><ul>"
    html_report += f"<li>Length Balance: {quality['document_length_balance']['classification']}</li>"
    html_report += f"<li>Sampling Adequacy: {quality['vocabulary_richness']['sampling_adequacy']}</li>"
    html_report += (
        f"<li>Size Adequacy: {quality['corpus_size_metrics']['size_adequacy']}</li>"
    )
    html_report += f"<li>Follows Zipf's Law: {zipf['follows_zipf']}</li>"
    html_report += "</ul>"
    if corpus.analysis_results:
        html_report += "<h2>Module Analyses</h2><ul>"
        for module_name, data in corpus.analysis_results.items():
            html_report += f"<li>{module_name}: Version {data['version']} ({data['timestamp']})</li>"
        html_report += "</ul>"
    html_report += "</body></html>"

    # Determine output format
    if html:
        report = html_report
        ext = ".html"
    else:
        # Convert HTML to Markdown
        report = convert(html_report)
        ext = ".md"

    # Save report if output_dir is provided
    if output_dir is not None:
        report_path = output_dir / f"analysis_report{ext}"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        if console_output:
            msg.good(f"✓ Created comprehensive report at {report_path}")

            msg.good(f"\n📂 All files saved to: {output_dir.absolute()}")
            msg.good(f"\n📋 Files created:")
            for file_path in output_dir.glob("*"):
                file_size = file_path.stat().st_size
                msg.info(f"   📄 {file_path.name}: {file_size:,} bytes")

            msg.good(f"\n💡 Sharing Tips:")
            msg.info(f"   📊 Use CSV files for data analysis in Excel/R/Python")
            msg.info(f"   📋 Share the report file for quick overview")
            msg.info(f"   🔗 Use JSON files for integration with other tools")
            msg.info(
                f"   💾 The corpus directory contains all original documents and metadata"
            )

    return report
