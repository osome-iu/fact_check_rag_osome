# data_analysis/

Scripts for analyzing fact-checking results, calculating performance metrics, and generating summary tables.

## Contents

- `006_not_enough_info_analyses.py`: Analyze patterns in "Not Enough Information" responses
- `007_nei_performance_by_veracity.py`: Calculate NEI performance metrics broken down by veracity level
- `008_topk_accuracy_by_veracity.py`: Calculate top-k accuracy metrics by veracity level
- `analyze_web_url_citations.py`: Analyze citation patterns in web-grounded model responses
- `calculate_faithfulness_agreement.py`: Calculate inter-annotator agreement for faithfulness labels
- `calculate_topk_accuracy.py`: Calculate information retrieval top-k accuracy metrics
- `enrich_web_url_data.py`: Enrich extracted web URLs with domain quality and political leaning data
- `generate_binary_performance_tables.py`: Generate binary classification performance tables
- `generate_citation_statistics_table.py`: Generate summary statistics for citation usage
- `generate_classification_reports.py`: Generate comprehensive classification reports for all models
- `generate_performance_tables.py`: Generate multi-class performance tables
- `generate_relative_increase_table.py`: Generate tables comparing relative performance improvements
- `test_information_retrieval.py`: Test RAG information retrieval accuracy
