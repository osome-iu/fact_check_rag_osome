# PolitiFact Scraper Integration

## Overview

This directory contains the PolitiFact scraper code from https://github.com/mr-devs/politifact-scraper, integrated into the fact-checking RAG evaluation project.

## Integration Date

November 19, 2025

## Files Included

### Package Code (`politifact_pkg/`)

- `__init__.py` - Package initialization, exports PolitiFactCheck class
- `data_models.py` - PolitiFactCheck data model for processing fact check webpages
- `parsing.py` - HTML parsing functions for extracting fact check data
- `utils.py` - Utility functions for fetching URLs and finding max pages

### Scripts (`scripts/`)

- `scrape_politifact_all.py` - Main script to scrape all PolitiFact fact checks
- `scrape_politifact_missing_links.py` - Script to scrape any missing fact check links

### Documentation

- `README.md` - Original repository README (overview)
- `PACKAGE_README.md` - Package-specific installation instructions
- `setup.py` - Package setup configuration for local installation
- `INTEGRATION.md` - This file (integration documentation)
