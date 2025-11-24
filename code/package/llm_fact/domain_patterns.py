"""
Domain classification patterns for categorizing web URLs.

This module contains lists of patterns used to classify domains into
different categories: fact_checking, news, government, educational,
research, wikipedia, and other.

Patterns are designed to minimize false positives while maintaining
reasonable recall for academic research purposes.

Author: Matthew DeVerna
"""

# Fact-checking specific organizations and platforms
FACT_CHECK_PATTERNS = [
    "snopes",
    "politifact",
    "factcheck.org",
    "truthorfiction",
    "factchecker",
    "fullfact",
    "checkyourfact",
    "factscan",
]

# News organizations - mix of specific outlets and clear news indicators
NEWS_PATTERNS = [
    "news.",  # news.domain.com format
    "cnn.com",
    "pbs.org",
    "reuters.com",
    "wusf.org",  # Local NPR affiliate
    "denver7.com",  # Local news outlet
    # These others can be add but, as of now, do not appear in the data
    # "npr.org",
    # "bbc.co",
    # "associatedpress",
    # "bloomberg.com",
    # "wsj.com",
    # "nytimes.com",
    # "washingtonpost.com",
    # "usatoday.com",
    # "abcnews",
    # "cbsnews",
    # "nbcnews",
    # "foxnews",
    # "cspan.org",
]

# Government - official domains and well-established government patterns
GOVERNMENT_PATTERNS = [
    ".gov",
    ".mil",
    ".state.fl.us",  # More specific state pattern
    ".state.ca.us",
    ".state.tx.us",
    ".state.ny.us",
    "govdelivery.com",  # Official government communication platform
    "uscourts.gov",
    "supremecourt.gov",
    "whitehouse.gov",
    "senate.gov",
    "house.gov",
    "congress.gov",
    "gao.gov",
    "cdc.gov",
    "nih.gov",
    "fed.us",
    ".gov.uk",  # International government domains
    ".gov.ca",
    ".gov.au",
    "floridajobs.org",  # Florida government jobs portal
    "stlouisfed.org",  # St. Louis Federal Reserve
]

# Educational institutions - clear institutional indicators
EDUCATIONAL_PATTERNS = [
    ".edu",
    ".ac.uk",  # UK academic institutions
    ".ac.au",  # Australian academic institutions
    ".edu.au",
    "university",
    "college",
    "schooldistrict",
    ".k12.",
    "isd.org",  # Independent school district
    "usd.org",  # Unified school district
]

# Research institutions, think tanks, and policy organizations
# Using more specific organizational names and established institutions
RESEARCH_PATTERNS = [
    "brookings.edu",
    "heritage.org",
    "cato.org",
    "aei.org",
    "cfr.org",
    "rand.org",
    "hoover.org",
    "pewresearch.org",
    "gallup.com",
    "census.gov",
    "bls.gov",  # Bureau of Labor Statistics
    "bea.gov",  # Bureau of Economic Analysis
    "oecd.org",
    "unesco.org",
    "worldbank.org",
    "imf.org",
    "who.int",
    "un.org",
    "unicef.org",
    "nber.org",  # National Bureau of Economic Research
    "urban.org",  # Urban Institute
    "americanprogress.org",
    "mercatus.org",
    "manhattan-institute.org",
    "taxpolicycenter.org",
    "cbpp.org",  # Center on Budget and Policy Priorities
    "reason.org",  # Reason Foundation: Libertarian think tank
    "policymattersohio.org",  # Policy Matters Ohio: non-profit policy research institute
]

# Specific non-profit organizations identified in the data
# These are exact matches to avoid false positives
NONPROFIT_PATTERNS = [
    "healthinsurance.org",
    "americashealthrankings.org",
    "ontheissues.org",
    "deathpenaltyinfo.org",
    "prisonpolicy.org",
    "aflcio.org",
    "welfareinfo.org",
    "nafsa.org",
    "independentvoterproject.org",
    "ballotpedia.org",
    "opensecrets.org",
    "followthemoney.org",
]
