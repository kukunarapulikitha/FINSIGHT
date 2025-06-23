import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv
import yfinance as yf
import requests
from io import BytesIO
import PyPDF2
import time
from functools import lru_cache
import re
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Add this import after your other imports
from llm_evaluation import show_llm_evaluation_page, LLM_PROFILES

import numpy as np
# Load environment variables
load_dotenv()

def format_currency(amount: float, decimals: int = 2) -> str:
    """Format currency values properly with proper spacing"""
    if amount is None or (isinstance(amount, float) and np.isnan(amount)):
        return "N/A"
    
    # Ensure amount is a number
    try:
        amount = float(amount)
    except:
        return "N/A"
    
    if amount >= 1e9:
        return f"${amount/1e9:.{decimals}f}B"
    elif amount >= 1e6:
        return f"${amount/1e6:.{decimals}f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.{decimals}f}K"
    else:
        return f"${amount:.{decimals}f}"


def clean_response_text(text: str) -> str:
    """Clean and format response text for proper display"""
    if not text:
        return text
    
    import re
    
    # Fix currency amount patterns: $107.0million -> $107.0 million
    text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)(million|billion|thousand|M|B|K)', r'$\1 \2', text, flags=re.IGNORECASE)
    
    # First, fix the specific pattern causing issues: $numberText should be $number Text
    # This regex looks for currency followed by numbers then letters without space
    text = re.sub(r'\$(\d+(?:,\d{3})*(?:\.\d+)?[BMK]?)([A-Za-z])', r'$\1 \2', text)
    
    # Fix merged bold markers: **word** should have spaces
    # First, ensure spaces around bold markers
    text = re.sub(r'(\S)\*\*(\S)', r'\1 **\2', text)
    text = re.sub(r'(\S)\*\*', r'\1 **', text)
    text = re.sub(r'\*\*(\S)', r'** \1', text)
    
    # Fix percentage patterns: number% word should be number% word
    text = re.sub(r'(\d+(?:\.\d+)?%)([A-Za-z])', r'\1 \2', text)
    
    # Fix patterns like: of0.0 -> of 0.0
    text = re.sub(r'([a-z])(\d+(?:\.\d+)?)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Fix patterns like: marginof10.0 -> margin of 10.0
    text = re.sub(r'([a-z])(of)(\d+)', r'\1 \2 \3', text, flags=re.IGNORECASE)
    
    # Fix patterns like: 4,242Mwith -> 4,242M with
    text = re.sub(r'(\d+(?:,\d{3})*(?:\.\d+)?[BMK])([a-z])', r'\1 \2', text)
    
    # Fix merged words: millionand -> million and
    text = re.sub(r'(million|billion|thousand)(and|with|or|for)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Remove duplicate asterisks (e.g., ***** -> **)
    text = re.sub(r'\*{3,}', '**', text)
    
    # Clean up any remaining stray asterisks (single asterisks not part of bold)
    text = re.sub(r'(?<!\*)\*(?!\*)', ' ', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    
    # Fix line breaks
    text = text.replace('\\n', '\n')
    
    # Normalize whitespace
    text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
    
    return text.strip()


def clean_list_item(text: str) -> str:
    """Clean individual list items, removing leading asterisks and formatting"""
    if not text:
        return text
    
    import re
    
    # Remove leading asterisks, numbers, periods, and whitespace
    text = re.sub(r'^[\*\s\d\.]+', '', text.strip())
    
    # Also clean using the main clean function
    text = clean_response_text(text)
    
    return text.strip()
def format_response_for_display(response_parts: List[str], response_mode: str, query_intent: str, 
                               company: str, ticker: str, insights: Dict, kpis: Dict, 
                               risk_assessment: Dict, market_data: Dict, parsed_data: Dict) -> str:
    """Format the complete response with proper styling"""
    
    formatted_parts = []
    
    if response_mode == "concise":
        # CONCISE MODE - Clean, structured format
        
        # Main answer with proper formatting
        if insights.get("query_specific_answer"):
            answer = clean_response_text(insights["query_specific_answer"])
            formatted_parts.append(f"### 📌 {answer}\n")
        
        # For earnings/financial analysis queries, show key metrics
        elif query_intent in ["document_analysis", "general_analysis"] and "earnings" in query.lower():
            # Format the earnings summary properly
            revenue_val = parsed_data.get('revenue', 0)
            revenue_str = format_currency(revenue_val)
            net_margin_val = kpis.get('net_margin', 0)
            pe_ratio_val = kpis.get('price_to_earnings', 0)
            
            answer = f"{company}'s earnings in {period} show a revenue of {revenue_str} with a net margin of {net_margin_val:.1f}% and P/E ratio of {pe_ratio_val:.1f}."
            formatted_parts.append(f"### 📌 {answer}\n")
        
        # Investment decision
        if insights.get("investment_decision") and query_intent == "investment_decision":
            decision = insights["investment_decision"]
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "WAIT": "⏸️"}.get(decision, "")
            formatted_parts.append(f"Decision: {emoji} {decision}\n")
        
        # Current price
        if market_data.get('price') and query_intent == "investment_decision":
            formatted_parts.append(f"Current Price: {format_currency(market_data['price'])}")
        
        # Risk level
        if query_intent in ["risk_assessment", "investment_decision"]:
            risk_score = risk_assessment.get('overall_risk_score', 0)
            risk_level = risk_assessment.get('risk_level', 'Unknown')
            formatted_parts.append(f"**Risk Level:** {risk_level} ({risk_score:.0f}/100)\n")
        
        # Key metrics in a clean table format
        if query_intent in ["investment_decision", "general_analysis", "document_analysis"]:
            formatted_parts.append("#### Key Metrics")
            formatted_parts.append("| Metric | Value |")
            formatted_parts.append("|--------|-------|")
            
            # Safely format each metric
            net_margin = kpis.get('net_margin', 0)
            pe_ratio = kpis.get('price_to_earnings', 0)
            
            formatted_parts.append(f"| Net Margin | {net_margin:.1f}% |")
            formatted_parts.append(f"| P/E Ratio | {pe_ratio:.1f} |")
            
            if market_data.get('52_week_high') and market_data.get('52_week_low'):
                range_str = f"{format_currency(market_data['52_week_low'])} - {format_currency(market_data['52_week_high'])}"
                formatted_parts.append(f"| 52-Week Range | {range_str} |")
            formatted_parts.append("")
        
        # Concerns with bullet points
        if insights.get("key_concerns") and len(insights["key_concerns"]) > 0:
            formatted_parts.append("#### ⚠️ Concerns")
            for concern in insights["key_concerns"][:2]:
                formatted_parts.append(f"- {concern}")
            formatted_parts.append("")
        
        # Recommendations
        if insights.get("recommendations"):
            formatted_parts.append("#### 💡 What to do")
            for rec in insights["recommendations"][:2]:
                formatted_parts.append(f"- {rec}")
            formatted_parts.append("")
        
        # Volatility warning
        volatile_stocks = ['TSLA', 'GME', 'AMC', 'COIN', 'RIOT', 'MARA', 'PLTR', 'NIO', 'RIVN']
        if ticker in volatile_stocks and query_intent == "investment_decision":
            formatted_parts.append("*⚠️ Note: This stock is known for high volatility*\n")
    
    else:
        # DETAILED MODE - Professional format with sections
        
        # Header
        formatted_parts.append(f"# {company} ({ticker}) Financial Analysis\n")
        
        # Investment Overview Box
        if query_intent == "investment_decision":
            formatted_parts.append("## 📊 Investment Overview")
            formatted_parts.append("```")
            if market_data.get('price'):
                formatted_parts.append(f"Current Price:  {format_currency(market_data['price'])}")
            if market_data.get('market_cap'):
                formatted_parts.append(f"Market Cap:     {format_currency(market_data['market_cap'])}")
            if market_data.get('52_week_high') and market_data.get('52_week_low'):
                current = market_data.get('price', 0)
                high = market_data['52_week_high']
                low = market_data['52_week_low']
                pct_from_high = ((current - high) / high * 100) if high > 0 else 0
                formatted_parts.append(f"52-Week Range:  {format_currency(low)} - {format_currency(high)}")
                formatted_parts.append(f"From High:      {pct_from_high:+.1f}%")
            formatted_parts.append("```\n")
        

        # Analysis section
        if insights.get("query_specific_answer"):
            formatted_parts.append("## 📈 Analysis")
            # Clean the answer text before adding it
            answer = insights["query_specific_answer"]
            
            # Apply the same cleaning as in concise mode
            answer = clean_response_text(answer)
            
            # Additional cleaning for common patterns in financial text
            import re
            
            # Fix currency patterns: $107.0million -> $107.0 million
            answer = re.sub(r'\$(\d+(?:\.\d+)?)(million|billion|thousand)', r'$\1 \2', answer, flags=re.IGNORECASE)
            
            # Fix percentage patterns: of10.0% -> of 10.0%
            answer = re.sub(r'of(\d+(?:\.\d+)?%)', r'of \1', answer)
            
            # Fix merged words with numbers: margin10.0 -> margin 10.0
            answer = re.sub(r'([a-z])(\d+(?:\.\d+)?)', r'\1 \2', answer, flags=re.IGNORECASE)
            
            # Fix merged percentage with words: 10.0%and -> 10.0% and
            answer = re.sub(r'(\d+(?:\.\d+)?%)([a-z])', r'\1 \2', answer, flags=re.IGNORECASE)
            
            # Remove any stray asterisks
            answer = re.sub(r'(?<!\*)\*(?!\*)', ' ', answer)
            
            formatted_parts.append(f"{answer}\n")

        
        # Investment Decision
        if insights.get("investment_decision") and query_intent == "investment_decision":
            decision = insights["investment_decision"]
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "WAIT": "⏸️"}.get(decision, "")
            formatted_parts.append(f"## {emoji} Investment Decision: {decision}")
            
            # Add reasoning
            if decision == "BUY":
                formatted_parts.append("*Strong fundamentals support accumulation at current levels.*")
            elif decision == "HOLD":
                formatted_parts.append("*Mixed signals suggest maintaining current position while monitoring developments.*")
            elif decision == "SELL":
                formatted_parts.append("*Risk factors outweigh potential rewards at current valuation.*")
            else:
                formatted_parts.append("*Unclear trends warrant patience before taking a position.*")
            formatted_parts.append("")
        
        # Financial Metrics Table
        if query_intent in ["investment_decision", "general_analysis"]:
            formatted_parts.append("## 📊 Financial Metrics")
            formatted_parts.append("| Metric | Value |")
            formatted_parts.append("|--------|-------|")
            formatted_parts.append(f"| Revenue | {format_currency(parsed_data.get('revenue', 0))} |")
            formatted_parts.append(f"| Gross Margin | {kpis.get('gross_margin', 0):.1f}% |")
            formatted_parts.append(f"| Net Margin | {kpis.get('net_margin', 0):.1f}% |")
            formatted_parts.append(f"| P/E Ratio | {kpis.get('price_to_earnings', 0):.1f} |")
            formatted_parts.append(f"| ROE | {kpis.get('return_on_equity', 0):.1f}% |")
            formatted_parts.append(f"| Debt/Equity | {kpis.get('debt_to_equity', 0):.2f} |")
            formatted_parts.append("")
        
        # Strengths and Concerns in columns
        if insights.get("key_strengths") or insights.get("key_concerns"):
            formatted_parts.append("## 💪 Strengths vs ⚠️ Concerns")
            formatted_parts.append("")
            formatted_parts.append("### Strengths")
            if insights.get("key_strengths"):
                for strength in insights["key_strengths"]:
                    # Clean the strength text
                    clean_strength = clean_list_item(strength)
                    formatted_parts.append(f"- ✅ {clean_strength}")
            formatted_parts.append("")
            
            formatted_parts.append("### Concerns")
            if insights.get("key_concerns"):
                for concern in insights["key_concerns"]:
                    # Clean the concern text
                    clean_concern = clean_list_item(concern)
                    formatted_parts.append(f"- ⚠️ {clean_concern}")
            formatted_parts.append("")

        # Investment Strategy
        if insights.get("recommendations"):
            formatted_parts.append("## 🎯 Investment Strategy")
            for i, rec in enumerate(insights["recommendations"], 1):
                # Clean the recommendation text by removing any existing numbering and asterisks
                clean_rec = rec.strip()
                
                # Remove patterns like "** 1. **", "**1.**", "1.", "1)", etc. from the beginning
                import re
                clean_rec = re.sub(r'^[\*\s]*\d+[\.\)]\s*[\*\s]*', '', clean_rec)
                
                # Remove any remaining leading/trailing asterisks
                clean_rec = re.sub(r'^[\*\s]+|[\*\s]+$', '', clean_rec)
                
                # Remove internal double asterisks that should be formatting
                clean_rec = re.sub(r'\*\*\s*', '', clean_rec)
                clean_rec = re.sub(r'\s*\*\*', '', clean_rec)
                
                # Format as a clean numbered list
                formatted_parts.append(f"{i}. {clean_rec}")
            formatted_parts.append("")
        
        # Outlook sections
        if insights.get("market_position"):
            formatted_parts.append("## 🏆 Market Position")
            formatted_parts.append(f"{insights['market_position']}\n")
        
        if insights.get("future_outlook"):
            formatted_parts.append("## 🔮 Future Outlook")
            formatted_parts.append(f"{insights['future_outlook']}\n")
    
    return "\n".join(formatted_parts)
# Add this after your imports
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and pandas Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

# SEC EDGAR API Configuration
SEC_HEADERS = {
    'User-Agent': 'Your Company Name your.email@example.com'
}
SEC_BASE_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"

# Company discovery function using Yahoo Finance search
def discover_company_info(company_query: str):
    """
    Discover company ticker and CIK using various methods
    Returns: dict with ticker, full_name, cik, and confidence score
    """
    # Common company name to ticker mappings
    COMPANY_TICKER_MAP = {
        # Banks and Financial Services
        "state street": "STT",
        "state street corporation": "STT",
        "state street corp": "STT",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "jpmorgan chase": "JPM",
        "bank of america": "BAC",
        "wells fargo": "WFC",
        "goldman sachs": "GS",
        "morgan stanley": "MS",
        "charles schwab": "SCHW",
        
        # Tech Companies
        "snowflake": "SNOW",
        "palantir": "PLTR",
        "servicenow": "NOW",
        "service now": "NOW",
        "salesforce": "CRM",
        "spotify": "SPOT",
        "uber": "UBER",
        "lyft": "LYFT",
        "airbnb": "ABNB",
        "coinbase": "COIN",
        "robinhood": "HOOD",
        
        # Big Tech
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        
        # Other Major Companies
        "walmart": "WMT",
        "disney": "DIS",
        "coca cola": "KO",
        "coca-cola": "KO",
        "pepsi": "PEP",
        "pepsico": "PEP",
        "mcdonald's": "MCD",
        "mcdonalds": "MCD",
        "starbucks": "SBUX",
        "nike": "NKE",
        "intel": "INTC",
        "amd": "AMD",
        "advanced micro devices": "AMD"
    }
    
    try:
        # Clean the query
        clean_query = company_query.lower().strip()
        
        # Method 1: Check our mapping first
        for company_name, ticker in COMPANY_TICKER_MAP.items():
            if company_name in clean_query or clean_query in company_name:
                # Verify with yfinance
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    if info and 'longName' in info:
                        return {
                            "ticker": ticker,
                            "full_name": info.get('longName', company_query),
                            "cik": None,
                            "confidence": 0.95,
                            "method": "known_mapping"
                        }
                except:
                    pass
        
        # Method 2: Try direct ticker lookup (if query is short and uppercase)
        if len(company_query) <= 5 and company_query.isupper():
            try:
                stock = yf.Ticker(company_query)
                info = stock.info
                if info and 'longName' in info:
                    return {
                        "ticker": company_query,
                        "full_name": info.get('longName', company_query),
                        "cik": None,
                        "confidence": 0.9,
                        "method": "direct_ticker"
                    }
            except:
                pass
        
        # Method 3: Try the query as a ticker (uppercase it)
        ticker_test = company_query.upper().replace(" ", "")
        if len(ticker_test) <= 5:
            try:
                stock = yf.Ticker(ticker_test)
                info = stock.info
                if info and 'longName' in info:
                    return {
                        "ticker": ticker_test,
                        "full_name": info.get('longName', company_query),
                        "cik": None,
                        "confidence": 0.8,
                        "method": "uppercase_ticker"
                    }
            except:
                pass
        
        # Method 4: Remove common suffixes and try again
        for suffix in [' inc', ' corp', ' corporation', ' limited', ' ltd', ' plc', ' sa', ' ag', ' nv', ' se', ' co', '.com']:
            if clean_query.endswith(suffix):
                base_name = clean_query[:-len(suffix)].strip()
                # Check mapping again with base name
                for company_name, ticker in COMPANY_TICKER_MAP.items():
                    if base_name in company_name or company_name in base_name:
                        try:
                            stock = yf.Ticker(ticker)
                            info = stock.info
                            if info and 'longName' in info:
                                return {
                                    "ticker": ticker,
                                    "full_name": info.get('longName', company_query),
                                    "cik": None,
                                    "confidence": 0.85,
                                    "method": "suffix_removed_mapping"
                                }
                        except:
                            pass
        
        # Method 5: Use SEC Edgar search API for CIK lookup
        sec_search_url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={company_query}&output=json"
        try:
            response = requests.get(sec_search_url, headers=SEC_HEADERS, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'companyMatch' in data and len(data['companyMatch']) > 0:
                    match = data['companyMatch'][0]
                    cik = str(match.get('cik', '')).zfill(10)
                    name = match.get('name', company_query)
                    
                    # Try to find ticker from SEC data or use a mapping
                    ticker = None
                    # You could enhance this by maintaining a CIK-to-ticker mapping
                    
                    return {
                        "ticker": ticker or company_query.upper()[:10],  # Longer limit for unknown tickers
                        "full_name": name,
                        "cik": cik,
                        "confidence": 0.6,
                        "method": "sec_search"
                    }
        except:
            pass
        
        # Method 6: Fallback - but don't truncate to 5 chars
        # Instead, return the query as-is for the LLM to handle
        return {
            "ticker": company_query.upper().replace(" ", "")[:10],  # Allow up to 10 chars
            "full_name": company_query,
            "cik": None,
            "confidence": 0.2,  # Very low confidence
            "method": "fallback",
            "needs_llm_help": True  # Flag for LLM to help
        }
        
    except Exception as e:
        return {
            "ticker": "UNKNOWN",
            "full_name": company_query,
            "cik": None,
            "confidence": 0.1,
            "method": "error",
            "error": str(e)
        }
# Function to search for company CIK if not found
def search_company_cik(company_name: str, ticker: str = None):
    """Search for company CIK using SEC Edgar database"""
    try:
        # Try with company name
        search_queries = [company_name]
        if ticker:
            search_queries.append(ticker)
        
        for query in search_queries:
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?company={query}&output=json"
            response = requests.get(url, headers=SEC_HEADERS, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'companyMatch' in data and len(data['companyMatch']) > 0:
                    # Return the first match
                    match = data['companyMatch'][0]
                    return str(match.get('cik', '')).zfill(10)
        
        return None
        
    except Exception:
        return None



def extract_query_entities(query: str):
    """Extract company name and time period from natural language query"""
    query_lower = query.lower()
    
    # First, try to identify company from the query
    company_name = None
    
    # Common patterns for company mentions
    # Pattern 1: "Company Name" (in quotes)
    quote_pattern = r'"([^"]+)"'
    quote_match = re.search(quote_pattern, query)
    if quote_match:
        company_name = quote_match.group(1)
    
    # Pattern 2: Known company keywords followed by company name
    company_keywords = ['invest in', 'analyze', 'buy', 'sell', 'about', 'is', 'how is', "what's", 'should i invest in', 'earnings of', 'financials of']
    for keyword in company_keywords:
        pattern = rf'{keyword}\s+([A-Z][A-Za-z0-9\s&\.\,\-]+?)(?:\s+(?:stock|shares|company|corp|inc|ltd|limited|good|bad|risky|safe|for|in|a\s|an\s|the\s|\?|$))'
        match = re.search(pattern, query, re.IGNORECASE)
        if match and not company_name:
            potential_company = match.group(1).strip()
            # Clean up the extracted name
            for stop_word in ['for', 'in', 'a', 'an', 'the', 'good', 'bad', 'risky', 'safe']:
                potential_company = re.sub(rf'\s+{stop_word}\s*$', '', potential_company, flags=re.IGNORECASE)
            company_name = potential_company
            break
    
    # Pattern 3: Ticker symbols (usually uppercase, 1-5 letters)
    if not company_name:
        ticker_pattern = r'\b([A-Z]{1,5})\b'
        ticker_matches = re.findall(ticker_pattern, query)
        for potential_ticker in ticker_matches:
            # Skip common words that might be mistaken for tickers
            if potential_ticker not in ['I', 'A', 'Q', 'FY', 'CEO', 'CFO', 'IPO', 'EPS', 'PE']:
                company_name = potential_ticker
                break
    
    # Pattern 4: Any capitalized phrase that might be a company
    if not company_name:
        # Look for capitalized words/phrases
        cap_pattern = r'(?:^|(?<=\s))([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        cap_matches = re.findall(cap_pattern, query)
        for match in cap_matches:
            # Filter out common non-company words
            if match.lower() not in ['should', 'what', 'how', 'when', 'where', 'why', 'quarter', 'fiscal', 'year', 'analyze', 'earnings']:
                company_name = match
                break
    
    # Time period extraction - IMPROVED
    detected_period = None
    current_year = datetime.now().year
    
    # Check for specific year mentions (e.g., "in 2024", "2024 earnings", "fiscal 2024")
    year_pattern = r'(?:in\s+|for\s+|fiscal\s+|fy\s*)?(\d{4})(?:\s+earnings|\s+financials|\s+results)?'
    year_match = re.search(year_pattern, query_lower)
    if year_match:
        year = year_match.group(1)
        detected_period = f"FY {year}"
    
    # Check for specific quarters
    elif re.search(r'q(\d)\s*(\d{4})', query_lower):
        quarter_match = re.search(r'q(\d)\s*(\d{4})', query_lower)
        detected_period = f"Q{quarter_match.group(1)} {quarter_match.group(2)}"
    
    # Check for year ranges
    elif "past 5 years" in query_lower or "last 5 years" in query_lower:
        detected_period = f"FY {current_year-5} to FY {current_year}"
    elif "past 3 years" in query_lower or "last 3 years" in query_lower:
        detected_period = f"FY {current_year-3} to FY {current_year}"
    elif "past year" in query_lower or "last year" in query_lower:
        detected_period = f"FY {current_year-1}"
    elif "this year" in query_lower or "current year" in query_lower:
        detected_period = f"FY {current_year}"
    
    # Check for quarters without year
    elif re.search(r'q(\d)(?:\s|$)', query_lower):
        quarter_match = re.search(r'q(\d)(?:\s|$)', query_lower)
        # Assume current year for quarter without year
        detected_period = f"Q{quarter_match.group(1)} {current_year}"
    
    # Check for fiscal year pattern
    elif re.search(r'fy\s*(\d{4})', query_lower):
        fy_match = re.search(r'fy\s*(\d{4})', query_lower)
        detected_period = f"FY {fy_match.group(1)}"
    
    return company_name, detected_period
# Company Discovery Agent Prompt
COMPANY_DISCOVERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a company discovery agent. Your job is to identify the company being discussed and find its trading information.

Given the user query: {query}
Extracted company name: {extracted_company}
Discovery results: {discovery_results}

Your task:
1. Confirm if the discovered company matches the user's intent
2. If multiple matches exist, choose the most likely one
3. Extract and validate the ticker symbol
4. Determine confidence in the match

Return a JSON response with:
- company_name: The official company name
- ticker: The stock ticker symbol
- cik: SEC CIK number (if available)
- confidence: 0-1 score of match accuracy
- alternative_matches: List of other possible companies if confidence < 0.8
- reasoning: Brief explanation of your choice

Be careful with ambiguous names. For example:
- "Apple" usually means Apple Inc. (AAPL)
- "Meta" means Meta Platforms Inc. (META), formerly Facebook
- "Google" means Alphabet Inc. (GOOGL/GOOG)
"""),
    MessagesPlaceholder(variable_name="messages"),
])

# Updated Graph State to include company discovery
class FinancialAnalysisState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    extracted_company: str  # Raw extracted company
    company: str  # Resolved company name
    ticker: str  # Resolved ticker
    cik: str  # Resolved CIK
    period: str
    parsed_data: Dict
    kpis: Dict
    risk_assessment: Dict
    insights: Dict
    current_agent: str
    uploaded_content: str
    market_data: Dict
    data_sources: List[Dict]
    period_validation: Dict
    response_mode: str
    company_confidence: float

# Create company discovery agent
# def create_company_discovery_agent(llm):
#     """Agent to discover and validate company information"""
#     def company_discoverer(state: FinancialAnalysisState):
#         extracted_company = state.get("extracted_company", "")
        
#         if not extracted_company:
#             # If no company extracted, ask the LLM to identify it from the query
#             extract_prompt = f"Extract the company name from this query: '{state['query']}'. Return only the company name or ticker symbol."
#             response = llm.invoke([HumanMessage(content=extract_prompt)])
#             extracted_company = response.content.strip()
        
#         # Use discovery function
#         discovery_results = discover_company_info(extracted_company)
        
#         # Get CIK if not found
#         if not discovery_results.get("cik") and discovery_results.get("ticker"):
#             cik = search_company_cik(
#                 discovery_results.get("full_name", extracted_company),
#                 discovery_results.get("ticker")
#             )
#             discovery_results["cik"] = cik
        
#         # If low confidence, try to improve with LLM
#         if discovery_results["confidence"] < 0.7:
#             prompt = COMPANY_DISCOVERY_PROMPT.format_messages(
#                 query=state["query"],
#                 extracted_company=extracted_company,
#                 discovery_results=json.dumps(discovery_results),
#                 messages=state["messages"]
#             )
            
#             response = llm.invoke(prompt)
            
#             try:
#                 import re
#                 json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
#                 if json_match:
#                     llm_result = json.loads(json_match.group())
                    
#                     # Update discovery results with LLM insights
#                     if llm_result.get("confidence", 0) > discovery_results["confidence"]:
#                         discovery_results.update({
#                             "ticker": llm_result.get("ticker", discovery_results["ticker"]),
#                             "full_name": llm_result.get("company_name", discovery_results["full_name"]),
#                             "cik": llm_result.get("cik", discovery_results["cik"]),
#                             "confidence": llm_result.get("confidence", discovery_results["confidence"])
#                         })
#             except:
#                 pass
        
#         # Update state with discovered information
#         return {
#             "messages": [AIMessage(content=f"Discovered company: {discovery_results['full_name']} ({discovery_results['ticker']}) with confidence {discovery_results['confidence']:.2f}")],
#             "company": discovery_results["full_name"],
#             "ticker": discovery_results["ticker"],
#             "cik": discovery_results["cik"],
#             "company_confidence": discovery_results["confidence"],
#             "current_agent": "parser"
#         }
    
#     return company_discoverer

# Enhanced company discovery agent that uses LLM when confidence is low
def create_company_discovery_agent(llm):
    """Agent to discover and validate company information"""
    def company_discoverer(state: FinancialAnalysisState):
        extracted_company = state.get("extracted_company", "")
        
        if not extracted_company:
            # If no company extracted, ask the LLM to identify it from the query
            extract_prompt = f"Extract the company name from this query: '{state['query']}'. Return only the company name or ticker symbol."
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            extracted_company = response.content.strip()
        
        # Use discovery function
        discovery_results = discover_company_info(extracted_company)
        
        # If low confidence or needs LLM help, use LLM to improve
        if discovery_results["confidence"] < 0.7 or discovery_results.get("needs_llm_help", False):
            # Enhanced prompt for LLM
            llm_prompt = f"""Identify the correct stock ticker for this company query: "{extracted_company}"

Common examples:
- "State Street" or "State Street Corporation" → STT (State Street Corp)
- "Snowflake" → SNOW (Snowflake Inc)
- "Palantir" → PLTR (Palantir Technologies)
- "ServiceNow" or "Service Now" → NOW (ServiceNow Inc)

If this is a well-known public company, provide the correct ticker symbol.
Return JSON with: {{"ticker": "XXX", "company_name": "Full Company Name", "confidence": 0.0-1.0}}

Query: {extracted_company}"""
            
            response = llm.invoke([HumanMessage(content=llm_prompt)])
            
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    llm_result = json.loads(json_match.group())
                    
                    # Verify the LLM's suggestion with yfinance
                    suggested_ticker = llm_result.get("ticker", "")
                    if suggested_ticker and suggested_ticker != "UNKNOWN":
                        try:
                            stock = yf.Ticker(suggested_ticker)
                            info = stock.info
                            if info and 'longName' in info:
                                discovery_results = {
                                    "ticker": suggested_ticker,
                                    "full_name": info.get('longName', llm_result.get("company_name", extracted_company)),
                                    "cik": None,
                                    "confidence": min(llm_result.get("confidence", 0.8), 0.9),
                                    "method": "llm_assisted"
                                }
                        except:
                            # LLM suggestion didn't verify, keep original but with LLM's name
                            discovery_results["full_name"] = llm_result.get("company_name", discovery_results["full_name"])
                            discovery_results["confidence"] = min(discovery_results["confidence"], 0.5)
            except:
                pass
        
        # Get CIK if not found
        if not discovery_results.get("cik") and discovery_results.get("ticker"):
            cik = search_company_cik(
                discovery_results.get("full_name", extracted_company),
                discovery_results.get("ticker")
            )
            discovery_results["cik"] = cik
        
        # Final validation - if ticker is suspicious (like "STATE" or "SNOWF"), mark low confidence
        suspicious_patterns = [
            len(discovery_results["ticker"]) == 5 and discovery_results["ticker"] == extracted_company.upper()[:5],
            discovery_results["ticker"] in ["STATE", "SNOWF", "UNKNOWN"],
            discovery_results["method"] == "fallback"
        ]
        
        if any(suspicious_patterns):
            discovery_results["confidence"] = min(discovery_results["confidence"], 0.3)
        
        # Update state with discovered information
        return {
            "messages": [AIMessage(content=f"Discovered company: {discovery_results['full_name']} ({discovery_results['ticker']}) with confidence {discovery_results['confidence']:.2f}")],
            "company": discovery_results["full_name"],
            "ticker": discovery_results["ticker"],
            "cik": discovery_results["cik"],
            "company_confidence": discovery_results["confidence"],
            "current_agent": "parser"
        }
    
    return company_discoverer
# Update the detect_query_intent function (keep existing)
def detect_query_intent(query: str):
    """Detect the user's intent from their query"""
    query_lower = query.lower()
    
    # Investment decision queries
    investment_keywords = ["invest", "buy", "sell", "hold", "should i", "good investment", "worth buying", "risky"]
    
    # Analysis queries
    analysis_keywords = ["analyze", "review", "summarize", "explain", "tell me about", "what are", "show me"]
    
    # Risk queries
    risk_keywords = ["risk", "risky", "safe", "dangerous", "volatility"]
    
    # Comparison queries
    comparison_keywords = ["compare", "versus", "vs", "better than", "difference between"]
    
    # Document-specific queries
    document_keywords = ["attached", "document", "pdf", "file", "earnings report", "10-k", "10-q"]
    
    # Determine primary intent
    if any(keyword in query_lower for keyword in investment_keywords):
        return "investment_decision"
    elif any(keyword in query_lower for keyword in risk_keywords):
        return "risk_assessment"
    elif any(keyword in query_lower for keyword in comparison_keywords):
        return "comparison"
    elif any(keyword in query_lower for keyword in document_keywords) and any(keyword in query_lower for keyword in analysis_keywords):
        return "document_analysis"
    elif any(keyword in query_lower for keyword in analysis_keywords):
        return "general_analysis"
    else:
        return "general_query"

# Function to parse period to date with support for ranges
def parse_period_to_date(period: str):
    """Convert period string to date range"""
    current_year = datetime.now().year
    
    # Handle year ranges (e.g., "FY 2019 to FY 2024")
    range_pattern = r'FY (\d{4}) to FY (\d{4})'
    range_match = re.match(range_pattern, period)
    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        return datetime(start_year, 1, 1), datetime(end_year, 12, 31)
    
    # Handle fiscal year
    if period.startswith("FY"):
        year = int(period.replace("FY ", ""))
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        return start_date, end_date
    
    # Handle quarters
    quarter_match = re.match(r"Q(\d) (\d{4})", period)
    if quarter_match:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        
        # Quarter to month mapping
        quarter_months = {
            1: (1, 3),   # Jan-Mar
            2: (4, 6),   # Apr-Jun
            3: (7, 9),   # Jul-Sep
            4: (10, 12)  # Oct-Dec
        }
        
        start_month, end_month = quarter_months[quarter]
        start_date = datetime(year, start_month, 1)
        # Get last day of end month
        if end_month == 12:
            end_date = datetime(year, 12, 31)
        else:
            end_date = datetime(year, end_month + 1, 1) - timedelta(days=1)
        
        return start_date, end_date
    
    # Default to current quarter if parsing fails
    return None, None

def get_actual_data_period(data_date):
    """Convert a date to period string (Q1 2024, etc)"""
    if not data_date:
        return "Unknown"
    
    month = data_date.month
    year = data_date.year
    
    if month <= 3:
        return f"Q1 {year}"
    elif month <= 6:
        return f"Q2 {year}"
    elif month <= 9:
        return f"Q3 {year}"
    else:
        return f"Q4 {year}"

def clean_llm_text(text: str) -> str:
    """Clean text from LLM output while preserving word boundaries"""
    if not text:
        return text

    import re
    # Step 1: Normalize spacing around asterisks to avoid word merging
    # Add spaces around asterisks first to protect word boundaries
    text = re.sub(r'(\S)\*(\S)', r'\1 * \2', text)  # e.g., "word*word" -> "word * word"
    text = re.sub(r'\s*\*\s*', ' ', text)           # Normalize all * to single space

    # Step 2: Remove stray single asterisks (not double for bold markdown)
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)

    # Step 3: Remove strange unicode or leftover asterisk characters
    text = text.replace('∗', '')  # Unicode asterisk
    text = text.replace('*', '')  # Fallback, though most handled above

    # Step 4: Unescape characters
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '    ')

    # Step 5: Normalize whitespace
    text = re.sub(r' +', ' ', text)        # Reduce multiple spaces
    text = re.sub(r'\n{3,}', '\n\n', text) # Reduce excessive newlines

    return text.strip()

# Initialize LLMs
def get_llms():
    """Initialize all LLMs with API keys"""
    llms = {}
    
    # OpenAI GPT-4
    if os.getenv("OPENAI_API_KEY"):
        llms["gpt-4"] = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    # Groq (Fast inference with open models)
    if os.getenv("GROQ_API_KEY"):
        # Groq offers multiple models - using mistral for best quality
        llms["groq-mixtral"] = ChatGroq(
            model="mistral-saba-24b",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        # Also add Llama 2 70B as an option
        llms["groq-llama3"] = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
    
    # Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        llms["gemini-pro"] = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    # Anthropic Claude (keeping as optional)
    if os.getenv("ANTHROPIC_API_KEY"):
        llms["claude-3"] = ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.7,
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    return llms



def fetch_yahoo_finance_data(ticker: str, period: str):
    """Fetch financial data from Yahoo Finance for specific period"""
    
    # Check cache first
    cache_key = f"{ticker}_{period}_{datetime.now().strftime('%Y-%m-%d')}"
    if cache_key in st.session_state.yahoo_cache:
        return st.session_state.yahoo_cache[cache_key]
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - st.session_state.last_yahoo_request
    if time_since_last < 2:
        time.sleep(2 - time_since_last)
    
    try:
        st.session_state.last_yahoo_request = time.time()
        stock = yf.Ticker(ticker)
        
        # Parse the requested period
        start_date, end_date = parse_period_to_date(period)
        
        result = {
            "requested_period": period,
            "data_period": "Unknown",
            "data_date": None,
            "period_match": False,
            "ticker": ticker
        }
        
        # Get basic info
        try:
            info = stock.info
            if info:
                result.update({
                    "company_name": info.get("longName", ticker),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "eps": info.get("trailingEps"),
                    "52_week_high": info.get("fiftyTwoWeekHigh"),
                    "52_week_low": info.get("fiftyTwoWeekLow"),
                    "dividend_yield": info.get("dividendYield"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry")
                })
        except Exception as e:
            st.warning(f"Could not fetch basic info for {ticker}: {str(e)}")
        
        # For historical ranges, get price history
        if "to" in period:
            try:
                history = stock.history(start=start_date, end=end_date)
                if not history.empty:
                    result["historical_prices"] = history['Close'].to_dict()
                    result["price_change"] = ((history['Close'].iloc[-1] - history['Close'].iloc[0]) / history['Close'].iloc[0] * 100)
            except Exception:
                pass
        
        # Get quarterly financials
        try:
            quarterly_financials = stock.quarterly_financials
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            # Find the column (date) that matches our period
            target_data = None
            actual_date = None
            
            if not quarterly_financials.empty and start_date and end_date:
                # First, try to find data within the requested date range
                for col_date in quarterly_financials.columns:
                    if start_date <= col_date <= end_date:
                        target_data = col_date
                        actual_date = col_date
                        break
                
                # If no exact match and we're looking for a full year (like "2024")
                if target_data is None and period.startswith("FY"):
                    # For full year requests, get ALL quarters in that year
                    year_data = []
                    for col_date in quarterly_financials.columns:
                        if col_date.year == start_date.year:
                            year_data.append(col_date)
                    
                    if year_data:
                        # Use the most recent quarter from the requested year
                        target_data = max(year_data)
                        actual_date = target_data
                        # Note: You could also aggregate all quarters here if needed
                
                # If still no match, get the most recent data WITHIN the requested year
                if target_data is None:
                    valid_dates = [d for d in quarterly_financials.columns 
                                 if d >= start_date and d <= end_date]
                    if valid_dates:
                        target_data = max(valid_dates)
                        actual_date = target_data
                    else:
                        # Only get data from BEFORE the requested period, not after
                        past_dates = [d for d in quarterly_financials.columns if d <= end_date]
                        if past_dates:
                            target_data = max(past_dates)
                            actual_date = target_data
                        else:
                            # No suitable data found
                            result["error"] = f"No financial data available for {period}"
                            return result
            
            # Extract financial data for the target period
            if target_data is not None:
                # Update period information
                if actual_date is not None:
                    # Convert pandas Timestamp to string
                    if hasattr(actual_date, 'strftime'):
                        actual_date_str = actual_date.strftime('%Y-%m-%d')
                    else:
                        actual_date_str = str(actual_date)
                    
                    result["data_date"] = actual_date_str
                    result["data_period"] = get_actual_data_period(actual_date)
                    
                    # Check if the actual data year matches requested year
                    if period.startswith("FY") and actual_date:
                        requested_year = int(period.replace("FY ", ""))
                        actual_year = actual_date.year
                        result["period_match"] = (requested_year == actual_year)
                    else:
                        result["period_match"] = (result["data_period"] == period)
                
                # Get financial data
                if not quarterly_financials.empty:
                    fin_data = quarterly_financials[target_data]
                    result.update({
                        "revenue": fin_data.get("Total Revenue"),
                        "gross_profit": fin_data.get("Gross Profit"),
                        "operating_income": fin_data.get("Operating Income"),
                        "net_income": fin_data.get("Net Income"),
                        "ebitda": fin_data.get("EBITDA")
                    })
                
                if not quarterly_balance.empty and target_data in quarterly_balance.columns:
                    bal_data = quarterly_balance[target_data]
                    result.update({
                        "total_assets": bal_data.get("Total Assets"),
                        "total_liabilities": bal_data.get("Total Liabilities Net Minority Interest"),
                        "total_equity": bal_data.get("Total Stockholder Equity"),
                        "cash": bal_data.get("Cash"),
                        "total_debt": bal_data.get("Total Debt")
                    })
                
                if not quarterly_cashflow.empty and target_data in quarterly_cashflow.columns:
                    cf_data = quarterly_cashflow[target_data]
                    result.update({
                        "operating_cash_flow": cf_data.get("Operating Cash Flow"),
                        "free_cash_flow": cf_data.get("Free Cash Flow"),
                        "capital_expenditures": cf_data.get("Capital Expenditures")
                    })
            
        except Exception as e:
            st.warning(f"Could not fetch complete quarterly data for {ticker}: {str(e)}")
        
        # Clean up None values and convert any remaining timestamps
        cleaned_result = {}
        for k, v in result.items():
            if v is not None and not pd.isna(v):
                # Convert any pandas Timestamp to string
                if hasattr(v, 'strftime'):
                    cleaned_result[k] = v.strftime('%Y-%m-%d')
                elif pd.api.types.is_datetime64_any_dtype(type(v)):
                    cleaned_result[k] = str(v)
                else:
                    cleaned_result[k] = v
        
        # Cache the result
        st.session_state.yahoo_cache[cache_key] = cleaned_result
        
        return cleaned_result
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            st.warning("⚠️ Yahoo Finance rate limit reached. Try again in a few minutes.")
        else:
            st.warning(f"Could not fetch Yahoo Finance data for {ticker}: {error_msg}")
        
        return {
            "requested_period": period,
            "data_period": "Unknown",
            "period_match": False,
            "ticker": ticker,
            "error": error_msg
        }
# Function to fetch SEC EDGAR data using CIK
def fetch_sec_edgar_data(company: str, period: str, cik: str = None):
    """Fetch financial data from SEC EDGAR API for specific period"""
    try:
        if not cik:
            return {"error": "No CIK available for SEC data"}
        
        # Parse period to get date range
        start_date, end_date = parse_period_to_date(period)
        
        result = {
            "requested_period": period,
            "data_period": "Unknown",
            "period_match": False,
            "cik": cik
        }
        
        # Concepts to fetch
        concepts = {
            "Assets": "total_assets",
            "Revenues": "revenue",
            "NetIncomeLoss": "net_income",
            "GrossProfit": "gross_profit",
            "Liabilities": "total_liabilities",
            "CashAndCashEquivalentsAtCarryingValue": "cash",
            "StockholdersEquity": "total_equity"
        }
        
        for concept, field_name in concepts.items():
            try:
                url = SEC_BASE_URL.format(cik=cik.lstrip('0'), concept=concept)
                response = requests.get(url, headers=SEC_HEADERS, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    units = data.get("units", {})
                    
                    if "USD" in units:
                        entries = units["USD"]
                        
                        # Find entry matching our period
                        matching_entry = None
                        
                        if start_date and end_date:
                            for entry in entries:
                                # Parse entry date
                                entry_end = entry.get("end", "")
                                if entry_end:
                                    entry_date = datetime.strptime(entry_end, "%Y-%m-%d")
                                    
                                    # Check if this entry falls within our period
                                    if start_date <= entry_date <= end_date:
                                        # Prefer 10-Q over 10-K for quarterly data
                                        if period.startswith("Q") and "10-Q" in entry.get("form", ""):
                                            matching_entry = entry
                                            break
                                        elif period.startswith("FY") and "10-K" in entry.get("form", ""):
                                            matching_entry = entry
                                            break
                                        elif matching_entry is None:
                                            matching_entry = entry
                        
                        # If no exact match, get most recent before requested period
                        if not matching_entry and entries and end_date:
                            valid_entries = []
                            for entry in entries:
                                entry_end = entry.get("end", "")
                                if entry_end:
                                    entry_date = datetime.strptime(entry_end, "%Y-%m-%d")
                                    if entry_date <= end_date:
                                        valid_entries.append((entry_date, entry))
                            
                            if valid_entries:
                                valid_entries.sort(key=lambda x: x[0], reverse=True)
                                matching_entry = valid_entries[0][1]
                        
                        # Use the matching entry
                        if matching_entry:
                            result[field_name] = matching_entry.get("val")
                            
                            # Update period info from first successful fetch
                            if result["data_period"] == "Unknown":
                                entry_date = datetime.strptime(matching_entry.get("end", ""), "%Y-%m-%d")
                                result["data_period"] = get_actual_data_period(entry_date)
                                result["data_date"] = entry_date.strftime('%Y-%m-%d')
                                result["period_match"] = (result["data_period"] == period)
                                result["form_type"] = matching_entry.get("form", "")
                
            except Exception as e:
                continue
        
        return result
        
    except Exception as e:
        return {
            "requested_period": period,
            "data_period": "Unknown",
            "period_match": False,
            "error": str(e)
        }

# Function to extract text from PDF with RAG
def process_pdf_with_rag(pdf_file, query: str = "", company: str = "", period: str = ""):
    """Process PDF with RAG for better retrieval"""
    try:
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        
        # Create text splitter for financial documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Split into chunks
        chunks = text_splitter.split_text(full_text)
        
        # If we have OpenAI API key, use embeddings
        if os.getenv("OPENAI_API_KEY") and chunks:
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_file.name,
                        "page": i,
                        "company": company,
                        "period": period
                    }
                )
                documents.append(doc)
            
            # Create embeddings and vector store
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Create a query combining user query and financial data needs
            retrieval_query = f"""
            {query}
            Extract financial information including:
            - Revenue and earnings
            - Key financial metrics
            - Risk factors
            - Business performance
            Company: {company}
            Period: {period}
            """
            
            # Search for relevant chunks (limit to top 5 to avoid token limits)
            relevant_docs = vectorstore.similarity_search(retrieval_query, k=5)
            
            # Combine relevant chunks
            relevant_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            return {
                "success": True,
                "content": relevant_text[:8000],  # Limit to avoid token issues
                "num_chunks": len(chunks),
                "chunks_used": len(relevant_docs)
            }
        else:
            # Fallback without embeddings - just return first portion
            return {
                "success": True,
                "content": full_text[:8000],  # Limit to avoid token issues
                "num_chunks": len(chunks),
                "chunks_used": 1
            }
            
    except Exception as e:
        return {
            "success": False,
            "content": "",
            "error": str(e)
        }

# Updated prompts with company discovery awareness
PARSER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial document parser agent. Extract and structure financial data from the given information.
    
    Company: {company}
    Ticker: {ticker}
    Requested Period: {period}
    Response Mode: {response_mode}
    Company Discovery Confidence: {company_confidence}
    
    IMPORTANT: The data provided may be from a different period than requested.
    Actual data period: {period_validation}
    
    Additional context from uploaded document: {uploaded_content}
    
    Market data from APIs: {market_data}
    
    Provide the following in JSON format:
    - revenue (in USD)
    - gross_profit (in USD)
    - net_income (in USD)
    - total_assets (in USD)
    - total_liabilities (in USD)
    - cash_flow (in USD)
    - earnings_per_share (EPS)
    - market_cap (if available)
    - data_period (actual period of the data)
    - data_quality_score (0-100, based on how well data matches requested period and company confidence)
    
    If company confidence is low (<0.5), adjust data_quality_score accordingly.
    Always return valid JSON."""),
    MessagesPlaceholder(variable_name="messages"),
])

# Keep the rest of the prompts the same (KPI_PROMPT, RISK_PROMPT, INSIGHT_PROMPT)...
# [Keep all the existing prompt definitions as they are]

KPI_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a KPI extraction agent. Calculate key financial metrics from the parsed data.
    
    Given the financial data: {parsed_data}
    Response Mode: {response_mode}
    
    Calculate and return in JSON format:
    - gross_margin (%)
    - net_margin (%)
    - debt_to_equity ratio
    - return_on_assets (%)
    - return_on_equity (%)
    - current_ratio
    - quick_ratio
    - revenue_growth (% if historical data available)
    - ebitda_margin (%)
    - price_to_earnings (if market data available)
    
    Show your calculations and ensure accuracy."""),
    MessagesPlaceholder(variable_name="messages"),
])

RISK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial risk assessment agent. Analyze the financial health and risks.
    
    Given the KPIs: {kpis}
    Financial Data: {parsed_data}
    Response Mode: {response_mode}
    
    Provide a comprehensive risk assessment in JSON format including:
    - overall_risk_score (0-100)
    - risk_level (Low/Medium/High)
    - debt_risk (0-100)
    - liquidity_risk (0-100)
    - profitability_risk (0-100)
    - market_risk (0-100)
    - operational_risk (0-100)
    - risk_factors (list of specific concerns)
    - mitigation_strategies (list of recommendations)
    
    Base your assessment on financial best practices and industry standards."""),
    MessagesPlaceholder(variable_name="messages"),
])

INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a financial analyst. Provide insights based on the query intent.

Company: {company} ({ticker})
Period: {period}
Query: {query}
Intent: {query_intent}
Mode: {response_mode}

Key Data:
- Revenue: ${revenue}M, Net Margin: {net_margin}%
- P/E: {pe_ratio}, Risk Score: {risk_score}/100
- Price: ${current_price}

INSTRUCTIONS BY INTENT:

1. investment_decision:
   - Give clear BUY/HOLD/SELL/WAIT
   - Include current price context
   - Separate advice for current holders vs new investors
   - Mention volatility if relevant

2. document_analysis:
   - Summarize key findings
   - NO investment advice unless asked

3. risk_assessment:
   - Focus on risk factors
   - Provide mitigation strategies

4. general_analysis:
   - Balanced overview
   - Minimal investment advice

Return JSON with ONLY these fields:
- query_specific_answer: Direct answer to user's question
- key_strengths: 2-3 items max
- key_concerns: 2-3 items max
- recommendations: 2-3 investor-focused actions
- investment_decision: Only if intent is investment_decision
- confidence_note: Only if company confidence < 0.5"""),
    MessagesPlaceholder(variable_name="messages"),
])

def create_parser_agent(llm):
    """Document parsing agent with improved market data handling"""
    def parser(state: FinancialAnalysisState) -> Dict:
        # Ensure we have all required state fields
        period_validation = state.get("period_validation", {
            "requested": state.get("period", "Unknown"),
            "actual": "Unknown",
            "matches": False
        })
        
        # Get market data
        market_data = state.get("market_data", {})
        
        # Use custom encoder for JSON serialization
        prompt = PARSER_PROMPT.format_messages(
            company=state.get("company", "Unknown"),
            ticker=state.get("ticker", "Unknown"),
            period=state.get("period", "Unknown"),
            response_mode=state.get("response_mode", "concise"),
            company_confidence=state.get("company_confidence", 0.5),
            period_validation=json.dumps(period_validation, cls=DateTimeEncoder),
            uploaded_content=state.get("uploaded_content", "No document uploaded"),
            market_data=json.dumps(market_data, cls=DateTimeEncoder),
            messages=state.get("messages", [])
        )
        
        response = llm.invoke(prompt)
        
        try:
            # Extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                
                # Merge with market data if LLM missed some fields
                for key in ["revenue", "gross_profit", "net_income", "total_assets", 
                           "total_liabilities", "cash_flow", "earnings_per_share", "market_cap"]:
                    if key in market_data and (key not in parsed_data or parsed_data[key] == 0):
                        parsed_data[key] = market_data[key]
            else:
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            print(f"Parser error: {str(e)}")
            
            # Better fallback - use market data if available
            parsed_data = {}
            
            # First, try to use actual market data
            if market_data:
                # Map market data fields to parsed data fields
                field_mapping = {
                    "revenue": "revenue",
                    "gross_profit": "gross_profit", 
                    "net_income": "net_income",
                    "total_assets": "total_assets",
                    "total_liabilities": "total_liabilities",
                    "operating_cash_flow": "cash_flow",
                    "free_cash_flow": "cash_flow",  # Use free cash flow if operating not available
                    "eps": "earnings_per_share",
                    "market_cap": "market_cap"
                }
                
                for parsed_field, market_field in field_mapping.items():
                    if market_field in market_data and market_data[market_field] is not None:
                        parsed_data[parsed_field] = market_data[market_field]
                
                # Handle cash flow specially - prefer operating, then free
                if "operating_cash_flow" in market_data and market_data["operating_cash_flow"] is not None:
                    parsed_data["cash_flow"] = market_data["operating_cash_flow"]
                elif "free_cash_flow" in market_data and market_data["free_cash_flow"] is not None:
                    parsed_data["cash_flow"] = market_data["free_cash_flow"]
            
            # Only use defaults for fields not found in market data
            default_values = {
                "revenue": 1000000000,
                "gross_profit": 400000000,
                "net_income": 100000000,
                "total_assets": 5000000000,
                "total_liabilities": 2000000000,
                "cash_flow": 150000000,
                "earnings_per_share": 5.0,
                "market_cap": 1000000000000
            }
            
            # Fill in any missing fields with defaults
            for key, default in default_values.items():
                if key not in parsed_data or parsed_data[key] is None or parsed_data[key] == 0:
                    # Special handling for some fields
                    if key == "gross_profit" and "revenue" in parsed_data and "gross_margin" in market_data:
                        # Calculate gross profit from revenue and margin if available
                        parsed_data[key] = parsed_data["revenue"] * (market_data["gross_margin"] / 100)
                    elif key == "market_cap" and "price" in market_data and market_data["price"]:
                        # Use actual market cap from market data if available
                        parsed_data[key] = market_data.get("market_cap", default)
                    else:
                        parsed_data[key] = default
            
            # Always use the actual data period if available
            parsed_data["data_period"] = market_data.get("data_period", state.get("period", "Unknown"))
            
            # Calculate data quality score based on how much real data we have
            real_data_count = sum(1 for k, v in parsed_data.items() 
                                 if k != "data_quality_score" and k != "data_period" 
                                 and v != default_values.get(k))
            total_fields = len(default_values)
            data_quality = (real_data_count / total_fields) * 100
            
            # Adjust for company confidence
            if state.get("company_confidence", 0.5) < 0.5:
                data_quality *= 0.6
            
            parsed_data["data_quality_score"] = max(10, min(100, data_quality))
        
        # Log what data source was used
        data_source = "LLM extraction" if 'json_match' in locals() else "Market data fallback"
        print(f"Parser used: {data_source} for {state.get('company', 'Unknown')}")
        
        # Return the updated state
        return {
            "messages": state.get("messages", []) + [AIMessage(content=f"Parsed financial data from {data_source}")],
            "parsed_data": parsed_data,
            "current_agent": "kpi_extractor"
        }
    
    return parser
def create_kpi_agent(llm):
    """KPI calculation agent"""
    def kpi_extractor(state: FinancialAnalysisState):
        prompt = KPI_PROMPT.format_messages(
            parsed_data=json.dumps(state["parsed_data"], indent=2),
            response_mode=state.get("response_mode", "concise"),
            messages=state["messages"]
        )
        
        response = llm.invoke(prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                kpis = json.loads(json_match.group())
            else:
                # Calculate KPIs from parsed data
                pd_data = state["parsed_data"]
                revenue = pd_data.get("revenue", 1) or 1  # Avoid None and 0
                total_assets = pd_data.get("total_assets", 1) or 1
                total_liabilities = pd_data.get("total_liabilities", 0) or 0
                gross_profit = pd_data.get("gross_profit", 0) or 0
                net_income = pd_data.get("net_income", 0) or 0
                
                total_equity = max(total_assets - total_liabilities, 1)  # Avoid negative or zero equity
                
                kpis = {
                    "gross_margin": (gross_profit / revenue * 100) if revenue > 0 else 0,
                    "net_margin": (net_income / revenue * 100) if revenue > 0 else 0,
                    "debt_to_equity": total_liabilities / total_equity,
                    "return_on_assets": (net_income / total_assets * 100) if total_assets > 0 else 0,
                    "return_on_equity": (net_income / total_equity * 100) if total_equity > 0 else 0,
                    "current_ratio": 1.5,
                    "quick_ratio": 1.2,
                    "revenue_growth": 15.0,
                    "ebitda_margin": 25.0,
                    "price_to_earnings": pd_data.get("market_cap", 0) / (net_income * 4) if net_income > 0 else 0
                }
        except:
            kpis = {
                "gross_margin": 40.0,
                "net_margin": 10.0,
                "debt_to_equity": 0.5,
                "return_on_assets": 8.0,
                "return_on_equity": 15.0,
                "current_ratio": 1.5,
                "quick_ratio": 1.2,
                "revenue_growth": 15.0,
                "ebitda_margin": 25.0,
                "price_to_earnings": 25.0
            }
        
        return {
            "messages": [AIMessage(content=f"Calculated KPIs: {json.dumps(kpis, indent=2)}")],
            "kpis": kpis,
            "current_agent": "risk_assessor"
        }
    
    return kpi_extractor

def create_risk_agent(llm):
    """Risk assessment agent"""
    def risk_assessor(state: FinancialAnalysisState):
        prompt = RISK_PROMPT.format_messages(
            kpis=json.dumps(state["kpis"], indent=2),
            parsed_data=json.dumps(state["parsed_data"], indent=2),
            response_mode=state.get("response_mode", "concise"),
            messages=state["messages"]
        )
        
        response = llm.invoke(prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                risk_assessment = json.loads(json_match.group())
            else:
                # Calculate risk scores based on KPIs
                kpis = state["kpis"]
                debt_to_equity = kpis.get("debt_to_equity", 0)
                current_ratio = kpis.get("current_ratio", 1)
                net_margin = kpis.get("net_margin", 0)
                
                # Risk calculations
                debt_risk = min(100, debt_to_equity * 30)
                liquidity_risk = max(0, 100 - current_ratio * 50)
                profitability_risk = max(0, 100 - net_margin * 5)
                
                # Adjust for company confidence
                confidence_adjustment = 1 + (0.5 - state.get("company_confidence", 0.5)) * 0.5
                
                overall_risk = (debt_risk + liquidity_risk + profitability_risk) / 3 * confidence_adjustment
                
                risk_assessment = {
                    "overall_risk_score": overall_risk,
                    "risk_level": "High" if overall_risk > 70 else "Medium" if overall_risk > 40 else "Low",
                    "debt_risk": debt_risk,
                    "liquidity_risk": liquidity_risk,
                    "profitability_risk": profitability_risk,
                    "market_risk": 50.0,
                    "operational_risk": 40.0,
                    "risk_factors": [],
                    "mitigation_strategies": []
                }
                
                # Add risk factors
                if debt_risk > 60:
                    risk_assessment["risk_factors"].append("High debt levels relative to equity")
                    risk_assessment["mitigation_strategies"].append("Consider debt reduction strategies")
                if liquidity_risk > 60:
                    risk_assessment["risk_factors"].append("Potential liquidity concerns")
                    risk_assessment["mitigation_strategies"].append("Improve working capital management")
                if profitability_risk > 60:
                    risk_assessment["risk_factors"].append("Low profitability margins")
                    risk_assessment["mitigation_strategies"].append("Focus on operational efficiency")
                if state.get("company_confidence", 0.5) < 0.5:
                    risk_assessment["risk_factors"].append("Uncertainty in company identification")
                    risk_assessment["mitigation_strategies"].append("Verify company information before making decisions")
        except:
            risk_assessment = {
                "overall_risk_score": 45.0,
                "risk_level": "Medium",
                "debt_risk": 30.0,
                "liquidity_risk": 40.0,
                "profitability_risk": 50.0,
                "market_risk": 60.0,
                "operational_risk": 40.0,
                "risk_factors": ["Moderate debt levels", "Market volatility"],
                "mitigation_strategies": ["Improve operational efficiency", "Diversify revenue streams"]
            }
        
        return {
            "messages": [AIMessage(content=f"Risk assessment complete: {json.dumps(risk_assessment, indent=2)}")],
            "risk_assessment": risk_assessment,
            "current_agent": "insight_generator"
        }
    
    return risk_assessor

# 3. Improved default insights generator
def generate_improved_default_insights(state, query_intent, market_data=None):
    """Generate better default insights for investment decisions"""
    company = state.get("company", "Unknown")
    ticker = state.get("ticker", "Unknown")
    kpis = state.get("kpis", {})
    risk = state.get("risk_assessment", {})
    parsed_data = state.get("parsed_data", {})
    
    # Get market data with None handling
    market_data = market_data or {}
    current_price = market_data.get('price', 0) or 0
    pe_ratio = kpis.get('price_to_earnings', 0) or 0
    net_margin = kpis.get('net_margin', 0) or 0
    gross_margin = kpis.get('gross_margin', 0) or 0
    risk_score = risk.get('overall_risk_score', 0) or 0
    risk_level = risk.get('risk_level', 'Medium') or 'Medium'
    
    if query_intent == "investment_decision":
        # Determine investment decision with better logic
        if risk_score < 40 and net_margin > 10 and pe_ratio < 30:
            decision = "BUY"
            reason = "strong fundamentals with reasonable valuation"
        elif risk_score > 70 or net_margin < 0:
            decision = "SELL"
            reason = "high risk or unprofitable operations"
        elif risk_score < 60 and net_margin > 5:
            decision = "HOLD"
            reason = "mixed signals - stable but watch for improvements"
        else:
            decision = "WAIT"
            reason = "valuation concerns or unclear trend"
        
        # Investment-specific answer
        price_info = f"at ${current_price:.2f}" if current_price > 0 else ""
        
        answer = f"{company} ({ticker}) {price_info} presents a {decision} opportunity - {reason}. "
        answer += f"Risk: {risk_level} ({risk_score:.0f}/100), "
        answer += f"Net Margin: {net_margin:.1f}%, P/E: {pe_ratio:.1f}."
        
        # Different recommendations for holders vs new investors
        recommendations = []
        if decision == "BUY":
            if current_price > 0:
                recommendations = [
                    f"New investors: Consider entry {price_info}, start with partial position",
                    f"Current holders: Add on any dips below ${current_price * 0.95:.2f}"
                ]
            else:
                recommendations = [
                    "New investors: Consider starting with partial position",
                    "Current holders: Consider adding to position"
                ]
        elif decision == "HOLD":
            recommendations = [
                "Current holders: Maintain position, monitor quarterly earnings",
                "New investors: Wait for better entry point or clearer trend"
            ]
        elif decision == "SELL":
            recommendations = [
                "Current holders: Consider reducing position or setting stop-loss",
                "New investors: Avoid - look for better opportunities"
            ]
        else:  # WAIT
            if current_price > 0:
                recommendations = [
                    "Monitor next earnings report for margin improvements",
                    f"Set price alert at ${current_price * 0.85:.2f} for potential entry"
                ]
            else:
                recommendations = [
                    "Monitor next earnings report for margin improvements",
                    "Wait for clearer financial trends"
                ]
        
        return {
            "query_specific_answer": answer,
            "key_strengths": [
                f"Market position in {ticker} sector" if ticker != "Unknown" else "Established business",
                f"Gross margin of {gross_margin:.1f}%" if gross_margin > 30 else "Revenue generation capability"
            ],
            "key_concerns": risk.get("risk_factors", ["Market volatility", "Margin pressure"])[:2],
            "recommendations": recommendations,
            "investment_decision": decision
        }
    
    else:
        # Non-investment queries - keep existing logic but simplified
        return {
            "query_specific_answer": f"{company} shows {risk_level} risk profile with {net_margin:.1f}% net margins.",
            "key_strengths": [f"Gross margin: {gross_margin:.1f}%", "Established market presence"],
            "key_concerns": risk.get("risk_factors", ["Monitor performance"])[:2],
            "recommendations": ["Review detailed financials", "Track quarterly progress"]
        }
def create_insight_agent(llm):
    """Insight generation agent - optimized version"""
    def insight_generator(state: FinancialAnalysisState):
        # Detect query intent
        query_intent = detect_query_intent(state["query"])
        
        # Extract key metrics for the prompt with None handling
        parsed_data = state.get("parsed_data", {})
        kpis = state.get("kpis", {})
        risk_assessment = state.get("risk_assessment", {})
        market_data = state.get("market_data", {})
        
        # Prepare simplified data for prompt - handle None values
        revenue = parsed_data.get('revenue', 0)
        revenue = (revenue / 1e6 if revenue and revenue != 0 else 0)
        
        net_margin = kpis.get('net_margin', 0) or 0
        pe_ratio = kpis.get('price_to_earnings', 0) or 0
        risk_score = risk_assessment.get('overall_risk_score', 0) or 0
        current_price = market_data.get('price', 0) or 0
        
        prompt = INSIGHT_PROMPT.format_messages(
            company=state.get("company", "Unknown"),
            ticker=state.get("ticker", "Unknown"),
            period=state.get("period", "Unknown"),
            query=state.get("query", ""),
            query_intent=query_intent,
            response_mode=state.get("response_mode", "concise"),
            revenue=f"{revenue:.1f}",
            net_margin=f"{net_margin:.1f}",
            pe_ratio=f"{pe_ratio:.1f}",
            risk_score=f"{risk_score:.0f}",
            current_price=f"{current_price:.2f}",
            messages=state.get("messages", [])[-3:]  # Only last 3 messages to save tokens
        )
        
        response = llm.invoke(prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                insights = json.loads(json_match.group())
            else:
                insights = generate_improved_default_insights(state, query_intent, market_data)
        except:
            insights = generate_improved_default_insights(state, query_intent, market_data)
        
        return {
            "messages": [AIMessage(content=f"Insights generated")],  # Shorter message
            "insights": insights,
            "query_intent": query_intent,
            "current_agent": "end"
        }
    
    return insight_generator
def generate_default_insights(state, query_intent=None):
    """Generate default insights based on the data and query intent"""
    company = state["company"]
    ticker = state.get("ticker", "Unknown")
    kpis = state["kpis"]
    risk = state["risk_assessment"]
    response_mode = state.get("response_mode", "concise")
    query = state.get("query", "")
    parsed_data = state.get("parsed_data", {})
    company_confidence = state.get("company_confidence", 0.5)
    
    # If no intent provided, detect it
    if not query_intent:
        query_intent = detect_query_intent(query)
    
    # Format numbers properly
    risk_score = risk.get('overall_risk_score', 0)
    risk_level = risk.get('risk_level', 'Medium')
    gross_margin = kpis.get('gross_margin', 0)
    net_margin = kpis.get('net_margin', 0)
    revenue = parsed_data.get('revenue', 0)
    
    # Add confidence note if low
    confidence_note = ""
    if company_confidence < 0.5:
        confidence_note = f"Note: Company identification confidence is low ({company_confidence:.0%}). Please verify the company before making decisions."
    
    if query_intent == "document_analysis":
        # Focus on document findings, not investment advice
        if response_mode == "concise":
            answer = f"The document shows {company} ({ticker}) with revenue of ${revenue/1e6:.1f}M, gross margin of {gross_margin:.1f}%, and net margin of {net_margin:.1f}%."
            
            return {
                "executive_summary": "",
                "query_specific_answer": answer,
                "key_strengths": [
                    f"Revenue: ${revenue/1e6:.1f}M",
                    f"Gross margin: {gross_margin:.1f}%"
                ],
                "key_concerns": [
                    "Data completeness",
                    "Period alignment"
                ],
                "recommendations": [
                    "Review full financial statements",
                    "Compare with previous periods"
                ],
                "market_position": "",
                "future_outlook": "",
                "confidence_note": confidence_note if confidence_note else None
            }
        else:
            return {
                "executive_summary": f"""## {company} ({ticker}) Earnings Analysis

### Financial Performance
- **Revenue**: ${revenue/1e6:.1f}M
- **Gross Profit Margin**: {gross_margin:.1f}%
- **Net Profit Margin**: {net_margin:.1f}%

### Key Findings
The earnings document reveals {company}'s financial performance with notable metrics showing {'strong' if gross_margin > 40 else 'moderate' if gross_margin > 25 else 'weak'} profitability.

{confidence_note}
                """.strip(),
                "query_specific_answer": f"The earnings analysis reveals {company} ({ticker}) generated ${revenue/1e6:.1f}M in revenue with a gross margin of {gross_margin:.1f}% and net margin of {net_margin:.1f}%. The company maintains {'strong' if gross_margin > 40 else 'moderate'} profitability metrics.",
                "key_strengths": [
                    f"Revenue of ${revenue/1e6:.1f}M",
                    f"Gross margin of {gross_margin:.1f}%",
                    f"Net margin of {net_margin:.1f}%"
                ],
                "key_concerns": [
                    "Need complete balance sheet data",
                    "Period-over-period comparison needed"
                ],
                "recommendations": [
                    "Analyze year-over-year growth trends",
                    "Compare margins with industry averages",
                    "Review cash flow statements"
                ],
                "market_position": f"{company} shows {'strong' if gross_margin > 40 else 'competitive'} market positioning based on margin analysis.",
                "future_outlook": "Further analysis of growth trends and cash flows needed for forward projections.",
                "confidence_note": confidence_note if confidence_note else None
            }
    
    else:
        # Original investment-focused logic for investment queries
        # Determine investment decision
        if risk_score < 40 and gross_margin > 30 and net_margin > 10:
            decision = "BUY"
            reason = "low risk with strong profitability"
        elif risk_score > 70 or net_margin < 5:
            decision = "SELL"
            reason = "high risk or poor profitability"
        elif risk_score < 60 and gross_margin > 20:
            decision = "HOLD"
            reason = "moderate risk with decent returns"
        else:
            decision = "WAIT"
            reason = "mixed signals require more analysis"
        
        # Adjust decision if company confidence is low
        if company_confidence < 0.5 and decision in ["BUY", "SELL"]:
            decision = "WAIT"
            reason = "company identification uncertainty"
        
        # Return investment-focused insights
        if response_mode == "concise":
            if "invest" in query.lower():
                answer = f"{decision} - {reason}. Risk: {risk_level}, Margins: {gross_margin:.0f}%"
            else:
                answer = f"{company} ({ticker}) shows {risk_level} risk. Decision: {decision} - {reason}"
            
            if confidence_note:
                answer += f" {confidence_note}"
                
            return {
                "executive_summary": "",
                "query_specific_answer": answer,
                "key_strengths": [
                    f"Gross margin: {gross_margin:.0f}%",
                    "Established market position"
                ],
                "key_concerns": risk.get("risk_factors", ["Market competition"])[:2],
                "recommendations": [
                    "Monitor quarterly results",
                    "Set stop-loss if buying"
                ],
                "market_position": "Competitive position",
                "future_outlook": f"{decision} - {reason}",
                "investment_decision": decision,
                "confidence_note": confidence_note if confidence_note else None
            }
        
        # Detailed mode investment insights
        debt_to_equity = kpis.get('debt_to_equity', 0)
        current_ratio = kpis.get('current_ratio', 0)
        roe = kpis.get('return_on_equity', 0)
        
        return {
            "executive_summary": f"""## {company} ({ticker}) Financial Analysis

### Financial Health
- **Risk Profile**: {risk_level} (Score: {risk_score:.1f}/100)
- **Profitability**: Gross margin {gross_margin:.1f}%, Net margin {net_margin:.1f}%
- **Leverage**: Debt-to-Equity ratio of {debt_to_equity:.2f}
- **Liquidity**: Current ratio of {current_ratio:.2f}

### Investment Thesis
Based on comprehensive analysis, {company} presents a {risk_level.lower()} risk investment opportunity. The company's financial metrics suggest {reason}.

**Investment Decision**: {decision}

{confidence_note}
            """.strip(),
            "query_specific_answer": f"Based on detailed financial analysis, {company} ({ticker}) is a {decision} recommendation. The company shows {risk_level} risk (score: {risk_score:.1f}/100) with {gross_margin:.1f}% gross margins and {net_margin:.1f}% net margins. Key factors: {reason}. ROE of {roe:.1f}% indicates {'strong' if roe > 15 else 'moderate' if roe > 10 else 'weak'} capital efficiency.",
            "key_strengths": [
                f"Gross margin of {gross_margin:.1f}% indicates pricing power",
                f"Current ratio of {current_ratio:.2f} suggests {'strong' if current_ratio > 1.5 else 'adequate'} liquidity",
                f"Return on equity of {roe:.1f}% shows {'excellent' if roe > 20 else 'good' if roe > 15 else 'moderate'} capital efficiency"
            ],
            "key_concerns": risk.get("risk_factors", ["Market competition", "Economic uncertainty", "Regulatory risks"]),
            "recommendations": [
                f"{'Initiate position' if decision == 'BUY' else 'Maintain current position' if decision == 'HOLD' else 'Consider reducing exposure' if decision == 'SELL' else 'Wait for better entry point'}",
                "Set stop-loss at 8-10% below entry price",
                "Review position quarterly based on earnings reports"
            ],
            "market_position": f"{company} maintains a competitive position in its industry with {'strong' if gross_margin > 40 else 'moderate' if gross_margin > 25 else 'challenged'} pricing power and {'solid' if risk_score < 50 else 'moderate' if risk_score < 70 else 'weak'} financial stability.",
            "competitive_advantages": [
                "Established brand recognition",
                f"{'Superior' if gross_margin > 40 else 'Competitive'} profit margins",
                f"{'Strong' if current_ratio > 2 else 'Adequate'} balance sheet"
            ],
            "future_outlook": f"Forward outlook is {'positive' if decision == 'BUY' else 'neutral' if decision == 'HOLD' else 'cautious' if decision == 'SELL' else 'uncertain'}. Expected performance driven by {reason}. {'Consider accumulating on dips' if decision == 'BUY' else 'Monitor for improvement' if decision == 'HOLD' else 'Look for exit opportunities' if decision == 'SELL' else 'Await clearer signals'}.",
            "technical_analysis": f"P/E ratio: {kpis.get('price_to_earnings', 25):.1f}x, ROA: {kpis.get('return_on_assets', 0):.1f}%, Debt/Equity: {debt_to_equity:.2f}",
            "investment_decision": decision,
            "confidence_note": confidence_note if confidence_note else None
        }

# Create the updated workflow with company discovery
def create_financial_workflow():
    """Create the LangGraph workflow with company discovery"""
    
    # Get available LLMs
    llms = get_llms()
    
    if not llms:
        st.error("No LLMs configured! Please add API keys to your .env file.")
        return None
    
    # Select LLMs for each agent (with fallbacks)
    # Use Gemini 1.5 Flash for company discovery as requested
    discovery_llm = llms.get("gemini-pro", llms.get("gpt-4", llms.get("groq-mixtral", list(llms.values())[0])))
    parser_llm = llms.get("groq-llama3", llms.get("gemini-pro", llms.get("gpt-4", list(llms.values())[0])))
    kpi_llm = llms.get("groq-llama3", llms.get("gemini-pro", llms.get("gpt-4", list(llms.values())[0])))
    risk_llm = llms.get("groq-mixtral", llms.get("gpt-4", llms.get("claude-3", list(llms.values())[0])))
    insight_llm = llms.get("gpt-4", llms.get("groq-mixtral", llms.get("claude-3", list(llms.values())[0])))
    
    # Create the graph
    workflow = StateGraph(FinancialAnalysisState)
    
    # Add nodes
    workflow.add_node("company_discovery", create_company_discovery_agent(discovery_llm))
    workflow.add_node("parser", create_parser_agent(parser_llm))
    workflow.add_node("kpi_extractor", create_kpi_agent(kpi_llm))
    workflow.add_node("risk_assessor", create_risk_agent(risk_llm))
    workflow.add_node("insight_generator", create_insight_agent(insight_llm))
    
    # Add edges
    workflow.set_entry_point("company_discovery")
    workflow.add_edge("company_discovery", "parser")
    workflow.add_edge("parser", "kpi_extractor")
    workflow.add_edge("kpi_extractor", "risk_assessor")
    workflow.add_edge("risk_assessor", "insight_generator")
    workflow.add_edge("insight_generator", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app, llms


# Fixed render function with better styling
def render_formatted_chat_message(message_type: str, content: str, timestamp: str = None, message_data: Dict = None):
    """Render chat message with improved formatting"""
    
    # Clean the content first
    cleaned_content = clean_response_text(content)
    
    if message_type == "human":
        # Create a container for the human message
        with st.container():
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                st.markdown("""
                <div style="
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-top: 5px;
                ">👤</div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="
                    background-color: #f7f8fa;
                    padding: 0.75rem 1rem;
                    border-radius: 18px;
                    border-top-left-radius: 4px;
                    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
                    margin-bottom: 0.5rem;
                ">
                    <div style="color: #1a1a1a; line-height: 1.5;">{cleaned_content}</div>
                    {f'<div style="color: #8b8b8b; font-size: 0.75rem; margin-top: 0.25rem;">{timestamp}</div>' if timestamp else ''}
                </div>
                """, unsafe_allow_html=True)
    
    else:  # AI message
        # Create a container for the AI message
        with st.container():
            col1, col2, col3 = st.columns([1, 10, 1])
            with col1:
                st.markdown("""
                <div style="
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-top: 5px;
                ">🤖</div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create a nice container for the AI response
                with st.container():
                    # st.markdown("""
                    # <style>
                    # .ai-message-container {
                    #     background-color: #ffffff;
                    #     padding: 1rem;
                    #     border-radius: 18px;
                    #     border-top-left-radius: 4px;
                    #     box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    #     border: 1px solid #e8e8e8;
                    #     margin-bottom: 0.5rem;
                    # }
                    # </style>
                    # """, unsafe_allow_html=True)
                    
                    # # Use a div with class for the message content
                    # st.markdown(f'<div class="ai-message-container">', unsafe_allow_html=True)
                    
                    # Render the cleaned content using Streamlit's markdown
                    st.markdown(cleaned_content)
                    
                    # Add timestamp if available
                    if timestamp:
                        st.markdown(f'<div style="color: #8b8b8b; font-size: 0.75rem; margin-top: 0.5rem;">{timestamp}</div>', 
                                   unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Add the detailed analysis expander if available
        if message_data and "full_results" in message_data:
            with st.expander("📊 View Detailed Analysis", expanded=False):
                results = message_data["full_results"]
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "⚠️ Risk", "📈 Charts", "📜 Raw Data"])
                
                with tab1:
                    # Financial metrics
                    parsed_data = results.get("parsed_data", {})
                    kpis = results.get("kpis", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        revenue = parsed_data.get('revenue', 0)
                        revenue_val = revenue if revenue is not None else 0
                        st.metric(
                            "Revenue",
                            f"${revenue_val/1e6:.1f}M" if revenue_val != 0 else "N/A"
                        )
                        
                    with col2:
                        net_income = parsed_data.get('net_income', 0)
                        net_income_val = net_income if net_income is not None else 0
                        st.metric(
                            "Net Income",
                            f"${net_income_val/1e6:.1f}M" if net_income_val != 0 else "N/A"
                        )
                        
                    with col3:
                        gross_margin = kpis.get('gross_margin', 0)
                        gross_margin_val = gross_margin if gross_margin is not None else 0
                        st.metric(
                            "Gross Margin",
                            f"{gross_margin_val:.1f}%" if gross_margin_val != 0 else "N/A"
                        )
                        
                    with col4:
                        pe_ratio = kpis.get('price_to_earnings', 0)
                        pe_val = pe_ratio if pe_ratio is not None else 0
                        st.metric(
                            "P/E Ratio",
                            f"{pe_val:.1f}" if pe_val != 0 else "N/A"
                        )
                    
                    # KPI table
                    if kpis:
                        st.subheader("Key Performance Indicators")
                        kpi_rows = []
                        
                        roa = kpis.get('return_on_assets', 0)
                        if roa is not None:
                            kpi_rows.append({"Metric": "Return on Assets", "Value": f"{roa:.1f}%"})
                        
                        roe = kpis.get('return_on_equity', 0)
                        if roe is not None:
                            kpi_rows.append({"Metric": "Return on Equity", "Value": f"{roe:.1f}%"})
                        
                        current_ratio = kpis.get('current_ratio', 0)
                        if current_ratio is not None:
                            kpi_rows.append({"Metric": "Current Ratio", "Value": f"{current_ratio:.2f}"})
                        
                        debt_to_equity = kpis.get('debt_to_equity', 0)
                        if debt_to_equity is not None:
                            kpi_rows.append({"Metric": "Debt to Equity", "Value": f"{debt_to_equity:.2f}"})
                        
                        if kpi_rows:
                            kpi_df = pd.DataFrame(kpi_rows)
                            st.dataframe(kpi_df, use_container_width=True, hide_index=True)
                
                with tab2:
                    # Risk assessment
                    risk_data = results.get("risk_assessment", {})
                    
                    if risk_data:
                        # Risk gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = risk_data.get('overall_risk_score', 0),
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Overall Risk Score"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 40], 'color': "lightgreen"},
                                    {'range': [40, 70], 'color': "yellow"},
                                    {'range': [70, 100], 'color': "lightcoral"}
                                ]
                            }
                        ))
                        fig.update_layout(height=300)
                        gauge_key = f"risk_gauge_{message_data.get('timestamp', '').replace(':', '').replace(' ', '')}"
                        st.plotly_chart(fig, use_container_width=True, key=gauge_key)
                        
                        # Risk factors
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("⚠️ Risk Factors")
                            for factor in risk_data.get('risk_factors', []):
                                st.warning(f"• {factor}")
                        
                        with col2:
                            st.subheader("✅ Mitigation Strategies")
                            for strategy in risk_data.get('mitigation_strategies', []):
                                st.success(f"• {strategy}")
                
                with tab3:
                    # Visualizations
                    if parsed_data and kpis:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Margin comparison
                            margins = {
                                'Gross Margin': kpis.get('gross_margin', 0),
                                'Net Margin': kpis.get('net_margin', 0),
                                'EBITDA Margin': kpis.get('ebitda_margin', 0)
                            }
                            valid_margins = {k: v for k, v in margins.items() if v is not None}
                            
                            if valid_margins:
                                margin_data = {
                                    'Metric': list(valid_margins.keys()),
                                    'Percentage': list(valid_margins.values())
                                }
                                fig_margins = px.bar(
                                    margin_data,
                                    x='Metric',
                                    y='Percentage',
                                    title="Profitability Margins",
                                    color='Percentage',
                                    color_continuous_scale='RdYlGn'
                                )
                                chart_key = f"margins_{message_data.get('timestamp', '').replace(':', '').replace(' ', '')}"
                                st.plotly_chart(fig_margins, use_container_width=True, key=chart_key)
                        
                        with col2:
                            # Financial health pie
                            assets = parsed_data.get('total_assets', 0)
                            liabilities = parsed_data.get('total_liabilities', 0)
                            
                            if assets and liabilities and assets > 0 and liabilities > 0:
                                health_metrics = {
                                    'Assets': assets,
                                    'Liabilities': liabilities
                                }
                                fig_health = px.pie(
                                    values=list(health_metrics.values()),
                                    names=list(health_metrics.keys()),
                                    title="Asset vs Liability Distribution"
                                )
                                chart_key = f"health_{message_data.get('timestamp', '').replace(':', '').replace(' ', '')}"
                                st.plotly_chart(fig_health, use_container_width=True, key=chart_key)
                
                with tab4:
                    # Raw data
                    st.json({
                        "company": results.get("company", "Unknown"),
                        "ticker": results.get("ticker", "Unknown"),
                        "company_confidence": results.get("company_confidence", 0),
                        "parsed_data": results.get("parsed_data", {}),
                        "kpis": results.get("kpis", {}),
                        "risk_assessment": results.get("risk_assessment", {}),
                        "data_sources": results.get("data_sources", [])
                    })
def get_default_period():
    """Get the most recent likely available financial period"""
    current_date = datetime.now()
    current_quarter = (current_date.month - 1) // 3 + 1
    current_year = current_date.year
    
    # Financial data is typically delayed by 1-2 quarters
    # So in June 2025 (Q2), the latest available data is likely Q1 2025 or Q4 2024
    if current_quarter == 1:
        # In Q1, latest data is usually Q3 or Q4 of previous year
        return f"Q4 {current_year - 1}"
    elif current_quarter == 2:
        # In Q2, latest data is usually Q4 previous year or Q1 current year
        return f"Q1 {current_year}"
    else:
        # In Q3/Q4, latest data is usually 1-2 quarters behind
        return f"Q{current_quarter - 2} {current_year}"
# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Financial Analyst Chat",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "yahoo_cache" not in st.session_state:
        st.session_state.yahoo_cache = {}
    if "last_yahoo_request" not in st.session_state:
        st.session_state.last_yahoo_request = 0
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "current_company" not in st.session_state:
        st.session_state.current_company = None
    if "current_period" not in st.session_state:
        st.session_state.current_period = None
    
    # Custom CSS for chat interface
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    /* Chat interface styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .quick-action-btn {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .quick-action-btn:hover {
        background-color: #bbdefb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("💬 AI Financial Analyst Chat")
    st.markdown("Ask me about ANY company's financials, market analysis, or investment decisions!")
    
    # Check for API keys
    api_keys_configured = bool(
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("GROQ_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )
    
    if not api_keys_configured:
        st.error("""
        ⚠️ No API keys found! Please create a `.env` file with at least one of:
        - OPENAI_API_KEY
        - GROQ_API_KEY
        - GOOGLE_API_KEY
        - ANTHROPIC_API_KEY
        """)
        st.stop()
    
    # Initialize workflow
    workflow, llms = create_financial_workflow()

    if st.session_state.get("show_llm_evaluation", False):
        # Add back button
        if st.button("← Back to Chat"):
            st.session_state.show_llm_evaluation = False
            st.rerun()
        
        # Show evaluation page
        show_llm_evaluation_page(llms)
        st.stop()  # Stop here, don't show chat interface
    
    if not workflow:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Response mode selector
        st.subheader("📊 Response Detail Level")
        response_mode = st.radio(
            "Choose your expertise level:",
            ["concise", "detailed"],
            format_func=lambda x: "🎯 Concise (Beginner-friendly)" if x == "concise" else "📈 Detailed (Professional)",
            help="Concise mode provides simplified insights perfect for beginners. Detailed mode includes comprehensive analysis for professionals."
        )
        
        # Document upload
        st.subheader("📄 Upload Financial Document")
        uploaded_file = st.file_uploader(
            "Upload PDF (10-K, 10-Q, etc.)",
            type=['pdf'],
            help="Upload financial reports for more accurate analysis"
        )
        
        # Data source selection
        st.subheader("📊 Data Sources")
        use_yahoo = st.checkbox("Yahoo Finance API", value=True)
        use_sec = st.checkbox("SEC EDGAR API", value=True)
        
        # Show current context
        if st.session_state.current_company:
            st.subheader("📍 Current Context")
            st.info(f"**Company:** {st.session_state.current_company}")
            if st.session_state.current_period:
                st.info(f"**Period:** {st.session_state.current_period}")
        

        # LLM Status
        with st.expander("🤖 Active LLMs"):
            for llm_name, llm_instance in llms.items():
                st.success(f"✅ {llm_name}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_company = None
            st.session_state.current_period = None
            st.rerun()

        # In the sidebar section, add this:
        st.divider()
        st.subheader("🔬 Advanced Tools")

        if st.button("📊 LLM Evaluation Dashboard", use_container_width=True):
            st.session_state.show_llm_evaluation = True
            st.rerun()
    # Main chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["type"] == "human":
                render_formatted_chat_message("human", message["content"], message.get("timestamp"))
            else:
                # Pass the entire message data to render function
                render_formatted_chat_message("ai", message["content"], message.get("timestamp"), message)
    
    # Chat input with example queries
    st.markdown("### 💭 Your Question")
    
    # Example queries for inspiration
    example_queries = [
        "Should I invest in Tesla?",
        "Analyze Palantir's financial health",
        "Is ServiceNow overvalued?",
        "Compare AMD vs Intel financials",
        "What's the risk of investing in Spotify?",
        "Show me Snowflake's Q3 2024 earnings"
    ]
    
    col1, col2, col3 = st.columns(3)
    for i, example in enumerate(example_queries):
        with [col1, col2, col3][i % 3]:
            if st.button(f"💡 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.prefilled_query = example
    
    query = st.text_area(
        "Type your question here...",
        placeholder="Ask about any company! e.g., Should I invest in NVIDIA? What's Apple's debt situation? How risky is Tesla stock?",
        height=100,
        key="chat_input",
        value=st.session_state.get("prefilled_query", "")
    )
    
    # Clear prefilled query after use
    if "prefilled_query" in st.session_state:
        del st.session_state.prefilled_query
    
    col1, col2 = st.columns([4, 1])
    with col1:
        analyze_button = st.button("🔍 Send", type="primary", use_container_width=True)
    with col2:
        if st.button("💡 Help", use_container_width=True):
            st.info("""
            **Tips for asking questions:**
            - I can analyze ANY publicly traded company!
            - Just mention the company name or ticker
            - Specify time periods if needed
            - Ask specific questions for better answers
            
            **Examples:**
            - "Is Palantir a good investment?"
            - "Analyze Shopify's Q2 2024 earnings"
            - "Compare Uber and Lyft financial health"
            - "What's the risk of investing in Coinbase?"
            """)
    
    # Process query
    if analyze_button and query:
        # Add user message to chat
        st.session_state.chat_history.append({
            "type": "human",
            "content": query,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        
        # Extract company and period from query
        detected_company, detected_period = extract_query_entities(query)
        
        # Update context if new company/period detected
        if detected_company:
            st.session_state.current_company = detected_company
        if detected_period:
            st.session_state.current_period = detected_period
        
        # For period, if not detected and query implies "now" or "current", use most recent quarter
        # Updated logic:
        if not detected_period:
            query_lower = query.lower()
            
            # Check for temporal indicators
            if any(word in query_lower for word in ["now", "today", "current", "latest", "recent"]):
                period = get_default_period()
            # Check for investment-related queries (assume they want current data)
            elif any(word in query_lower for word in ["invest", "buy", "sell", "risk", "worth", "should i"]):
                period = get_default_period()
            else:
                # Use session state or default to most recent likely period
                period = st.session_state.current_period or get_default_period()
        else:
            period = detected_period
        
        with st.spinner("🤔 Thinking..."):
            # Extract text from uploaded PDF if available
            uploaded_content = ""
            pdf_chunks_info = ""
            if uploaded_file:
                try:
                    with st.spinner(f"📄 Processing PDF with intelligent extraction..."):
                        pdf_result = process_pdf_with_rag(uploaded_file, query, detected_company or "", period)
                        
                        if pdf_result["success"]:
                            uploaded_content = pdf_result["content"]
                            pdf_chunks_info = f"(Analyzed {pdf_result['chunks_used']} relevant sections from {pdf_result['num_chunks']} total)"
                            st.success(f"✅ PDF processed successfully {pdf_chunks_info}")
                        else:
                            st.warning(f"⚠️ Could not process PDF: {pdf_result.get('error', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    uploaded_content = ""
            
            # Create initial state with extracted company for discovery
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "extracted_company": detected_company or "",  # Raw extracted company
                "company": "",  # Will be resolved by discovery agent
                "ticker": "",  # Will be resolved by discovery agent
                "cik": "",  # Will be resolved by discovery agent
                "period": period,
                "parsed_data": {},
                "kpis": {},
                "risk_assessment": {},
                "insights": {},
                "current_agent": "company_discovery",
                "uploaded_content": uploaded_content,
                "market_data": {},
                "data_sources": [],
                "period_validation": {
                    "requested": period,
                    "actual": "Unknown",
                    "matches": False
                },
                "response_mode": response_mode,
                "company_confidence": 0.0
            }
            
            # Run the workflow
            try:
                with st.spinner("🔍 Discovering company information..."):
                    final_state = workflow.invoke(initial_state)
                
                # Get discovered company info
                company = final_state.get("company", "Unknown Company")
                ticker = final_state.get("ticker", "Unknown")
                cik = final_state.get("cik", "")
                company_confidence = final_state.get("company_confidence", 0.0)
                
                # Update session state
                st.session_state.current_company = company
                st.session_state.analysis_results = final_state
                
                # Show discovery results
                if company_confidence < 0.5:
                    st.warning(f"⚠️ Company identification confidence is low ({company_confidence:.0%}). Found: {company} ({ticker})")
                else:
                    st.success(f"✅ Identified: {company} ({ticker})")
                
                # Fetch market data for discovered company
                market_data = {}
                data_sources = []
                
                if use_yahoo and ticker and ticker != "Unknown":
                    with st.spinner(f"📈 Fetching market data for {company} ({ticker})..."):
                        yahoo_data = fetch_yahoo_finance_data(ticker, period)
                        if yahoo_data and len(yahoo_data) > 3:
                            market_data.update(yahoo_data)
                            data_sources.append({
                                "source": "Yahoo Finance",
                                "period": yahoo_data.get("data_period", "Unknown"),
                                "match": yahoo_data.get("period_match", False)
                            })
                
                if use_sec and cik:
                    with st.spinner(f"📑 Fetching SEC filings for {company}..."):
                        sec_data = fetch_sec_edgar_data(company, period, cik)
                        if sec_data and len(sec_data) > 3 and "error" not in sec_data:
                            for key, value in sec_data.items():
                                if key not in ["requested_period", "data_period", "period_match"] and value is not None:
                                    market_data[key] = value
                            
                            data_sources.append({
                                "source": "SEC EDGAR",
                                "period": sec_data.get("data_period", "Unknown"),
                                "match": sec_data.get("period_match", False),
                                "form": sec_data.get("form_type", "")
                            })
                
                # Update state with market data
                final_state["market_data"] = market_data
                final_state["data_sources"] = data_sources
                
                # Re-run analysis with market data
                if market_data:
                    with st.spinner(f"📊 Analyzing {company} financial data..."):
                        # Start from parser agent with updated data
                        updated_state = {
                            **final_state,
                            "current_agent": "parser",
                            "market_data": market_data,
                            "data_sources": data_sources,
                            "period_validation": {
                                "requested": period,
                                "actual": market_data.get("data_period", "Unknown"),
                                "matches": market_data.get("period_match", False)
                            }
                        }
                        
                        # Continue workflow from parser
                        final_state = workflow.invoke(updated_state)
                
                                # Generate response based on mode
                insights = final_state.get("insights", {})
                kpis = final_state.get("kpis", {})
                parsed_data = final_state.get("parsed_data", {})
                risk_assessment = final_state.get("risk_assessment", {})

                # Build response based on mode and intent
                query_intent = final_state.get("query_intent", detect_query_intent(query))

                # Use the new formatting function
                ai_response = format_response_for_display(
                    response_parts=[],  # Not needed with new formatter
                    response_mode=response_mode,
                    query_intent=query_intent,
                    company=company,
                    ticker=ticker,
                    insights=insights,
                    kpis=kpis,
                    risk_assessment=risk_assessment,
                    market_data=market_data,
                    parsed_data=parsed_data
                )

                # Add data source info to the response
                if data_sources or pdf_chunks_info:
                    data_source_parts = ["\n\n📊 *Data sources used:*"]
                    for source in data_sources:
                        data_source_parts.append(f"• {source['source']} - {source['period']}")
                    if pdf_chunks_info:
                        data_source_parts.append(f"• Uploaded PDF {pdf_chunks_info}")
                    
                    ai_response += "\n".join(data_source_parts)
                
                # Add AI response to chat with full results embedded
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": ai_response,
                    "timestamp": datetime.now().strftime("%I:%M %p"),
                    "full_results": final_state,  # Store full results for detailed view
                    "company": company,
                    "ticker": ticker,
                    "period": period
                })
                
                # Rerun to update chat display
                st.rerun()
                
            except Exception as e:
                st.error(f"Debug Error: {str(e)}")
                error_message = f"I encountered an error while analyzing your query. Please try again or rephrase your question.\n\nError details: {str(e)[:200]}..."
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": error_message,
                    "timestamp": datetime.now().strftime("%I:%M %p")
                })
                st.rerun()
    
    # Footer with tips
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**💡 Pro Tips:**")
        st.caption("• I can analyze ANY public company")
        st.caption("• Use company names or tickers")
        st.caption("• Ask for specific metrics")
    
    with col2:
        st.markdown("**📊 Popular Topics:**")
        st.caption("• Investment decisions")
        st.caption("• Risk assessment")
        st.caption("• Financial health checks")
    
    with col3:
        st.markdown("**🔍 Example Companies:**")
        st.caption("• Tech giants (AAPL, MSFT)")
        st.caption("• Emerging tech (PLTR, SNOW)")
        st.caption("• Any ticker you want!")

if __name__ == "__main__":
    main()
