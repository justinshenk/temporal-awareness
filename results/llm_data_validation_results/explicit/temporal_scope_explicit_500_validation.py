import json
import re
from pathlib import Path

# Load the data
input_path = Path("/Users/avigyapaudel/Documents/AI Safety/AISC/temporal-awareness/data/raw/temporal_scope_explicit_expanded_500.json")
with open(input_path, 'r') as f:
    data = json.load(f)

pairs = data['pairs']

# Temporal keyword lists
IMMEDIATE_KEYWORDS = [
    'now', 'today', 'immediate', 'immediately', 'current', 'currently', 
    'this week', 'this month', 'this quarter', 'this year', 'this sprint',
    'this period', 'this moment', 'right now', 'tonight', 'tomorrow',
    'next week', 'next month', 'near-term', 'near term', 'short-term', 
    'short term', 'quick', 'quickly', 'urgent', 'urgency', 'recent',
    'recently', 'daily', 'weekly', 'hourly', 'hour', 'hours', 'day', 'days',
    'minute', 'minutes', 'instant', 'deadline', 'present', 'active',
    'this afternoon', 'this morning', 'end of day', 'end of week',
    'this session', 'this cycle', 'this phase', 'this project'
]

LONG_TERM_KEYWORDS = [
    'years', 'year', 'decades', 'decade', 'generations', 'generation',
    'centuries', 'century', 'lifetime', 'lifelong', 'long-term', 'long term',
    'far-term', 'far term', 'future', 'perpetual', 'eternal', 'enduring',
    'lasting', 'permanent', 'sustainable', 'multi-year', 'multi-decade',
    'multi-generational', 'multigenerational', 'historical', 'legacy',
    'over time', 'across years', 'across decades', 'across generations',
    'coming years', 'coming decades', 'years ahead', 'decades ahead',
    'career-long', 'career-spanning', 'institution', 'institutional',
    'civilizational', 'evolutionary', 'compound', 'compounding'
]

def has_temporal_keywords(text, keyword_list):
    """Check if text contains any temporal keywords"""
    text_lower = text.lower()
    found = []
    for kw in keyword_list:
        if kw.lower() in text_lower:
            found.append(kw)
    return found

def calculate_length_ratio(text1, text2):
    """Calculate length ratio between two texts"""
    len1 = len(text1.strip())
    len2 = len(text2.strip())
    if max(len1, len2) == 0:
        return 1.0
    return min(len1, len2) / max(len1, len2)

def score_temporal_keywords(immediate_text, long_term_text):
    """Score temporal keyword presence (1-5)"""
    imm_keywords = has_temporal_keywords(immediate_text, IMMEDIATE_KEYWORDS)
    lt_keywords = has_temporal_keywords(long_term_text, LONG_TERM_KEYWORDS)
    
    # Also check for cross-contamination (immediate keywords in long-term, etc.)
    imm_in_lt = has_temporal_keywords(long_term_text, IMMEDIATE_KEYWORDS)
    lt_in_imm = has_temporal_keywords(immediate_text, LONG_TERM_KEYWORDS)
    
    score = 5
    if not imm_keywords:
        score -= 2
    if not lt_keywords:
        score -= 2
    # Minor penalty for some cross-contamination (can be contextually valid)
    if lt_in_imm and not imm_keywords:
        score -= 1
    if imm_in_lt and not lt_keywords:
        score -= 1
        
    return max(1, min(5, score)), imm_keywords, lt_keywords

def score_vocabulary_balance(text1, text2):
    """Score vocabulary complexity balance (1-5)"""
    # Simple heuristic: compare average word length and unique word ratio
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    if not words1 or not words2:
        return 3
    
    avg_len1 = sum(len(w) for w in words1) / len(words1)
    avg_len2 = sum(len(w) for w in words2) / len(words2)
    
    diff = abs(avg_len1 - avg_len2)
    if diff < 0.5:
        return 5
    elif diff < 1.0:
        return 4
    elif diff < 1.5:
        return 3
    elif diff < 2.0:
        return 2
    return 1

def score_ngram_leakage(text1, text2):
    """Score for absence of idiomatic phrases that give away temporal scope (1-5)"""
    # Generally these explicit pairs are designed to have temporal markers
    # Check for overly obvious/cliched phrases
    problematic_phrases = [
        'strike while the iron is hot',
        'rome wasn\'t built in a day',
        'good things come to those who wait',
        'time is money',
        'carpe diem',
        'in the long run'
    ]
    
    combined = (text1 + " " + text2).lower()
    found = [p for p in problematic_phrases if p in combined]
    
    if not found:
        return 5
    elif len(found) == 1:
        return 3
    return 2

def score_length_match(text1, text2):
    """Score length similarity (1-5)"""
    ratio = calculate_length_ratio(text1, text2)
    if ratio >= 0.8:
        return 5
    elif ratio >= 0.6:
        return 4
    elif ratio >= 0.5:
        return 3
    elif ratio >= 0.4:
        return 2
    return 1

def score_structure(text1, text2):
    """Score grammatical parallelism (1-5)"""
    # Check if both start similarly (with article, verb, noun pattern)
    # Simple heuristic: check first word patterns
    words1 = text1.strip().split()
    words2 = text2.strip().split()
    
    if not words1 or not words2:
        return 3
    
    # Remove option labels like (A), (B)
    first1 = re.sub(r'^\([AB]\)\s*', '', text1.strip()).split()[0] if words1 else ''
    first2 = re.sub(r'^\([AB]\)\s*', '', text2.strip()).split()[0] if words2 else ''
    
    # Check if both are noun phrases, verb phrases, etc.
    score = 4  # Default good
    
    # Both start with similar word types (articles, verbs, adjectives)
    articles = ['the', 'a', 'an', 'this', 'that', 'these', 'those']
    if (first1.lower() in articles) == (first2.lower() in articles):
        score = 5
    
    return score

def score_formality(text1, text2):
    """Score formality match (1-5)"""
    # Both options in this dataset appear to use consistent business/professional register
    # Simple check for contractions, slang, etc.
    informal_markers = ["'ll", "'ve", "'re", "'d", "gonna", "wanna", "kinda", "gotta"]
    
    informal1 = any(m in text1.lower() for m in informal_markers)
    informal2 = any(m in text2.lower() for m in informal_markers)
    
    if informal1 == informal2:
        return 5
    return 3

def score_hedging(text1, text2):
    """Score certainty level match (1-5)"""
    hedging_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably', 'likely']
    
    hedge1 = sum(1 for h in hedging_words if h in text1.lower())
    hedge2 = sum(1 for h in hedging_words if h in text2.lower())
    
    if abs(hedge1 - hedge2) == 0:
        return 5
    elif abs(hedge1 - hedge2) == 1:
        return 4
    return 3

def score_specificity(text1, text2):
    """Score abstraction level match (1-5)"""
    # Check for numbers, specific timeframes, concrete nouns
    numbers1 = len(re.findall(r'\d+', text1))
    numbers2 = len(re.findall(r'\d+', text2))
    
    if abs(numbers1 - numbers2) <= 1:
        return 5
    elif abs(numbers1 - numbers2) <= 2:
        return 4
    return 3

def score_sentiment(text1, text2):
    """Score sentiment/valence match (1-5)"""
    # Both options are typically neutral in this dataset
    positive_words = ['best', 'optimal', 'excellent', 'great', 'success', 'win', 'benefit']
    negative_words = ['worst', 'fail', 'loss', 'problem', 'crisis', 'threat', 'risk']
    
    pos1 = sum(1 for w in positive_words if w in text1.lower())
    neg1 = sum(1 for w in negative_words if w in text1.lower())
    pos2 = sum(1 for w in positive_words if w in text2.lower())
    neg2 = sum(1 for w in negative_words if w in text2.lower())
    
    sentiment1 = pos1 - neg1
    sentiment2 = pos2 - neg2
    
    if abs(sentiment1 - sentiment2) == 0:
        return 5
    elif abs(sentiment1 - sentiment2) <= 1:
        return 4
    return 3

def score_clear_temporal(immediate_text, long_term_text):
    """Score clarity of temporal distinction (1-5)"""
    imm_kw = has_temporal_keywords(immediate_text, IMMEDIATE_KEYWORDS)
    lt_kw = has_temporal_keywords(long_term_text, LONG_TERM_KEYWORDS)
    
    # Check that immediate doesn't have long-term keywords dominating
    lt_in_imm = has_temporal_keywords(immediate_text, LONG_TERM_KEYWORDS)
    imm_in_lt = has_temporal_keywords(long_term_text, IMMEDIATE_KEYWORDS)
    
    score = 5
    if not imm_kw or not lt_kw:
        score -= 2
    if len(lt_in_imm) > len(imm_kw):
        score -= 1
    if len(imm_in_lt) > len(lt_kw):
        score -= 1
    
    return max(1, score)

def score_both_valid(question, text1, text2):
    """Score if both options are sensible responses (1-5)"""
    # Heuristic: both should be non-empty and grammatically complete-ish
    if len(text1.strip()) < 10 or len(text2.strip()) < 10:
        return 2
    if len(text1.strip()) < 20 or len(text2.strip()) < 20:
        return 3
    return 5  # Assume valid since dataset is pre-curated

def score_single_dimension(immediate_text, long_term_text):
    """Score if options differ ONLY on temporal scope (1-5)"""
    # This is harder to automate - check for semantic similarity minus temporal words
    # Remove temporal keywords and compare remaining content themes
    
    def remove_temporal(text):
        result = text.lower()
        for kw in IMMEDIATE_KEYWORDS + LONG_TERM_KEYWORDS:
            result = result.replace(kw.lower(), '')
        return result
    
    clean1 = remove_temporal(immediate_text)
    clean2 = remove_temporal(long_term_text)
    
    # Check word overlap after removing temporal markers
    words1 = set(re.findall(r'\b\w{3,}\b', clean1))
    words2 = set(re.findall(r'\b\w{3,}\b', clean2))
    
    if not words1 or not words2:
        return 4
    
    overlap = len(words1 & words2) / max(len(words1), len(words2))
    
    if overlap >= 0.3:
        return 5
    elif overlap >= 0.2:
        return 4
    elif overlap >= 0.1:
        return 3
    return 2

def score_labels_correct(immediate_text, long_term_text):
    """Score if labels are correctly assigned (1-5)"""
    imm_kw = has_temporal_keywords(immediate_text, IMMEDIATE_KEYWORDS)
    lt_kw = has_temporal_keywords(long_term_text, LONG_TERM_KEYWORDS)
    
    # Check for reversed labels
    lt_in_imm = has_temporal_keywords(immediate_text, LONG_TERM_KEYWORDS)
    imm_in_lt = has_temporal_keywords(long_term_text, IMMEDIATE_KEYWORDS)
    
    # If long-term keywords dominate in "immediate" or vice versa, labels may be wrong
    if len(lt_in_imm) > len(imm_kw) + 1 and len(imm_in_lt) > len(lt_kw) + 1:
        return 1  # Likely reversed
    
    if imm_kw and lt_kw:
        return 5
    elif imm_kw or lt_kw:
        return 4
    return 3

def identify_issues(scores, immediate_text, long_term_text):
    """Identify specific issues based on scores"""
    issues = []
    
    if scores['temporal_keywords'] <= 2:
        issues.append("Missing or weak temporal keywords in one or both options")
    if scores['vocabulary_balance'] <= 2:
        issues.append("Significant vocabulary complexity imbalance")
    if scores['length_match'] <= 2:
        issues.append("Options have very different lengths (>50% difference)")
    if scores['structure'] <= 2:
        issues.append("Poor grammatical parallelism between options")
    if scores['clear_temporal'] <= 2:
        issues.append("Temporal distinction is unclear or ambiguous")
    if scores['single_dimension'] <= 2:
        issues.append("Options may differ on dimensions beyond temporal scope")
    if scores['labels_correct'] <= 2:
        issues.append("Labels may be incorrectly assigned (possible reversal)")
    
    return issues

def generate_suggestion(scores, issues):
    """Generate improvement suggestion based on issues"""
    if not issues:
        return "No significant issues identified"
    
    suggestions = []
    if "temporal keywords" in str(issues).lower():
        suggestions.append("Add explicit temporal markers (e.g., 'today', 'this week' for immediate; 'years', 'decades' for long-term)")
    if "length" in str(issues).lower():
        suggestions.append("Balance option lengths to within 50% of each other")
    if "parallel" in str(issues).lower():
        suggestions.append("Restructure options to follow parallel grammatical patterns")
    if "labels" in str(issues).lower():
        suggestions.append("Verify and potentially swap the immediate/long-term labels")
    
    return "; ".join(suggestions) if suggestions else "Review and revise flagged issues"

# Process all pairs
results = []
for pair in pairs:
    pair_id = pair.get('id', 'unknown')
    category = pair.get('category', 'unknown')
    question = pair.get('question', '')
    immediate = pair.get('immediate', '')
    long_term = pair.get('long_term', '')
    
    # Calculate all scores
    temporal_score, imm_kw, lt_kw = score_temporal_keywords(immediate, long_term)
    
    scores = {
        'temporal_keywords': temporal_score,
        'vocabulary_balance': score_vocabulary_balance(immediate, long_term),
        'ngram_leakage': score_ngram_leakage(immediate, long_term),
        'length_match': score_length_match(immediate, long_term),
        'structure': score_structure(immediate, long_term),
        'formality': score_formality(immediate, long_term),
        'hedging': score_hedging(immediate, long_term),
        'specificity': score_specificity(immediate, long_term),
        'sentiment': score_sentiment(immediate, long_term),
        'clear_temporal': score_clear_temporal(immediate, long_term),
        'both_valid': score_both_valid(question, immediate, long_term),
        'single_dimension': score_single_dimension(immediate, long_term),
        'labels_correct': score_labels_correct(immediate, long_term)
    }
    
    average = sum(scores.values()) / len(scores)
    issues = identify_issues(scores, immediate, long_term)
    suggestion = generate_suggestion(scores, issues)
    
    results.append({
        'id': pair_id,
        'category': category,
        'question': question,
        'immediate_option': immediate,
        'long_term_option': long_term,
        'detected_immediate_keywords': imm_kw,
        'detected_long_term_keywords': lt_kw,
        'scores': scores,
        'average': round(average, 2),
        'issues': issues,
        'suggestion': suggestion
    })

# Calculate summary statistics
all_averages = [r['average'] for r in results]
score_distribution = {
    'excellent (4.5-5.0)': len([a for a in all_averages if a >= 4.5]),
    'good (4.0-4.49)': len([a for a in all_averages if 4.0 <= a < 4.5]),
    'acceptable (3.0-3.99)': len([a for a in all_averages if 3.0 <= a < 4.0]),
    'weak (2.0-2.99)': len([a for a in all_averages if 2.0 <= a < 3.0]),
    'poor (<2.0)': len([a for a in all_averages if a < 2.0])
}

category_averages = {}
for r in results:
    cat = r['category']
    if cat not in category_averages:
        category_averages[cat] = []
    category_averages[cat].append(r['average'])

category_summary = {cat: round(sum(avgs)/len(avgs), 2) for cat, avgs in category_averages.items()}

# Factor averages across all pairs
factor_names = list(results[0]['scores'].keys())
factor_averages = {}
for factor in factor_names:
    factor_averages[factor] = round(sum(r['scores'][factor] for r in results) / len(results), 2)

# Common issues
all_issues = []
for r in results:
    all_issues.extend(r['issues'])
from collections import Counter
issue_counts = dict(Counter(all_issues).most_common(10))

# Output structure
output = {
    'metadata': {
        'source_file': 'temporal_scope_explicit_expanded_500.json',
        'total_pairs_validated': len(results),
        'validation_date': '2026-02-03',
        'scoring_scale': '1-5 (1=Poor, 5=Excellent)'
    },
    'summary': {
        'overall_average': round(sum(all_averages) / len(all_averages), 2),
        'score_distribution': score_distribution,
        'category_averages': category_summary,
        'factor_averages': factor_averages,
        'common_issues': issue_counts
    },
    'pair_validations': results
}

# Save output
output_path = Path("/Users/avigyapaudel/Documents/AI Safety/AISC/temporal-awareness/data/validated/temporal_scope_explicit_500_validation_review.json")
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"Validation complete! Results saved to {output_path}")
print(f"\n=== SUMMARY ===")
print(f"Total pairs validated: {len(results)}")
print(f"Overall average score: {output['summary']['overall_average']}")
print(f"\nScore distribution:")
for k, v in score_distribution.items():
    print(f"  {k}: {v} pairs")
print(f"\nFactor averages:")
for k, v in factor_averages.items():
    print(f"  {k}: {v}")
print(f"\nTop issues found:")
for issue, count in list(issue_counts.items())[:5]:
    print(f"  {issue}: {count} pairs")
