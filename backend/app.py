# ============================================
# MACOS MULTIPROCESSING FIX (MUST BE FIRST)
# ============================================
import multiprocessing
import os

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# ============================================
# IMPORTS
# ============================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import warnings
import time
import re

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============================================
# MODEL LOADING (GLOBAL - LOAD ONCE)
# ============================================
print("=" * 50)
print("LOADING MODELS (This takes 1-2 minutes on first run)...")
print("=" * 50)
start_time = time.time()

# Embedding model (fast)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# LLM for summary only (not classification - for speed)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Limit threads for speed
torch.set_num_threads(4)

load_time = round(time.time() - start_time, 2)
print(f"✅ Models loaded in {load_time} seconds")
print("=" * 50)

# ============================================
# REFERENCE CORPUS (Pre-computed embeddings)
# ============================================
ETHICAL_REFERENCE_CLAUSES = [
    # Data Deletion & Control
    "You have the right to delete your personal data at any time.",
    "You can request permanent deletion of your account and all associated data.",
    "Users can delete their data without penalty or fee.",
    "We will securely erase your data within 30 days of request.",
    # Third-party Restrictions
    "We will never sell your personal information to third parties.",
    "Your data will never be shared with third-party advertisers.",
    "We do not sell data to any external companies or partners.",
    "Third parties have no access to your personal information.",
    # Opt-out & Consent
    "You can opt-out of data collection with a single click.",
    "Explicit consent is required before any data sharing.",
    "You can withdraw your consent at any time without consequence.",
    "Opt-in (not opt-out) is required for marketing communications.",
    "Users must affirmatively consent before tracking begins.",
    # Encryption & Security
    "Your data is encrypted and stored securely.",
    "We use industry-standard encryption for all data.",
    "Data is protected with end-to-end encryption.",
    "Security audits are conducted regularly by third parties.",
    # Transparency
    "We provide transparent information about data usage.",
    "We clearly disclose all data collection practices.",
    "You receive detailed reports of how your data is used.",
    "We explain our data practices in plain language.",
    # Data Ownership
    "You retain full ownership of your content.",
    "You can export all your personal data at any time.",
    "Your content remains yours unless explicitly transferred.",
    "Data portability is available in standard formats.",
    # Cross-site Tracking
    "We do not track your activity across other websites.",
    "We respect Do Not Track signals from your browser.",
    "We do not use cookies to monitor browsing on other sites.",
    "Cross-site tracking is disabled by default.",
    # Data Access
    "You can request a copy of all data we hold about you.",
    "Data access requests are fulfilled within 30 days.",
    "You receive a complete copy of your data in machine-readable format.",
    "All information about you is made available upon request.",
    # Data Minimization
    "We only collect data necessary for service provision.",
    "Data collection is limited to specified purposes.",
    "We minimize personal information collection by default.",
    "Unnecessary or excessive data collection is prohibited.",
    # Consent Management
    "Your consent is required before any data sharing.",
    "Consent decisions can be modified or revoked anytime.",
    "Granular consent controls allow selective data sharing.",
    "Default settings prioritize user privacy.",
    # Retention Policies
    "We delete data regularly and don't retain it longer than needed.",
    "Data retention periods are clearly specified.",
    "Automatic deletion occurs after the stated retention period.",
    "Historical data is purged according to retention schedule.",
    # Additional Protective Clauses
    "GDPR compliance is guaranteed for all users.",
    "We follow strict data processing agreements with partners.",
    "Children's data receives extra protective measures.",
    "Changes to privacy policy require explicit user consent."
]

NON_ETHICAL_REFERENCE_CLAUSES = [
    # Data Sales
    "We may sell your data to third-party partners.",
    "Your personal information may be sold to advertisers.",
    "We monetize user data by selling to brokers.",
    "Your data is a commodity we trade with other companies.",
    # Tracking & Monitoring
    "We track your browsing activity across all websites.",
    "We use cookies to monitor all your online behavior.",
    "Continuous tracking is enabled by default.",
    "Your location is tracked constantly for any purpose.",
    # Legal Waiver
    "You waive your right to legal action against us.",
    "Users cannot sue us for any data violation.",
    "We are exempt from liability for privacy breaches.",
    "You forfeit the right to join class action lawsuits.",
    # Liability Disclaimer
    "We are not liable for any data breaches.",
    "Unauthorized access to your data is not our responsibility.",
    "We cannot be held responsible for third-party leaks.",
    "Data loss is not covered by any warranty or insurance.",
    # Unrestricted Data Transfer
    "Your data may be transferred to any country worldwide.",
    "Data can be moved to countries with weak privacy laws.",
    "We may relocate your information without notice.",
    "International data transfers happen without restriction.",
    # Unilateral Changes
    "We reserve the right to change terms without notice.",
    "Terms of service can be altered at any time unilaterally.",
    "Privacy policy modifications take effect immediately.",
    "Users have no advance notice of policy changes.",
    # Implicit Consent
    "We collect data from your device without explicit consent.",
    "Continued use implies consent to all data practices.",
    "Silence equals agreement to our data collection.",
    "Data collection begins upon account creation without confirmation.",
    # Third-party Sharing
    "Your information may be shared with affiliated companies.",
    "We sell data to thousands of partner organizations.",
    "Sharing occurs without your knowledge or ability to opt out.",
    "Partners can use your data for their own marketing.",
    # Indefinite Retention
    "We retain your data indefinitely even after account deletion.",
    "Deleted data is kept in backup systems forever.",
    "We maintain historical records of all your activity indefinitely.",
    "Data purges never occur regardless of your request.",
    # Inadequate Security
    "We provide no encryption or security measures.",
    "Your data is stored in plain text with minimal protection.",
    "We do not conduct security audits or testing.",
    "Breaches are our customers' responsibility to detect.",
    # Additional Restrictive Clauses
    "You cannot access or download your data.",
    "GDPR and privacy laws do not apply to our service.",
    "We discriminate based on consent to data sharing.",
    "Privacy choices are not reversible once made."
]

# Pre-compute embeddings ONCE at startup
ethical_embeddings = semantic_model.encode(ETHICAL_REFERENCE_CLAUSES, convert_to_tensor=True)
non_ethical_embeddings = semantic_model.encode(NON_ETHICAL_REFERENCE_CLAUSES, convert_to_tensor=True)

# ============================================
# KEYWORD-BASED SCORING (SUPPLEMENTARY ONLY)
# ============================================
ETHICAL_KEYWORDS = {
    # Only high-confidence ethical-only indicators
    'delete your data': 2.0, 'never sell': 2.0, 'explicit consent': 2.0, 
    'opt-out': 1.5, 'encrypted': 1.5, 'secure': 1.0, 'transparent': 1.5,
    'gdpr complian': 2.0, 'ccpa complian': 2.0
}

NON_ETHICAL_KEYWORDS = {
    # Only high-confidence non-ethical-only indicators
    'sell your data': 2.0, 'track your': 1.5, 'waive': 2.0, 
    'no liability': 2.0, 'not liable': 2.0, 'indefnitely': 2.0,
    'must accept': 1.0, 'irrevocable': 1.5
}

def get_keyword_boost(text):
    """
    Get keyword-based confidence boost (supplementary only).
    Returns: (ethical_boost, non_ethical_boost, explanation)
    """
    text_lower = text.lower()
    ethical_boost = 0.0
    non_ethical_boost = 0.0
    matched_ethical = []
    matched_non_ethical = []
    
    for keyword, weight in ETHICAL_KEYWORDS.items():
        if keyword in text_lower:
            ethical_boost += weight * 0.01  # Very small influence (1% per keyword)
            matched_ethical.append(keyword)
    
    for keyword, weight in NON_ETHICAL_KEYWORDS.items():
        if keyword in text_lower:
            non_ethical_boost += weight * 0.01
            matched_non_ethical.append(keyword)
    
    explanation = ""
    if matched_ethical:
        explanation += f"Ethical signals: {', '.join(matched_ethical[:2])}"
    if matched_non_ethical:
        if explanation:
            explanation += "; "
        explanation += f"Non-ethical signals: {', '.join(matched_non_ethical[:2])}"
    
    return ethical_boost, non_ethical_boost, explanation

def detect_critical_negations(text):
    """
    Detect critical negations that modify meaning (conservative approach).
    Only returns True if negation modifies a critical ethical/non-ethical word.
    """
    text_lower = text.lower()
    critical_negations = [
        'not liable', 'not responsible', 'does not track', 'do not sell',
        'never track', 'never sell', 'no tracking', 'no liability'
    ]
    
    for nego in critical_negations:
        if nego in text_lower:
            return True
    
    return False

# ============================================
# IMPROVED CLASSIFICATION (Semantic-Primary)
# ============================================
def classify_clause_fast(sentence):
    """
    Semantic similarity is PRIMARY (0-1 confidence).
    Keywords and negations provide SMALL adjustments only.
    Returns: (label, confidence, explanation) 
    """
    # 1. PRIMARY: Semantic Similarity Scoring
    sentence_embedding = semantic_model.encode(sentence, convert_to_tensor=True)
    ethical_scores = util.cos_sim(sentence_embedding, ethical_embeddings)[0]
    max_ethical_score = torch.max(ethical_scores).item()
    
    non_ethical_scores = util.cos_sim(sentence_embedding, non_ethical_embeddings)[0]
    max_non_ethical_score = torch.max(non_ethical_scores).item()
    
    # Base decision from semantic similarity
    semantic_ethical_score = max_ethical_score
    semantic_non_ethical_score = max_non_ethical_score
    
    # 2. SUPPLEMENTARY: Keyword boost (very small influence)
    keyword_ethical_boost, keyword_non_ethical_boost, keyword_explanation = get_keyword_boost(sentence)
    
    final_ethical_score = semantic_ethical_score + keyword_ethical_boost
    final_non_ethical_score = semantic_non_ethical_score + keyword_non_ethical_boost
    
    # 3. SUPPLEMENTARY: Negation detection (very small influence)
    has_negation = detect_critical_negations(sentence)
    if has_negation:
        # Negations usually make unethical clauses more ethical
        final_ethical_score += 0.02
    
    # 4. Confidence Thresholding (minimum 0.5 for high confidence, 0.35 for medium)
    max_score = max(final_ethical_score, final_non_ethical_score)
    score_difference = abs(final_ethical_score - final_non_ethical_score)
    
    # Need both high score AND good differentiation
    if max_score < 0.35 or score_difference < 0.05:
        return 'unclear', round(max_score, 2), "Low confidence or unclear classification"
    
    # 5. Determine classification with explanation
    explanation = ""
    if keyword_explanation:
        explanation = keyword_explanation
    if has_negation:
        if explanation:
            explanation += " "
        explanation += "[Negation detected]"
    
    # If no explanation from keywords, provide a simple message
    if not explanation:
        explanation = "Classified based on semantic analysis"
    
    if final_ethical_score > final_non_ethical_score:
        return 'ethical', round(final_ethical_score, 2), explanation
    else:
        return 'non_ethical', round(final_non_ethical_score, 2), explanation

# ============================================
# MAIN ENDPOINT (OPTIMIZED FOR SPEED)
# ============================================
@app.route('/process', methods=['POST'])
def process_text():
    start_time = time.time()
    data = request.json
    text = data.get('text', '')
    incoming_links = data.get('links', [])  # Links from extension
    
    if not text or len(text) < 50:
        return jsonify({
            "ethical_points": [],
            "non_ethical_points": [],
            "unclear_points": [],
            "risk_level": "Unknown",
            "summary": "Please navigate to a page with more content.",
            "processing_time_seconds": 0,
            "highlighted_indices": [],
            "total_text_analyzed_chars": 0,
            "clauses_analyzed": 0
        }), 200
    
    original_text = text  # Keep original for highlighting
    
    # 1. Process ENTIRE text (up to 10,000,000 chars - NO VECTOR SEARCH BOTTLENECK)
    text = text[:10000000]
    
    # 2. Extract ALL sentences from the entire document with position tracking
    # Split by multiple sentence delimiters
    sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+|[.!?]')
    
    sentences_with_pos = []
    current_pos = 0
    
    # More sophisticated sentence splitting
    for match in sentence_pattern.finditer(text):
        if match.group() and match.group().strip():
            sentence_end = match.start()
            sentence_text = text[current_pos:sentence_end].strip()
            
            if len(sentence_text) > 10:  # Only keep meaningful sentences
                sentences_with_pos.append({
                    'text': sentence_text,
                    'start': current_pos,
                    'end': sentence_end
                })
            current_pos = match.end()
    
    # Handle last sentence if any
    if current_pos < len(text):
        sentence_text = text[current_pos:].strip()
        if len(sentence_text) > 10:
            sentences_with_pos.append({
                'text': sentence_text,
                'start': current_pos,
                'end': len(text)
            })
    
    # 3. Generate summary from first part of document
    summary_text = text[:5000]  # Use beginning of document for context
    summary_prompt = f"""Analyze these terms and identify the main privacy risks and protections in one sentence:
{summary_text}

Summary:"""
    inputs = tokenizer(summary_prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=60, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 4. Classify ALL sentences (no hard limit - process everything)
    ethical_points = []
    non_ethical_points = []
    unclear_points = []
    highlighted_indices = []
    
    for sent_data in sentences_with_pos:
        sentence_text = sent_data['text']
        label, confidence, explanation = classify_clause_fast(sentence_text)
        
        point_data = {
            "text": sentence_text,
            "confidence": confidence,
            "reasoning": explanation
        }
        
        # Track position for highlighting (use sentence text for more reliable highlighting)
        highlight_info = {
            "text": sentence_text,  # Use actual sentence text instead of positions
            "type": label
        }
        
        if label == 'ethical':
            ethical_points.append(point_data)
            highlighted_indices.append(highlight_info)
        elif label == 'non_ethical':
            non_ethical_points.append(point_data)
            highlighted_indices.append(highlight_info)
        else:  # unclear
            unclear_points.append(point_data)
            # Optional: uncomment to highlight unclear as well
            # highlighted_indices.append(highlight_info)
    
    # Sort by confidence (don't truncate - return all for transparency)
    ethical_points.sort(key=lambda x: x['confidence'], reverse=True)
    non_ethical_points.sort(key=lambda x: x['confidence'], reverse=True)
    unclear_points.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 6. Calculate Risk Level (more conservative)
    total_classified = len(ethical_points) + len(non_ethical_points)
    
    if total_classified == 0:
        risk_level = "Unknown"
    else:
        # Use both count and average confidence
        risk_ratio = len(non_ethical_points) / total_classified
        avg_non_ethical_conf = sum([p['confidence'] for p in non_ethical_points]) / len(non_ethical_points) if non_ethical_points else 0
        
        # Higher bar for "High" risk: need strong non-ethical signals
        if (risk_ratio >= 0.5 and avg_non_ethical_conf > 0.65) or avg_non_ethical_conf > 0.75:
            risk_level = "High"
        elif risk_ratio >= 0.35 or avg_non_ethical_conf > 0.60:
            risk_level = "Medium"
        else:
            risk_level = "Low"
    
    processing_time = round(time.time() - start_time, 2)
    
    # Extract URLs from the text and combine with extension-provided links
    url_pattern = r'https?://[^\s\)]+'
    found_urls = re.findall(url_pattern, text)
    all_links = list(set(found_urls))  # Remove duplicates
    
    # Add extension-provided links
    for link in incoming_links:
        if link.get('url') and link['url'] not in all_links:
            all_links.append(link['url'])
    
    return jsonify({
        "ethical_points": ethical_points,
        "non_ethical_points": non_ethical_points,
        "unclear_points": unclear_points,
        "risk_level": risk_level,
        "summary": summary,
        "processing_time_seconds": processing_time,
        "clauses_analyzed": len(sentences_with_pos),
        "highlighted_indices": highlighted_indices,
        "total_text_analyzed_chars": len(text),
        "links": all_links,
        "links_count": len(all_links),
        "classification_method": "full_document_analysis_no_vector_search",
        "analysis_coverage": "Entire page analyzed - no chunks skipped"
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "port": 5001,
        "models_loaded": True,
        "optimization": "fast_semantic_classification"
    })

if __name__ == '__main__':
    app.run(port=5001, debug=False, threaded=True)