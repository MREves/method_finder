"""Metadata, Method Specifications, Mathematical Formulations, and Pedagogical Guides
for the Natural Language Processing (NLP) Module.

Spans Classical Rule-Based NLP, Bag-of-Words/TF-IDF, Supervised Text Classifiers,
Unsupervised Topic Modelling (LDA), Dense Semantic Embeddings, Named Entity Recognition (NER),
and LLM Zero-Shot Classification with Retrieval-Augmented Generation (RAG).

DISCLAIMER: All textual examples, customer feedback logs, IT incident tickets, entity spans,
and policy documents across this module are completely fictional and fabricated solely
for educational and methodological illustration purposes.
"""

from typing import Dict, Any, List

NLP_CLUSTERS: Dict[str, Dict[str, Any]] = {
    "rule_lexicon": {
        "id": "rule_lexicon",
        "name": "Level 1: Rule-Based & Lexicon NLP",
        "icon": "📜",
        "level": 1,
        "description": "Rule-based sentiment valence scoring, negation scope detection, and regular expression text extraction.",
        "methods": ["sentiment_negex"]
    },
    "statistical_vectorization": {
        "id": "statistical_vectorization",
        "name": "Level 2: Statistical Vectorization & Bag-of-Words",
        "icon": "🔤",
        "level": 2,
        "description": "TF-IDF term weighting, n-gram keyword extraction, and supervised document classification using Naive Bayes and Linear SVM.",
        "methods": ["tfidf_keywords", "text_classification_bayes_svm"]
    },
    "topic_discovery": {
        "id": "topic_discovery",
        "name": "Level 3: Unsupervised Thematic Topic Modelling",
        "icon": "🧩",
        "level": 3,
        "description": "Latent Dirichlet Allocation (LDA) for automated discovery of latent themes across unlabelled customer reviews and support tickets.",
        "methods": ["lda_topic_modelling"]
    },
    "dense_vectors": {
        "id": "dense_vectors",
        "name": "Level 4: Dense Semantic Vector Embeddings",
        "icon": "🌌",
        "level": 4,
        "description": "Transformer-based dense vector representations, high-dimensional cosine similarity, and semantic k-nearest neighbors search.",
        "methods": ["dense_embeddings_similarity"]
    },
    "genai_llms": {
        "id": "genai_llms",
        "name": "Level 5: State-of-the-Art Generative AI & LLMs",
        "icon": "🤖",
        "level": 5,
        "description": "Named Entity Recognition (NER), zero-shot classification without model fine-tuning, and Retrieval-Augmented Generation (RAG).",
        "methods": ["clinical_ner", "llm_zeroshot_rag"]
    }
}

NLP_WIZARD_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "nlp_goal",
        "question": "What is the primary objective of your natural language processing task?",
        "options": [
            {
                "value": "sentiment_negation",
                "label": "❤️ Measure sentiment polarity & detect negation phrases (e.g. 'no complaints', 'free of defects')",
                "recommendation": "sentiment_negex"
            },
            {
                "value": "extract_keywords",
                "label": "🔤 Extract key domain terms & compute document similarities using TF-IDF",
                "recommendation": "tfidf_keywords"
            },
            {
                "value": "classify_incidents",
                "label": "🏷️ Automatically categorize support incident tickets into urgency tiers (e.g. Routine vs. Outage)",
                "recommendation": "text_classification_bayes_svm"
            },
            {
                "value": "discover_themes",
                "label": "🧩 Automatically discover unlabelled themes & clusters across large review volumes (LDA)",
                "recommendation": "lda_topic_modelling"
            },
            {
                "value": "semantic_search",
                "label": "🔍 Find conceptually similar comments across different phrasing via Dense Vector Embeddings",
                "recommendation": "dense_embeddings_similarity"
            },
            {
                "value": "extract_domain_entities",
                "label": "🏷️ Extract domain entities (organizations, products, specs, defects) via Named Entity Recognition",
                "recommendation": "clinical_ner"
            },
            {
                "value": "zeroshot_rag",
                "label": "🤖 Ground an LLM with external business policy documents using Retrieval-Augmented Generation (RAG)",
                "recommendation": "llm_zeroshot_rag"
            }
        ]
    }
]

NLP_METHODS: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------
    # Method 1: Lexicon Sentiment & Negation Scope Detection
    # -----------------------------------------------------------------
    "sentiment_negex": {
        "id": "sentiment_negex",
        "name": "Lexicon Sentiment Analysis & Negation Scope Detection",
        "category": "Level 1: Rule-Based & Lexicon NLP",
        "level": 1,
        "icon": "❤️",
        "summary": "Quantifies sentiment polarity in customer comments and user reviews while adjusting for negation scope (e.g. 'no complaints' vs 'complaints') to prevent inverted interpretations.",
        "when_to_use": [
            "Analyzing customer satisfaction open-text comments (CSAT, Net Promoter Score surveys, Trustpilot reviews).",
            "Screening user feedback where negation (e.g. 'no issues observed', 'never experienced downtime') completely flips customer sentiment.",
            "Auditing user review transcripts or customer support chat logs without requiring GPU infrastructure."
        ],
        "key_assumptions": [
            "Lexicon coverage: Words in the text must match the underlying valence dictionary (e.g. VADER, AFINN).",
            "Negation window: Negation phrases (e.g. 'no', 'never', 'without', 'free of') typically govern terms within a 3 to 5 token forward window.",
            "Punctuation and clause boundaries: Negation scopes terminate at sentence boundaries, commas, or contrastive conjunctions ('but', 'however')."
        ],
        "common_pitfalls": [
            "Blind Keyword Counting: Treating 'no complaints' as negative simply because it contains the word 'complaints'.",
            "Double Negation Neglect: Failing to resolve double negations ('not uncommon') causing erroneous polarity flipping.",
            "Sarcasm and Idiomatic Expressions: Overlooking colloquial expressions where literal word valence contradicts intended tone."
        ],
        "plain_english_rules": [
            "Every sentiment-bearing word receives a baseline valence weight between -4.0 (extremely negative) and +4.0 (extremely positive).",
            "Modifiers adjust word intensity: capital letters and booster words ('extremely', 'very') amplify valence (+0.29), while dampeners ('somewhat', 'slightly') attenuate it.",
            "When a negation keyword ('no', 'not', 'never', 'free of') precedes a sentiment term within 4 words, the polarity is inverted ($w_{\text{negated}} = -0.5 \cdot w$).",
            "The normalized compound score ranges from -1.0 (extreme negative) to +1.0 (extreme positive); scores between -0.05 and +0.05 represent neutral text."
        ],
        "formulas": {
            "compound_valence": {
                "title": "VADER Normalized Compound Valence Score",
                "latex": r"c = \frac{\sum_{i=1}^n w_i}{\sqrt{\left(\sum_{i=1}^n w_i\right)^2 + \alpha}}",
                "plain_english": "Sums all modified token valence weights (w_i) and normalizes the aggregate using a smoothing parameter (alpha = 15) to constrain the result strictly between -1.0 and +1.0.",
                "terms": [
                    ("c", "The normalized compound sentiment score, bounded continuously between -1.0 (extreme negative) and +1.0 (extreme positive)."),
                    ("w_i", "The modified valence weight of token i from the sentiment lexicon (e.g. +3.1 for 'superb', -2.8 for 'unresponsive')."),
                    (r"\sum_{i=1}^n w_i", "The raw arithmetic sum of all sentiment-bearing token weights across the customer text."),
                    (r"\alpha", "Normalization smoothing constant (empirically set to alpha = 15.0 in the VADER specification) establishing the threshold where aggregate scores approach the asymptotes."),
                    (r"\sqrt{(\sum w_i)^2 + \alpha}", "The Euclidean normalizing denominator that contracts extreme word sums toward +/-1.0.")
                ],
                "steps": [
                    "1. **Tokenization & Valence Lookup:** Split customer feedback into word tokens and retrieve base valence weights between -4.0 and +4.0 from the sentiment lexicon.",
                    "2. **Intensifier & Modifier Scaling:** For booster words ('extremely', 'very') preceding a word, increase magnitude by +0.293; for dampeners ('slightly'), decrease magnitude.",
                    "3. **Punctuation & Capitalization Amplification:** Add +0.733 if the word is in ALL CAPS and add +0.292 for trailing exclamation points.",
                    "4. **Raw Summation:** Calculate the net sum of all token weights: W = \\sum w_i.",
                    "5. **Asymptotic Normalization:** Divide W by \\sqrt{W^2 + 15} to produce compound score c. Classify as Positive (c >= 0.05), Negative (c <= -0.05), or Neutral (-0.05 < c < 0.05)."
                ],
                "why_used": "Raw word summation suffers from severe document-length bias: a 300-word review with five mild positive words would artificially score +18.0, dwarfing a concise 5-word message with ecstatic praise ('absolutely fantastic support!', +3.5). The hyperbolic normalization curve contracts extreme sums toward +/-1.0, reflecting human cognitive diminishing returns where additional praise has progressively smaller marginal impact."
            },
            "negex_inversion": {
                "title": "Negation Scope Valence Inversion (Chapman's NegEx Rule)",
                "latex": r"w_{\text{negated}} = -0.5 \times w_{\text{original}} \quad \text{for tokens within scope } [t_{\text{neg}}, t_{\text{neg}} + k]",
                "plain_english": "When a negation trigger is detected, subsequent sentiment terms within k tokens (typically k=4) have their valence inverted and halved to reflect negative state without over-exaggeration.",
                "terms": [
                    ("w_{original}", "The baseline dictionary valence weight of the sentiment word before negation."),
                    ("w_{negated}", "The transformed valence weight after applying negation scope rules."),
                    ("t_{neg}", "The word position index of the detected negation trigger word (e.g. 'no', 'never', 'without', 'free of', 'lacks')."),
                    ("k", "The forward scope window size (typically k=4 tokens), defining the boundary of negation influence."),
                    ("-0.5", "The inversion and damping scalar. The negative sign flips valence; the 0.5 factor attenuates intensity.")
                ],
                "steps": [
                    "1. **Trigger Keyword Scanning:** Scan the customer text for negation triggers ('no', 'not', 'never', 'without', 'free of', 'lacks').",
                    "2. **Scope Boundary Delineation:** When a trigger is identified at index t_neg, establish a forward window extending up to 4 words ahead, terminating early at commas, periods, or contrastive conjunctions ('but', 'however').",
                    "3. **Target Word Identification:** Identify any sentiment-bearing words falling within the active scope window.",
                    "4. **Polarity Inversion & Damping:** Multiply the target word's lexicon valence by -0.5. For example, a defect term like 'outage' (w = -3.2) preceded by 'no outage' becomes w_negated = -0.5 * (-3.2) = +1.6."
                ],
                "why_used": "Bag-of-words analyzers fail catastrophically on negated sentences: seeing 'no latency issues' or 'free of defects', a naive counter spots 'latency' and 'defects' as negative, classifying a delighted customer as a complaint! NegEx scope inversion flips the sign to positive, while the 0.5 attenuation reflects that stating the absence of a defect indicates acceptable baseline operation rather than extraordinary delight."
            }
        },
        "excel_guide": {
            "cell_mapping": "A1:E20 containing User_Review, Word_Count, Pos_Words, Neg_Words, Negated_Flag, Compound_Score",
            "steps": [
                "1. Tokenize comment text into individual word columns using TEXTSPLIT(A2, ' ').",
                "2. Cross-reference words against an AFINN/VADER lexicon table in sheet 'Lexicon!A:B' using XLOOKUP(token, Lexicon!A:A, Lexicon!B:B, 0).",
                "3. Check for negation triggers ('no', 'not', 'never', 'without') using OR(ISNUMBER(SEARCH({'no','not','never'}, A2))).",
                "4. Compute summated valence with =SUM(C2:Z2) and apply normalization formula =SUM_VAL / SQRT(SUM_VAL^2 + 15)."
            ]
        },
        "python_code": """import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER Sentiment Analyzer
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()

def analyze_review_sentiment(text: str) -> dict:
    # 1. Negation Scope Pre-Processing
    neg_triggers = [r"\\bno\\b", r"\\bnot\\b", r"\\bnever\\b", r"\\bwithout\\b", r"\\bfree of\\b", r"\\blacks\\b"]
    neg_pattern = re.compile("|".join(neg_triggers), re.IGNORECASE)
    
    has_negation = bool(neg_pattern.search(text))
    scores = sia.polarity_scores(text)
    
    # 2. Polarity Classification
    comp = scores['compound']
    if comp >= 0.05:
        category = "Positive"
    elif comp <= -0.05:
        category = "Negative"
    else:
        category = "Neutral"
        
    return {
        "compound": comp,
        "positive": scores['pos'],
        "neutral": scores['neu'],
        "negative": scores['neg'],
        "category": category,
        "has_negation": has_negation
    }

# Educational Sample Test: Customer Feedback (Fictional)
sample_review = "Customer service was prompt and helpful; no complaints about the onboarding process."
res = analyze_review_sentiment(sample_review)
print(f"Compound: {res['compound']:.3f} | Rating: {res['category']} | Negation Detected: {res['has_negation']}")
"""
    },

    # -----------------------------------------------------------------
    # Method 2: TF-IDF & Keyword Extraction
    # -----------------------------------------------------------------
    "tfidf_keywords": {
        "id": "tfidf_keywords",
        "name": "TF-IDF Vectorization & Keyword Extraction",
        "category": "Level 2: Statistical Vectorization & Bag-of-Words",
        "level": 2,
        "icon": "🔤",
        "summary": "Transforms unstructured text into weighted term vectors, boosting domain-specific keywords while penalizing frequent non-informative stopwords.",
        "when_to_use": [
            "Extracting salient technical terms and root causes across thousands of IT support tickets.",
            "Building baseline search indexing across enterprise policy documents or customer dispute logs.",
            "Measuring document-to-document similarity via Cosine Similarity on sparse numerical matrices."
        ],
        "key_assumptions": [
            "Bag-of-Words Independence: Word order within a document is disregarded; semantic meaning is captured through term co-occurrence frequencies.",
            "Zipf's Law Distribution: High-frequency terms across all documents (e.g. 'system', 'issue') carry low informational entropy and must be penalized via IDF.",
            "L2 Vector Normalization: Long documents must be normalized to prevent raw word counts from dominating similarity calculations."
        ],
        "common_pitfalls": [
            "Stopword Contamination: Failing to remove company or domain stop words ('ticket', 'user', 'customer', 'please') which dominate top TF-IDF ranks without providing distinction.",
            "Out-of-Vocabulary Blindness: TF-IDF cannot compute representations for novel words or spelling typos not present in the training vocabulary.",
            "Polysemy Ignorance: Words with multiple meanings (e.g. 'bug' insect vs. 'bug' software defect) receive identical vector indices."
        ],
        "plain_english_rules": [
            "Term Frequency (TF) measures how often word t appears in document d: words repeated frequently inside a single note receive higher local weight.",
            "Inverse Document Frequency (IDF) measures how rare word t is across the entire corpus: words appearing in every document receive near-zero weight.",
            "Multiplying TF by IDF rewards words that are frequent locally but rare globally (e.g. 'deadlock' vs. 'system').",
            "Cosine similarity between two TF-IDF vectors calculates the cosine of the angle between them (1.0 = identical vocabulary distribution, 0.0 = completely orthogonal)."
        ],
        "formulas": {
            "tf_formula": {
                "title": "Relative Term Frequency (TF)",
                "latex": r"\text{TF}(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}",
                "plain_english": "Calculates relative term frequency: the raw count of word t in document d divided by the total number of words in that document.",
                "terms": [
                    ("t", "The target vocabulary word or n-gram being evaluated."),
                    ("d", "The specific document, customer review, or support ticket."),
                    ("f_{t, d}", "The raw count of occurrences of word t inside document d."),
                    (r"\sum_{t' \in d} f_{t', d}", "The total word count of document d (document length)."),
                    (r"\text{TF}(t, d)", "The relative term frequency proportion, strictly bounded between 0.0 and 1.0.")
                ],
                "steps": [
                    "1. **Tokenization:** Split document d into lowercase word tokens.",
                    "2. **Frequency Count:** Count the number of times word t appears in document d (f_{t,d}).",
                    "3. **Document Length Calculation:** Sum the total number of word tokens in document d.",
                    "4. **Relative Proportion:** Divide the word frequency by the document length to compute TF(t, d)."
                ],
                "why_used": "Raw word counts heavily bias analysis toward verbose documents: a 500-word post-mortem report would have term frequencies 10 times higher than a 50-word helpdesk ticket simply due to length. Relative TF normalizes for document length, placing concise tickets on equal analytical footing."
            },
            "idf_formula": {
                "title": "Smooth Inverse Document Frequency (IDF)",
                "latex": r"\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1",
                "plain_english": "Smooth Inverse Document Frequency: takes the natural log of total corpus size (|D|) divided by document frequency, adding 1 to avoid division by zero and negative weights.",
                "terms": [
                    ("|D|", "Total number of documents in the entire corpus or database."),
                    (r"|\{d \in D : t \in d\}|", "Document Frequency (DF_t), the count of documents that contain word t at least once."),
                    ("1 + |D|", "Smoothed corpus size preventing zero numerators."),
                    ("1 + DF_t", "Smoothed document frequency preventing division by zero for rare words."),
                    (r"\ln(\cdot)", "The natural logarithm function, dampening extreme ratio scales."),
                    ("+ 1", "Baseline offset ensuring that even terms appearing in every document maintain an IDF weight >= 1.0 rather than 0.0.")
                ],
                "steps": [
                    "1. **Corpus Size Determination:** Count total documents in the collection (|D|).",
                    "2. **Document Frequency Counting:** Count how many documents contain word t at least once (DF_t).",
                    "3. **Smoothed Ratio Calculation:** Add 1 to both numerator and denominator: (1 + |D|) / (1 + DF_t).",
                    "4. **Logarithmic Scaling & Offset:** Take the natural log of the ratio and add 1: IDF = ln(ratio) + 1."
                ],
                "why_used": "Ubiquitous words like 'system', 'issue', and 'user' appear in virtually every ticket. They have high frequency but carry zero diagnostic specificity. Rare words like 'deadlock', 'segfault', or 'kernel panic' appear in very few tickets and pinpoint root causes. IDF formalizes this signal-to-noise ratio by heavily discounting ubiquitous words."
            },
            "cosine_similarity": {
                "title": "Cosine Similarity on L2-Normalized Vectors",
                "latex": r"\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2} = \frac{\sum_{i=1}^V u_i v_i}{\sqrt{\sum_{i=1}^V u_i^2} \sqrt{\sum_{i=1}^V v_i^2}}",
                "plain_english": "Calculates the dot product of two normalized TF-IDF vectors, yielding a similarity metric strictly bounded between 0.0 (no shared words) and 1.0 (identical proportions).",
                "terms": [
                    (r"\vec{u}, \vec{v}", "The TF-IDF weight vectors of two separate documents across vocabulary V."),
                    ("u_i, v_i", "The TF-IDF weight of vocabulary term i in documents u and v."),
                    (r"\vec{u} \cdot \vec{v}", "The dot product, summing the products of shared vocabulary terms."),
                    (r"\|\vec{u}\|_2", "The Euclidean (L2) norm of vector u: \\sqrt{\\sum u_i^2}."),
                    ("Cosine Similarity", "Angular similarity metric: 1.0 indicates identical vocabulary proportions; 0.0 indicates completely orthogonal documents with zero shared words.")
                ],
                "steps": [
                    "1. **Dot Product:** Multiply the matching TF-IDF weights of words appearing in both documents and sum the products: \\sum u_i v_i.",
                    "2. **Vector Magnitudes:** Compute Euclidean length for each vector by squaring each weight, summing them, and taking the square root.",
                    "3. **Angular Normalization:** Divide the dot product by the product of both vector lengths.",
                    "4. **Similarity Interpretation:** Evaluate similarity score: > 0.70 (duplicate/highly similar), 0.30 - 0.70 (moderate overlap), < 0.10 (unrelated)."
                ],
                "why_used": "Unlike Euclidean distance, which penalizes documents simply because one is longer than another, cosine similarity measures the angle (direction) of the vectors. Two incident tickets addressing the exact same database failure—one in 2 sentences and one in 2 paragraphs—point in the same direction and receive near-perfect cosine similarity."
            }
        },
        "excel_guide": {
            "cell_mapping": "Rows = Documents (1 to 20), Columns = Terms (A1:P20), Term Counts in C4:N15",
            "steps": [
                "1. Compute Term Frequency in cell C4: =C4/SUM($C4:$N4).",
                "2. Compute Document Frequency in row 16: =COUNTIF(C4:C15, '>0').",
                "3. Compute IDF in row 17: =LN((1 + 12)/(1 + C16)) + 1.",
                "4. Compute TF-IDF matrix: =C4 * C$17.",
                "5. Calculate Cosine Similarity between Doc 1 and Doc 2: =SUMPRODUCT(C4:N4, C5:N5) / (SQRT(SUMSQ(C4:N4)) * SQRT(SUMSQ(C5:N5)))."
            ]
        },
        "python_code": """import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# IT Incident Ticket Corpus (Educational & Fictional)
corpus = [
    "Server outage: Production database connection pool exhausted during payment gateway surge.",
    "Database connection timeout during checkout payment processing on production cluster.",
    "Warehouse shipment delayed due to inventory barcode scanner hardware failure.",
    "Payment gateway API rejected batch credit card transactions due to expired SSL certificate."
]

# Initialize TF-IDF Vectorizer with Stopword Filtration
vectorizer = TfidfVectorizer(stop_words='english', norm='l2', max_features=15)
tfidf_matrix = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()

df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Top Vocabulary Terms:", list(feature_names[:8]))
print(f"Cosine Similarity (Ticket 0 vs Ticket 1 - Database Incidents): {similarity_matrix[0, 1]:.3f}")
print(f"Cosine Similarity (Ticket 0 vs Ticket 2 - Database vs Warehouse): {similarity_matrix[0, 2]:.3f}")
"""
    },

    # -----------------------------------------------------------------
    # Method 3: Supervised Text Classification (Naive Bayes / Linear SVM)
    # -----------------------------------------------------------------
    "text_classification_bayes_svm": {
        "id": "text_classification_bayes_svm",
        "name": "Supervised Text Classification (Naive Bayes & Linear SVM)",
        "category": "Level 2: Statistical Vectorization & Bag-of-Words",
        "level": 2,
        "icon": "🏷️",
        "summary": "Trains interpretable supervised classifiers (Multinomial Naive Bayes and Linear Support Vector Machines) to triage IT incident tickets into priority tiers.",
        "when_to_use": [
            "Automated triage of IT service tickets and customer complaints into urgency tiers (Routine, Operational, Critical Outage).",
            "Classifying customer inquiries into product departments (Billing vs Technical Support vs Logistics).",
            "Benchmarking fast linear statistical classifiers before deploying resource-heavy LLMs."
        ],
        "key_assumptions": [
            "Conditional Word Independence (Naive Bayes): Words are assumed conditionally independent given the class label ($P(w_1, w_2 | C) = P(w_1 | C) P(w_2 | C)$).",
            "Linear Separability in High Dimensions (Linear SVM): Sparse TF-IDF feature spaces are largely linearly separable, allowing a maximal-margin hyperplane to classify texts.",
            "Representative Labelled Training Data: Class balance and category annotations must reflect real-world incident frequencies."
        ],
        "common_pitfalls": [
            "Zero Probability Problem: In Naive Bayes, encountering an unseen word in test data produces a zero probability without Laplace (add-1) smoothing.",
            "Severe Class Imbalance: Rare critical outages (e.g. 2% P1 Outages) get ignored by accuracy-maximizing classifiers; requires class weighting or PR-AUC tuning.",
            "Spurious Lexical Shortcuts: Model memorizing technician names or specific server hostnames as predictive indicators of critical outages."
        ],
        "plain_english_rules": [
            "Multinomial Naive Bayes multiplies the prior probability of each class by the likelihood of each observed word occurring in that class.",
            "Laplace (+1) smoothing adds a pseudo-count of 1 to every vocabulary word so unseen terms never zero out the entire posterior probability.",
            "Linear SVM finds the optimal decision boundary that maximizes the geometric margin between different incident classes.",
            "Model performance must be judged on Macro F1 and Precision/Recall trade-offs, rather than raw accuracy."
        ],
        "formulas": {
            "naive_bayes_posterior": {
                "title": "Multinomial Naive Bayes Posterior Probability with Laplace Smoothing",
                "latex": r"P(C_k | d) \propto P(C_k) \prod_{i=1}^{n} P(w_i | C_k), \quad P(w_i | C_k) = \frac{N_{ki} + \alpha}{N_k + \alpha |V|}",
                "plain_english": "Calculates class posterior: prior class probability P(C_k) multiplied by the product of word likelihoods with Laplace smoothing parameter alpha (usually 1.0) and vocabulary size |V|.",
                "terms": [
                    ("P(C_k | d)", "The posterior probability that ticket d belongs to severity category C_k (e.g. 'Critical Outage')."),
                    ("P(C_k)", "The prior probability of class C_k (proportion of historical training tickets in that tier)."),
                    ("w_i", "Word token i appearing in ticket d."),
                    ("N_{ki}", "The total occurrences of word w_i across all training tickets of class C_k."),
                    ("N_k", "The total word count across all tickets in class C_k."),
                    (r"\alpha", "Laplace smoothing parameter (alpha = 1.0 for add-one smoothing), preventing zero-probability penalties."),
                    ("|V|", "Vocabulary size (count of unique words across the entire training corpus).")
                ],
                "steps": [
                    "1. **Prior Estimation:** Compute P(C_k) = (Tickets in Class k) / (Total Training Tickets).",
                    "2. **Smoothed Likelihood Table:** For each word w, calculate P(w | C_k) = (N_{ki} + 1) / (N_k + |V|).",
                    "3. **Log-Likelihood Summation:** Convert to natural logs to prevent underflow: Score_k = ln P(C_k) + \\sum ln P(w_i | C_k).",
                    "4. **Softmax Normalization:** Exponentiate and normalize across classes so probabilities sum to 100%.",
                    "5. **Triage Classification:** Assign the ticket to the category with the highest posterior probability."
                ],
                "why_used": "Without Laplace smoothing (alpha = 1), a single novel word in an incoming ticket that never appeared in historical 'Critical Outage' logs would have P(w | Critical) = 0. Multiplying by zero zeroes out the entire class probability! Laplace smoothing adds a uniform pseudo-count across the vocabulary, guaranteeing robust classification."
            },
            "svm_hinge_loss": {
                "title": "Linear Support Vector Machine (SVM) Soft-Margin Objective",
                "latex": r"\min_{\vec{w}, b, \vec{\xi}} \frac{1}{2} \|\vec{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t.} \quad y_i (\vec{w} \cdot \vec{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0",
                "plain_english": "Linear SVM optimization: balances maximizing the classification margin (minimizing ||w||^2) against soft-margin hinge loss penalty C for misclassified documents.",
                "terms": [
                    (r"\vec{w}", "The normal weight vector defining the orientation of the separating hyperplane in feature space."),
                    ("b", "The scalar bias term positioning the hyperplane relative to the origin."),
                    (r"\frac{1}{2}\|\vec{w}\|^2", "The margin maximization term; minimizing ||w|| maximizes the geometric margin width 2 / ||w||."),
                    ("C", "Regularization penalty hyperparameter balancing margin width against training error tolerance."),
                    (r"\xi_i", "Slack variable for document i, quantifying penetration or violation across the margin boundary."),
                    ("y_i", "The true binary or one-vs-rest severity label (y_i in {-1, +1})."),
                    (r"\vec{x}_i", "The normalized TF-IDF feature vector of document i.")
                ],
                "steps": [
                    "1. **TF-IDF Feature Space:** Map all training tickets into high-dimensional normalized TF-IDF vectors.",
                    "2. **Convex Quadratic Optimization:** Solve the dual optimization problem to identify Support Vectors and optimal weights w.",
                    "3. **Decision Score Computation:** For an unlabelled incoming ticket x, calculate f(x) = w · x + b.",
                    "4. **Classification Rule:** Assign to Positive Class (+1) if f(x) >= 0; assign to Negative Class (-1) if f(x) < 0."
                ],
                "why_used": "Text feature spaces contain thousands of vocabulary dimensions (|V| > 10,000). While most algorithms overfit in high dimensions, SVM's theoretical generalization error is bounded by the geometric margin rather than the feature dimensionality, making Linear SVM the most robust, high-precision statistical classifier for text."
            }
        },
        "excel_guide": {
            "cell_mapping": "Training rows in A1:D50 (Incident_Text, Class, P(w|Low), P(w|High)); Test score in F2",
            "steps": [
                "1. Compute Class Priors in G1:G2: =COUNTIF(B2:B50, 'High') / COUNTA(B2:B50).",
                "2. Calculate word frequencies per class and compute smoothed log-likelihood: =LN((COUNTIF_Word_in_Class + 1) / (Total_Words_in_Class + Vocab_Size)).",
                "3. Sum log-probabilities for test document: =LN(G1) + SUMPRODUCT(Token_Counts, Word_Log_Likelihoods).",
                "4. Assign predicted category using =IF(Score_High > Score_Low, 'Critical Outage', 'Routine Ticket')."
            ]
        },
        "python_code": """import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Synthetic IT Incident Dataset (Educational & Fictional)
data = {
    "text": [
        "Production database connection pool exhausted during payment gateway surge; orders failing.",
        "Payment gateway API rejected batch credit card transactions due to expired SSL certificate.",
        "Warehouse inventory barcode scanner battery depleted, minor dispatch delay.",
        "Single user unable to export monthly billing report due to local browser cache.",
        "Requesting software license seat renewal for developer workstation in engineering.",
        "Primary routing switch hardware failure in regional datacentre, traffic rerouted."
    ],
    "severity": ["Critical", "Critical", "Medium", "Low", "Low", "Critical"]
}
df = pd.DataFrame(data)

# Vectorization & Model Training
vec = TfidfVectorizer(stop_words='english')
X = vec.fit_transform(df['text'])
y = df['severity']

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X, y)

svm_model = LinearSVC(C=1.0, random_state=42)
svm_model.fit(X, y)

# Predict on New Incident Note
new_incident = ["Cluster database node unresponsive; customer checkout transactions timing out."]
X_new = vec.transform(new_incident)
print("Naive Bayes Prediction:", nb_model.predict(X_new)[0])
print("Linear SVM Prediction:", svm_model.predict(X_new)[0])
"""
    },

    # -----------------------------------------------------------------
    # Method 4: LDA Topic Modelling
    # -----------------------------------------------------------------
    "lda_topic_modelling": {
        "id": "lda_topic_modelling",
        "name": "Latent Dirichlet Allocation (LDA) & Thematic Discovery",
        "category": "Level 3: Unsupervised Thematic Topic Modelling",
        "level": 3,
        "icon": "🧩",
        "summary": "Discovers latent generative themes and word probability distributions across unlabelled customer reviews or support transcripts.",
        "when_to_use": [
            "Synthesizing thousands of unlabelled customer reviews to identify emergent systemic themes (e.g. Shipping Speed, Software Bugs, Subscription Pricing, Customer Support Quality).",
            "Thematic mapping of open-text employee engagement surveys or user feedback forums.",
            "Exploratory qualitative thematic coding to guide product roadmap prioritisation."
        ],
        "key_assumptions": [
            "Document Mixture Hypothesis: Every document is modeled as a multinomial distribution over latent topics ($\theta \sim \text{Dir}(\alpha)$).",
            "Topic Word Mixture Hypothesis: Every topic is modeled as a multinomial distribution over vocabulary words ($\phi \sim \text{Dir}(\beta)$).",
            "Pre-Specified Number of Topics ($k$): The user must select the number of topics $k$; optimal $k$ is determined by maximizing Topic Coherence ($C_v$)."
        ],
        "common_pitfalls": [
            "Arbitrary Topic Count Selection: Picking $k=10$ without evaluating coherence or human interpretability, leading to fragmented or redundant topics.",
            "Stopword Contamination: Non-informative words ('also', 'good', 'said') dominating all topics without domain stopword filtration.",
            "Misinterpreting Probabilities as Mutually Exclusive: Expecting documents to belong to only one topic, when LDA naturally assigns continuous topic mixtures."
        ],
        "plain_english_rules": [
            "LDA imagines that every document was written by picking a mix of topics, and then picking words from each topic's vocabulary basket.",
            "Alpha ($\alpha$) controls document-topic sparsity: high alpha means documents contain a blend of many topics; low alpha means documents focus on 1 or 2 topics.",
            "Beta ($\beta$) controls topic-word sparsity: low beta means topics are characterized by a small, distinct set of vocabulary terms.",
            "Topic Coherence ($C_v$) measures whether the top words in a topic frequently co-occur together in natural text (higher coherence = more interpretable themes)."
        ],
        "formulas": {
            "lda_joint_distribution": {
                "title": "Generative Document-Word Latent Topic Decomposition",
                "latex": r"P(w | d) = \sum_{z=1}^{K} P(w | z) P(z | d) = \sum_{z=1}^{K} \phi_{z, w} \cdot \theta_{d, z}",
                "plain_english": "Joint generative distribution: models M documents where each word w_dn has a latent topic assignment z_dn sampled from document topic distribution theta_d and vocabulary matrix Phi.",
                "terms": [
                    ("P(w | d)", "The probability of observing word w inside customer document d."),
                    ("K", "The analyst-specified number of latent topics (e.g. K = 4)."),
                    ("z", "Latent topic index (z in {1, 2, ..., K})."),
                    (r"\theta_{d, z} = P(z | d)", "The document-topic mixture proportion (percentage of document d devoted to topic z)."),
                    (r"\phi_{z, w} = P(w | z)", "The topic-word probability (probability of word w being emitted when topic z is active).")
                ],
                "steps": [
                    "1. **Random Initialization:** Randomly assign each word token in every document to one of the K topics.",
                    "2. **Count Matrix Construction:** Tabulate document-topic counts n_{d,k} and topic-word counts n_{k,w}.",
                    "3. **Collapsed Gibbs Sampling:** Iteratively resample topic z for each token conditioned on all other assignments: P(z_i = k) \\propto \\frac{n_{d,k} + \\alpha}{n_d + K\\alpha} \\times \\frac{n_{k,w} + \\beta}{n_k + V\\beta}.",
                    "4. **Parameter Extraction:** Compute normalized matrices Theta (document mixtures) and Phi (topic vocabularies).",
                    "5. **Thematic Labeling:** Inspect top words with highest phi_{z,w} to assign business themes (e.g. 'Shipping & Delivery', 'Software Bugs')."
                ],
                "why_used": "Real customer feedback is multifaceted: a single complaint might address billing overcharges (Topic 1) and unhelpful support agents (Topic 2). Hard clustering algorithms (like k-means) force each review into a single bucket, whereas LDA naturally decomposes each review into a mixed proportion of topics."
            },
            "topic_coherence": {
                "title": "Topic Coherence (C_v) via Normalized Pointwise Mutual Information (NPMI)",
                "latex": r"C_v = \frac{2}{M(M-1)} \sum_{i < j} \text{NPMI}(w_i, w_j) = \frac{2}{M(M-1)} \sum_{i < j} \frac{\ln \frac{P(w_i, w_j) + \epsilon}{P(w_i)P(w_j)}}{-\ln(P(w_i, w_j) + \epsilon)}",
                "plain_english": "Normalized Pointwise Mutual Information (NPMI) coherence: measures the statistical association between the top N words of a topic across reference documents.",
                "terms": [
                    ("M", "Number of top characteristic words evaluated per topic (typically the top M = 10 highest-probability terms)."),
                    ("w_i, w_j", "A pair of distinct top words from the same topic."),
                    ("P(w_i, w_j)", "Co-occurrence probability: proportion of text windows containing both words simultaneously."),
                    ("P(w_i), P(w_j)", "Marginal word probabilities across the reference corpus."),
                    ("NPMI", "Normalized Pointwise Mutual Information, bounded between -1.0 (never co-occur) and +1.0 (always co-occur)."),
                    (r"\epsilon", "Small numerical constant avoiding log(0)."),
                    ("C_v", "Overall topic coherence score; scores >= 0.55 indicate human-interpretable, high-quality themes.")
                ],
                "steps": [
                    "1. **Top Word Selection:** Extract the top M = 10 words with the highest generation probability phi_{z,w} for topic z.",
                    "2. **Pairwise Enumeration:** Form all unique pairs of top words: (10 * 9) / 2 = 45 word pairs.",
                    "3. **Co-occurrence Counting:** Using a sliding window of 110 tokens across the corpus, count joint occurrences P(w_i, w_j) and marginal occurrences P(w_i), P(w_j).",
                    "4. **NPMI Calculation:** Compute NPMI for each word pair and take the arithmetic mean across all 45 pairs.",
                    "5. **Quality Evaluation:** Benchmarks: C_v >= 0.60 (Strong coherence), 0.45 - 0.59 (Moderate), < 0.45 (Diffuse/Noisy topic)."
                ],
                "why_used": "Mathematical perplexity measures how well the model predicts held-out words, but frequently rewards diffuse, uninterpretable topics that humans find nonsensical. Topic Coherence C_v directly measures semantic association, preventing the deployment of confusing statistical artifacts to executive leadership."
            }
        },
        "excel_guide": {
            "cell_mapping": "Document-Term Co-occurrence Matrix in A1:M15; Topic Proportions in P1:R15",
            "steps": [
                "1. Build document-term binary occurrence matrix using =IF(COUNTIF(Doc_Cell, '*term*') > 0, 1, 0).",
                "2. Approximate topic clusters using principal component correlation or matrix factorization.",
                "3. Normalize row topic mixtures so probabilities sum to 100%: =P2/SUM(P2:R2).",
                "4. Identify dominant topic using =INDEX(Topic_Names, MATCH(MAX(P2:R2), P2:R2, 0))."
            ]
        },
        "python_code": """import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Customer Feedback Corpus (Educational & Fictional)
reviews = [
    "Delivery took 10 business days and shipping package arrived torn and battered.",
    "Courier left parcel outside in the rain without ringing the doorbell; box soaked.",
    "Mobile app constantly freezes on startup and crashes after the latest update.",
    "Interface is sluggish and search bar takes 15 seconds to load catalog items.",
    "Billed twice for annual subscription and customer support took three weeks to issue refund.",
    "Support agent resolved my API webhook configuration in five minutes with great courtesy."
]

# Bag-of-Words Vectorization
vec = CountVectorizer(stop_words='english', min_df=1)
X = vec.fit_transform(reviews)
feature_names = vec.get_feature_names_out()

# 2. Fit LDA with 3 Latent Topics
lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=20)
lda.fit(X)

for topic_idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
"""
    },

    # -----------------------------------------------------------------
    # Method 5: Dense Semantic Embeddings & KNN
    # -----------------------------------------------------------------
    "dense_embeddings_similarity": {
        "id": "dense_embeddings_similarity",
        "name": "Semantic Embeddings & Vector Search (Sentence Transformers)",
        "category": "Level 4: Dense Semantic Vector Embeddings",
        "level": 4,
        "icon": "🌌",
        "summary": "Encodes free text into 384/768-dimensional dense vector embeddings, capturing deep semantic equivalence across distinct vocabularies via Cosine Similarity.",
        "when_to_use": [
            "Semantic search across customer reviews where phrasing differs ('app is sluggish' vs 'interface freezes frequently').",
            "Clustering enterprise documentation or user inquiries by conceptual meaning rather than exact keyword overlap.",
            "Building the vector database retrieval foundation for Retrieval-Augmented Generation (RAG) pipelines."
        ],
        "key_assumptions": [
            "Semantic Manifold Hypothesis: Conceptually related phrases map to nearby coordinates in a continuous dense vector space ($\mathbb{R}^{384}$ or $\mathbb{R}^{768}$).",
            "Contextual Attention: Words derive meaning from surrounding bi-directional token contexts via multi-head self-attention mechanisms.",
            "Cosine Proximity Metric: Angular separation between normalized vector embeddings reflects semantic similarity ($1.0 = \text{synonymous}$, $0.0 = \text{orthogonal}$)."
        ],
        "common_pitfalls": [
            "Negation Blindness in Naive Models: Some generic sentence transformers fail subtle negations ('no downtime experienced' mapping close to 'downtime experienced').",
            "Domain Shift: General-domain models (trained on Wikipedia/Reddit) misinterpreting specialized corporate or engineering acronyms.",
            "Truncation of Long Documents: Standard sentence transformers have a 256 or 512 token limit; enterprise text exceeding this must be chunked."
        ],
        "plain_english_rules": [
            "Unlike TF-IDF, which treats words as isolated index numbers, dense embeddings map text into a continuous geometric space of semantic concepts.",
            "Words with similar meanings ('support', 'helpdesk', 'representative') have vectors pointing in nearly identical directions.",
            "Sentence embeddings represent the entire semantic thought by mean-pooling the contextualized token vectors produced by a transformer model.",
            "Finding the most relevant customer comments involves computing the cosine similarity between the user's query vector and all stored document vectors."
        ],
        "formulas": {
            "mean_pooling": {
                "title": "Dense Semantic Embedding Projection & Mean Pooling",
                "latex": r"\vec{e}_{\text{doc}} = \frac{\sum_{i=1}^{L} m_i \cdot \vec{h}_i}{\sum_{i=1}^{L} m_i}, \quad \vec{h}_i = \text{Transformer}(\text{token}_i) \in \mathbb{R}^{d}",
                "plain_english": "Calculates sentence embedding by taking the element-wise average of all contextual token hidden states (h_i), masked by attention mask m_i to ignore padding tokens.",
                "terms": [
                    (r"\vec{e}_{\text{doc}}", "The pooled document embedding vector (e.g. \\vec{e} in R^{384} for all-MiniLM-L6-v2)."),
                    (r"\vec{h}_i", "The contextual hidden-state vector output by the final Transformer encoder layer for token i."),
                    ("m_i", "Binary attention mask value (m_i = 1 for real word tokens, m_i = 0 for padding tokens)."),
                    ("L", "Sequence token length (up to model context ceiling, e.g. 256 or 512)."),
                    (r"\sum m_i", "The count of non-padding tokens in the input text sequence.")
                ],
                "steps": [
                    "1. **Subword Tokenization:** Tokenize input text using Byte-Pair Encoding (BPE) or WordPiece into numerical token IDs.",
                    "2. **Transformer Encoding:** Pass token sequence through multi-head self-attention layers to compute contextual representations h_i.",
                    "3. **Attention Masking:** Multiply each hidden state h_i by its attention mask m_i to zero out padding tokens.",
                    "4. **Mean Pooling:** Sum the unmasked token vectors and divide by active token count \\sum m_i.",
                    "5. **L2 Unit Normalization:** Normalize the pooled vector: e_final = e / ||e||_2, guaranteeing ||e_final|| = 1.0."
                ],
                "why_used": "Transformers generate an individual contextual vector for every single subword token. Mean pooling condenses this variable-length sequence of token vectors into a single fixed-size semantic representation (384 dimensions) representing the holistic thought, enabling fast vector database indexing and search."
            },
            "vector_cosine_sim": {
                "title": "Continuous Semantic Cosine Proximity & Euclidean Manifold Distance",
                "latex": r"\text{Sim}(\vec{q}, \vec{d}) = \vec{q} \cdot \vec{d} = \sum_{j=1}^{d} q_j \cdot d_j, \quad D_{\text{Euclidean}}^2 = 2 \left(1 - \text{Sim}(\vec{q}, \vec{d})\right)",
                "plain_english": "Computes the cosine of the angle between two dense d-dimensional embeddings; normalized embeddings simplify to a direct dot product u · v.",
                "terms": [
                    (r"\vec{q}", "The normalized 384-dimensional dense embedding of the search inquiry (||q||_2 = 1.0)."),
                    (r"\vec{d}", "The normalized dense embedding of a stored feedback entry or ticket (||d||_2 = 1.0)."),
                    (r"\text{Sim}(\vec{q}, \vec{d})", "Cosine similarity bounded between -1.0 and +1.0 (typically 0.0 to 1.0 for text)."),
                    ("D_{Euclidean}^2", "Squared Euclidean distance on the unit hypersphere manifold.")
                ],
                "steps": [
                    "1. **Query Encoding:** Encode the user query into normalized vector q using the identical Transformer architecture.",
                    "2. **Dot Product Vector Search:** Compute the dot product between q and each pre-indexed document vector d_i.",
                    "3. **Top-K Ranking:** Sort corpus documents by cosine similarity score in descending order.",
                    "4. **Manifold Projection for Visualizer:** Project 384D vectors onto 2 principal components via PCA for 2D spatial scatter visualization."
                ],
                "why_used": "Traditional keyword search fails when phrasing differs: searching 'application is sluggish and freezes' yields zero matches against 'mobile software crashes and stutters' in keyword systems. Because dense embeddings position semantically synonymous concepts close together in continuous geometric space, their cosine similarity remains high (> 0.82) despite zero overlapping words."
            }
        },
        "excel_guide": {
            "cell_mapping": "Document Embeddings (16 dimensions for demo) in B2:Q10; Query Vector in B13:Q13",
            "steps": [
                "1. Place pre-computed 16-dimensional embedding vectors for each document in rows B2:Q10.",
                "2. Place user query vector in B13:Q13.",
                "3. Compute Dot Product with Doc 1: =SUMPRODUCT(B2:Q2, $B$13:$Q$13).",
                "4. Compute Vector Norms: =SQRT(SUMSQ(B2:Q2)) and =SQRT(SUMSQ($B$13:$Q$13)).",
                "5. Compute Cosine Similarity: =DotProduct / (Norm1 * Norm2). Rank with =RANK(R2, $R$2:$R$10)."
            ]
        },
        "python_code": """# Production Python Pipeline: Dense Semantic Embeddings & Vector Search
# Utilizing Sentence-Transformers (all-MiniLM-L6-v2)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Initialize Production Dense Embedding Encoder (384 dimensions)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 2. Customer Feedback & Incident Corpus
corpus = [
    "Support representative was polite, attentive, and resolved my issue in minutes.",
    "Support agent was rude, dismissive, and abruptly closed my helpdesk ticket.",
    "Mobile application is sluggish, stutters during scrolling, and freezes frequently.",
    "Subscription auto-renewed without email notice and pricing increased by 30 percent."
]

# 3. Compute Normalized Dense Vector Representations (shape: [4, 384])
corpus_embeddings = embedder.encode(corpus, normalize_embeddings=True)

# 4. Encode Search Inquiry
query = "customer service was impolite and unhelpful"
query_embedding = embedder.encode([query], normalize_embeddings=True)

# 5. Compute Pairwise Cosine Similarities
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

# 6. Rank Results by Semantic Proximity
ranked_indices = np.argsort(similarities)[::-1]
print(f"Query: '{query}'\\n")
for rank, idx in enumerate(ranked_indices, 1):
    print(f"Rank {rank} (Sim: {similarities[idx]:.3f}): {corpus[idx]}")
"""
    },

    # -----------------------------------------------------------------
    # Method 6: Named Entity Recognition (NER)
    # -----------------------------------------------------------------
    "clinical_ner": {
        "id": "clinical_ner",
        "name": "Named Entity Recognition (NER) & Information Extraction",
        "category": "Level 5: State-of-the-Art Generative AI & LLMs",
        "level": 5,
        "icon": "🏷️",
        "summary": "Extracts structured business and technical entities (Organizations, Products, Specifications, Defects, Prices) with IOB token span tagging.",
        "when_to_use": [
            "Parsing unstructured IT incident tickets or customer support logs into structured database records.",
            "Extracting product models, specifications, and prices from commercial invoices and sales agreements.",
            "De-identifying enterprise documents (GDPR compliance) by redacting client and organization names."
        ],
        "key_assumptions": [
            "Sequential Token Tagging: Entities are segmented using IOB tagging (B-Beginning, I-Inside, O-Outside).",
            "Contextual Boundary Precision: Domain entities often span multi-word compounds ('MacBook Pro', 'Dell PowerEdge', '32GB RAM').",
            "Ontological Mapping: Extracted surface mentions should map back to master catalog indices or enterprise databases."
        ],
        "common_pitfalls": [
            "Partial Span Truncation: Tagging 'Pro' but missing 'MacBook' in 'MacBook Pro', resulting in incomplete product tracking.",
            "Generic Pipeline Fallacy: Using standard off-the-shelf models trained on generic news text which misclassify technical hardware specifications.",
            "Abbreviation Ambiguity: Corporate acronyms having conflicting meanings depending on context (e.g. 'PR' = Pull Request vs Public Relations vs Purchase Requisition)."
        ],
        "plain_english_rules": [
            "Named Entity Recognition examines every word token and tags whether it begins an entity (B), continues inside an entity (I), or is outside any entity (O).",
            "Entities are categorized into semantic types: Organization, Product, Specification, Defect, and Price.",
            "Evaluation compares predicted token spans against gold-standard human annotations using strict F1 score (both boundaries and entity type must match).",
            "Structured extraction converts messy free-text support tickets into queryable database records."
        ],
        "formulas": {
            "iob_sequence": {
                "title": "IOB2 Token Span Sequence Representation",
                "latex": r"\mathbf{Y} = (y_1, y_2, \dots, y_T) \quad \text{where } y_t \in \{B\text{-TYPE}, I\text{-TYPE}, O\}",
                "plain_english": "IOB2 token tagging convention: 'B-' marks the beginning of an entity mention, 'I-' denotes continuation inside a multi-token entity, and 'O' denotes non-entity tokens.",
                "terms": [
                    ("y_t", "The discrete tag assigned to token t in a sentence of length T."),
                    (r"B\text{-TYPE}", "Beginning token of an entity of category TYPE (e.g. B-ORG, B-PRODUCT, B-DEFECT, B-PRICE)."),
                    (r"I\text{-TYPE}", "Inside token continuing an existing entity span of the same category (e.g. 'Pro' in 'MacBook [B-PROD] Pro [I-PROD]')."),
                    ("O", "Outside token not belonging to any recognized entity mention."),
                    ("TYPE", "Target entity ontology class: ORGANIZATION, PRODUCT, SPECIFICATION, DEFECT, or PRICE.")
                ],
                "steps": [
                    "1. **Tokenization:** Segment unstructured text into discrete word and punctuation tokens.",
                    "2. **Feature Extraction:** Generate contextual embeddings and regex pattern indicators for each token.",
                    "3. **Sequential Tag Prediction:** Predict emission probabilities across all IOB classes under grammatical transition constraints (e.g. I-TYPE cannot follow O without an intervening B-TYPE).",
                    "4. **Span Reconstruction:** Merge consecutive B-TYPE followed by I-TYPE tokens into complete multi-word entity mentions."
                ],
                "why_used": "Multi-word entity mentions are ubiquitous in business and technology (e.g. 'Dell PowerEdge R750', '32GB ECC RAM', 'Kernel Panic'). Without the IOB convention, an algorithm cannot distinguish between two consecutive single-word products vs. one single two-word product, resulting in corrupted database records."
            },
            "strict_ner_f1": {
                "title": "Strict Span-Level Precision, Recall, and F1 Score",
                "latex": r"\text{Strict Entity } F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \times \text{TP}}{2 \times \text{TP} + \text{FP} + \text{FN}}",
                "plain_english": "Strict entity F1 score: requires both the exact word boundary indices and the predicted entity category to match reference truth; partial overlaps count as errors.",
                "terms": [
                    ("TP", "True Positive: Extracted entity mention where start character, end character, AND category match ground truth exactly."),
                    ("FP", "False Positive: Spurious extraction where no entity exists, or where boundaries/category were predicted incorrectly."),
                    ("FN", "False Negative: Ground truth entity mention that was completely missed by the model."),
                    ("Precision", "Proportion of extracted entities that were strictly correct: TP / (TP + FP)."),
                    ("Recall", "Proportion of true entities that were successfully recovered: TP / (TP + FN)."),
                    ("Strict F_1", "Harmonic mean of Precision and Recall, penalizing both precision errors and omitted entities.")
                ],
                "steps": [
                    "1. **Span Alignment:** Compare predicted entity spans [start, end, label] against gold-standard manual annotations.",
                    "2. **Contingency Tabulation:** If start, end, and label match, increment TP. If prediction has no matching ground truth, increment FP. For unmatched ground truth, increment FN.",
                    "3. **Precision & Recall:** Compute Precision = TP / (TP + FP) and Recall = TP / (TP + FN).",
                    "4. **Harmonic Mean Calculation:** Calculate F_1 = 2 * (Precision * Recall) / (Precision + Recall)."
                ],
                "why_used": "In natural language, over 90% of tokens are 'O' (non-entity words). A trivial model that predicts 'O' for every single word achieves > 90% token accuracy while extracting zero actual entities! Strict span F1 penalizes partial boundary overlaps and label mismatches, ensuring extracted entities are sufficiently accurate for automated downstream processing."
            }
        },
        "excel_guide": {
            "cell_mapping": "Raw Text in A2; Extracted Entity Columns in B2:F2 (Organization, Product, Specification, Defect, Price)",
            "steps": [
                "1. Match organization and product keywords against a master catalog list in sheet 'Catalog!A:B' using XLOOKUP with wildcards.",
                "2. Extract memory or hardware specifications followed by 'GB', 'TB', or 'RAM' using REGEXEXTRACT or MID/SEARCH formulas.",
                "3. Compile structured row: Ticket_ID, Organization, Product_Model, Specification, Defect.",
                "4. Flag missing critical elements (e.g. reported hardware defect without explicit product model) with =IF(ISBLANK(C2), '⚠️ Product Missing', 'OK')."
            ]
        },
        "python_code": """import re
from typing import List, Dict

# Business & Tech Information Extraction Engine (Rule + Regex / SpaCy pattern)
def extract_named_entities(text: str) -> List[Dict[str, str]]:
    patterns = {
        "ORGANIZATION": r"\\b(apple|amazon|microsoft|google|stripe|salesforce|acme corp|dell)\\b",
        "PRODUCT": r"\\b(macbook pro|iphone 15|ubuntu 22.04|kubernetes|postgresql|cloud sql|api gateway)\\b",
        "SPECIFICATION": r"\\b(\\d+\\s*(?:gb|tb|mb|ram|vcpu|cores))\\b",
        "DEFECT": r"\\b(kernel panic|memory leak|outage|downtime|nullpointerexception|crash)\\b",
        "PRICE": r"([£$€]\\s*\\d+(?:,\\d{3})*(?:\\.\\d{2})?)"
    }
    
    entities = []
    for ent_type, pat in patterns.items():
        for match in re.finditer(pat, text, re.IGNORECASE):
            entities.append({
                "text": match.group(0),
                "label": ent_type,
                "start": match.start(),
                "end": match.end()
            })
            
    # Sort entities by appearance in text
    return sorted(entities, key=lambda x: x["start"])

sample_ticket = "Client purchased a MacBook Pro with 32GB RAM from Apple Store London; paid £2,499."
found = extract_named_entities(sample_ticket)
for ent in found:
    print(f"[{ent['label']}] '{ent['text']}' (Index {ent['start']}:{ent['end']})")
"""
    },

    # -----------------------------------------------------------------
    # Method 7: LLM Zero-Shot & RAG Architecture
    # -----------------------------------------------------------------
    "llm_zeroshot_rag": {
        "id": "llm_zeroshot_rag",
        "name": "LLM Zero-Shot Classification & Retrieval-Augmented Generation (RAG)",
        "category": "Level 5: State-of-the-Art Generative AI & LLMs",
        "level": 5,
        "icon": "🤖",
        "summary": "Coordinates Large Language Models for unlabelled zero-shot classification and grounds generative answers in external company policy documents via RAG vector retrieval.",
        "when_to_use": [
            "Classifying complex, nuanced customer feedback into dynamic categories without labelled training data (Zero-Shot).",
            "Synthesizing automated helpdesk answers grounded strictly in verified company policies and SLAs (RAG).",
            "Mitigating LLM hallucination risk through context verification and citation tracing."
        ],
        "key_assumptions": [
            "Zero-Shot NLI Hypothesis: Prompting a foundation model with candidate hypotheses translates text classification into Natural Language Inference (Entailment vs Contradiction).",
            "Retrieval Sufficiency: The vector search retrieval step must retrieve chunks that actually contain the factual answers needed for generation.",
            "Faithfulness Constraint: The generative LLM must restrict its assertions strictly to the retrieved context chunks without hallucinating unverified claims."
        ],
        "common_pitfalls": [
            "Hallucination Risk: Generative LLMs inventing plausible-sounding but false policy terms when not grounded in retrieved documents.",
            "Sub-optimal Chunking: Chunking technical policies by arbitrary character counts (e.g. 500 chars) splitting SLA tables across separate chunks.",
            "Prompt Injection Vulnerability: Free text from external customer inputs hijacking the system prompt instructions."
        ],
        "plain_english_rules": [
            "Zero-shot classification asks an LLM: 'Does this text entail category X, Y, or Z?' without requiring any prior model training.",
            "Retrieval-Augmented Generation (RAG) works like an open-book exam: before the LLM answers, a vector database looks up the most relevant policy paragraphs.",
            "The retrieved paragraphs are injected into the prompt context so the LLM writes an answer grounded directly in verified source material.",
            "RAG evaluation monitors Faithfulness (are all statements backed by retrieved text?) and Answer Relevance (did the LLM answer the user's specific inquiry?)."
        ],
        "formulas": {
            "rag_context_injection": {
                "title": "Vector-Based Top-k Context Retrieval & Grounded Prompt Assembly",
                "latex": r"\text{Prompt} = \text{System Instruction} + \sum_{k=1}^K \text{argtop}_k \left\{\cos(\vec{e}_q, \vec{e}_{d_i}) : d_i \in \mathcal{K}\right\} + \text{Query } (q)",
                "plain_english": "RAG prompt assembly: Combines the system persona instruction, the top-K semantically retrieved reference document chunks, and the user's specific query.",
                "terms": [
                    ("q", "The operational inquiry or customer question submitted to the system."),
                    (r"\vec{e}_q", "The dense embedding vector representing the semantic meaning of the inquiry."),
                    (r"\mathcal{K}", "The enterprise policy repository containing all indexed document passages {d_1, d_2, ..., d_M}."),
                    (r"\vec{e}_{d_i}", "The pre-computed embedding vector for policy chunk d_i."),
                    ("K", "Retrieval depth (the count of top-scoring passages injected into prompt context, e.g. K = 1, 2, 3)."),
                    (r"\cos(\vec{e}_q, \vec{e}_{d_i})", "Cosine similarity metric evaluating semantic relevance between query and chunk.")
                ],
                "steps": [
                    "1. **Policy Chunking:** Divide enterprise documentation (SLAs, return policies, API specs) into coherent 200-400 word passages.",
                    "2. **Vector Indexing:** Generate normalized dense vector embeddings for all passages and store in a vector database.",
                    "3. **Semantic Lookup:** When inquiry q arrives, encode it and compute cosine similarity against all stored passages to select the top-K chunks.",
                    "4. **Grounded Prompt Construction:** Assemble prompt with system constraints: 'Answer strictly using the provided context passages below. If the context does not contain the answer, state that official policy is unavailable.'",
                    "5. **Conditioned Generation:** Pass assembled prompt to the LLM to generate a factual, compliant response."
                ],
                "why_used": "Off-the-shelf foundation LLMs suffer from knowledge cutoff and generate convincing hallucinations when asked about internal company SLAs or return policies. RAG converts generative answering into an open-book exam, ensuring answers are grounded strictly in approved corporate policy documents."
            },
            "faithfulness_score": {
                "title": "RAGAS Faithfulness & Hallucination Audit Metric",
                "latex": r"\text{Faithfulness Score} = \frac{|\{c \in \mathcal{C}_{\text{answer}} : \text{Context } \mathcal{R}(q) \models c\}|}{|\mathcal{C}_{\text{answer}}|} \in [0.0, 1.0]",
                "plain_english": "Measures the proportion of factual statements in the LLM response that are directly substantiated by the retrieved document chunks (1.0 = fully grounded, 0.0 = total hallucination).",
                "terms": [
                    (r"\mathcal{C}_{\text{answer}}", "The set of discrete, atomic factual statements extracted from the LLM's response."),
                    (r"|\mathcal{C}_{\text{answer}}|", "Total count of factual statements made in the generated answer."),
                    ("c", "An individual factual claim (e.g. 'Opened hardware returns incur a 10% restocking fee')."),
                    (r"\mathcal{R}(q) \models c", "Logical entailment: statement c is directly supported by retrieved context chunks R(q)."),
                    ("Faithfulness", "Verification ratio: 1.0 indicates complete factual grounding; < 0.80 indicates high hallucination risk.")
                ],
                "steps": [
                    "1. **Atomic Claim Extraction:** Decompose the generated response into individual factual propositions.",
                    "2. **Entailment Verification:** For each proposition, verify whether it can be inferred directly from the retrieved context passages.",
                    "3. **Claim Counting:** Count the number of verified claims vs ungrounded/hallucinated claims.",
                    "4. **Ratio Computation:** Divide verified claims by total claims to calculate Faithfulness percentage.",
                    "5. **Automated Guardrail:** If Faithfulness < 0.85, reject the output and trigger escalation or human-in-the-loop review."
                ],
                "why_used": "Standard text metrics (like BLEU or ROUGE) only check surface word overlap, completely missing dangerous factual hallucinations (such as changing 'P1 response time is 15 minutes' to '15 hours'). Faithfulness provides an automated audit guarantee before generative answers reach end users."
            }
        },
        "excel_guide": {
            "is_not_viable": True,
            "cell_mapping": "⚠️ Not Viable for Native Excel Execution — External API / Vector Database Tooling Required",
            "steps": [
                "1. **Architectural Feasibility Assessment:** Foundation LLMs (7B–70B parameters) and dense vector indexing (HNSW / FAISS) cannot execute natively within Microsoft Excel formulas.",
                "2. **Vector Database Deployment:** Deploy dedicated vector stores (Qdrant, ChromaDB, Pinecone, pgvector) to index 384–1536 dimensional document embeddings.",
                "3. **Orchestration Framework:** Utilize LangChain, LlamaIndex, or Haystack to coordinate document chunking, semantic retrieval, and prompt context injection.",
                "4. **Inference Runtime:** Serve models using Hugging Face `transformers`, vLLM, or Ollama for high-throughput, low-latency GPU/CPU execution.",
                "5. **Client-Side Integration (Optional):** If spreadsheet workflows are mandatory, connect Excel as a client via Power Query REST calls or Python in Excel (`PY()`) to query external RAG endpoints."
            ],
            "why_not_viable": "LLM inference requires billions of tensor floating-point operations and GPU memory bandwidth that standalone spreadsheets cannot perform. RAG vector search requires 384+ dimensional approximate nearest neighbor indexing that freezes spreadsheet calculation engines.",
            "recommended_tooling": {
                "vector_databases": "Qdrant, ChromaDB, Pinecone, pgvector (PostgreSQL), FAISS",
                "orchestration": "LangChain, LlamaIndex, Haystack, Semantic Kernel",
                "inference_engines": "Hugging Face transformers, vLLM, Ollama, TensorRT-LLM, llama.cpp",
                "evaluation_guardrails": "RAGAS, TruLens, DeepEval, NeMo Guardrails",
                "low_code_alternatives": "Langflow, Flowise, Microsoft Copilot Studio"
            }
        },
        "python_code": """\"\"\"
Production Python Pipeline: LLM Zero-Shot Classification & RAG Architecture
Utilizing Hugging Face Transformers & Sentence-Transformers
\"\"\"

import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# =====================================================================
# 1. Zero-Shot Incident & Inquiry Classification (Hugging Face BART-MNLI)
# =====================================================================
# Uses Natural Language Inference (NLI) to classify text without fine-tuning
classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli"  # Standard production zero-shot model
)

ticket = "Payment gateway API returned 429 Too Many Requests; transactions dropping."
candidate_departments = [
    "API Gateway & Rate Limiting",
    "Billing & Invoicing",
    "Hardware & Infrastructure",
    "Security & Governance"
]

classification_result = classifier(ticket, candidate_departments)
top_category = classification_result['labels'][0]
top_score = classification_result['scores'][0]
print(f"Zero-Shot Triage Category: {top_category} (Confidence: {top_score:.1%})")


# =====================================================================
# 2. Dense Semantic Vector Retrieval (Sentence-Transformers)
# =====================================================================
# In production, vectors are indexed in Qdrant, ChromaDB, or Pinecone
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

enterprise_knowledge_base = [
    "SLA Policy SEC-101: Enterprise SaaS subscriptions provide a 99.99% monthly uptime guarantee. For P1 critical outages, engineering guarantees an initial human response within 15 minutes.",
    "Hardware Return Policy RET-204: Customers may return unopened hardware devices within 30 days of delivery for a full refund. Opened electronic items are subject to a 10% restocking fee.",
    "API Gateway Policy API-305: All public REST API endpoints enforce a rate limit of 10,000 requests per minute per tenant API key. Exceeding limits returns HTTP 429 Too Many Requests.",
    "Data Governance Policy SEC-410: Customer database records are encrypted at rest using AES-256 GCM. Automated snapshot backups are retained for 90 days in an isolated vault."
]

# Pre-compute document chunk embeddings (shape: [4, 384])
chunk_embeddings = embedder.encode(enterprise_knowledge_base, normalize_embeddings=True)

def retrieve_context(query: str, top_k: int = 1):
    query_vector = embedder.encode([query], normalize_embeddings=True)[0]
    cosine_similarities = np.dot(chunk_embeddings, query_vector)
    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]
    return [(enterprise_knowledge_base[i], float(cosine_similarities[i])) for i in top_indices]


# =====================================================================
# 3. Retrieval-Augmented Generation (RAG) Prompt Assembly & Generation
# =====================================================================
user_inquiry = "What happens if our application sends 15,000 API calls in a single minute?"
retrieved_chunks = retrieve_context(user_inquiry, top_k=1)
best_chunk, sim_score = retrieved_chunks[0]

# System prompt with strict anti-hallucination grounding constraints
rag_prompt = f'''<|system|>
You are an enterprise AI assistant. Answer the user's inquiry strictly and faithfully using only the verified policy context below.
If the context does not contain the answer, say "This information is not covered by official company policy."
Do not invent or extrapolate terms.

VERIFIED POLICY CONTEXT:
{best_chunk}
<|user|>
INQUIRY: {user_inquiry}
<|assistant|>
'''

# Text generation pipeline with low temperature for factual adherence
generator = pipeline(
    task="text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",  # Or local Mistral/Llama via Ollama/vLLM
    max_new_tokens=120,
    temperature=0.1,  # Low temperature minimizes hallucination risk
    do_sample=False
)

generation_output = generator(rag_prompt)[0]["generated_text"]
answer = generation_output.split("<|assistant|>")[-1].strip()

print(f"\\nRetrieved Policy Chunk (Sim: {sim_score:.3f}):\\n{best_chunk}")
print(f"\\nVerified Grounded LLM Response:\\n{answer}")
"""
    }
}

