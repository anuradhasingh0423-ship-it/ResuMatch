import spacy
import sqlite3, re, json
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# ---------- NLP ----------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



app = Flask(__name__)
app.secret_key = "resumatch_secret_key"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")

def get_db():
    return sqlite3.connect(DB_PATH)

# 2. Add the initialization function
def init_db_on_start():
    db = get_db()
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, email TEXT UNIQUE, password TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS history(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, detected_role TEXT, score REAL, analysis TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    db.commit()
    db.close()


# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer(stop_words="english")

# ---------- Scoring Config (Deploy-ready) ----------
SCORING_CONFIG = {
    "skill_weight": 30,
    "keyword_weight": 3,
    "experience_weight": 2,
    "length_weight": 10,
    "similarity_weight": 20,
    "shortlist_threshold": 70,
    "borderline_threshold": 55
}

# ---------- Skill Map ----------
SKILL_MAP = {
    "machine learning": ["ml", "svm", "knn", "random forest", "naive bayes"],
    "deep learning": ["cnn", "tensorflow", "pytorch", "keras"],
    "computer vision": ["opencv", "image processing", "cv"],
    "data analysis": ["pandas", "numpy", "eda", "sql"],
    "evaluation": [
        "evaluation", "evaluate", "evaluating",
        "accuracy", "precision", "recall", "f1", "metric", "score"
    ],
    "nlp": ["tf idf", "sentiment analysis", "text processing", "cosine similarity"]
}


WEB_SKILL_MAP = {
    "html": ["html5"],
    "css": ["css3"],
    "javascript": ["js", "react", "react.js"],
    "ui ux": ["figma", "adobe xd", "canva", "ux", "ui"],
    "responsive design": ["bootstrap", "media queries"],
    "web frameworks": ["react", "wordpress", "wix"]
}

HR_SKILL_MAP = {
    "recruitment": ["hiring", "talent acquisition"],
    "communication": ["verbal", "written"],
    "employee engagement": ["engagement", "retention"],
    "payroll": ["salary", "ctc"],
    "hr tools": ["hrms", "ats"]
}

DEVOPS_SKILL_MAP = {
    "docker": ["container"],
    "kubernetes": ["k8s"],
    "cloud": ["aws", "azure", "gcp"],
    "ci cd": ["jenkins", "github actions"],
    "monitoring": ["prometheus", "grafana"],
    "linux": ["bash", "shell"]
}

PYTHON_DEV_SKILL_MAP = {
    "python": ["core python", "pandas", "numpy"], # Keep it specific
    "flask": ["flask api", "werkzeug"],
    "django": ["django rest", "orm"],
    "backend": ["fastapi", "python backend"] # Make these more specific
}

AUTOMATION_SKILL_MAP = {
    "automation workflows": ["n8n", "zapier", "make.com"],
    "messaging api": ["whatsapp api", "twilio"],
    "social automation": ["instagram api", "facebook api"],
    "ai tools": ["claude", "openai", "chatgpt"],
    "crm": ["crm", "client management"],
    "marketing": ["digital marketing", "lead generation"]
}

JAVA_DEV_SKILL_MAP = {
    "java": ["j2ee", "core java", "jdk", "jre"],
    "spring": ["spring boot", "spring mvc", "hibernate", "microservices"],
    "apis": ["restful api", "json", "soap", "endpoints"],
    "databases": ["sql", "mysql", "postgresql", "oracle", "jdbc"],
    "testing": ["junit", "mockito", "tdd", "assertion"],
    "tools": ["maven", "gradle", "git", "intellij"]
}

ROLE_SKILL_MAP = {
    "Data Science": SKILL_MAP,
    "Web Designing": WEB_SKILL_MAP,
    "HR": HR_SKILL_MAP,
    "DevOps Engineer": DEVOPS_SKILL_MAP,
    "Python Developer": PYTHON_DEV_SKILL_MAP,
    "Java Developer": JAVA_DEV_SKILL_MAP ,
    "AI Automation Engineer": AUTOMATION_SKILL_MAP
}



# ---------- Job Descriptions ----------
job_descriptions = {
    "Data Science": """
        Python SQL Pandas NumPy Statistics Probability Machine Learning Deep Learning NLP Computer Vision
        Data Analysis EDA Feature Engineering Scikit-learn TensorFlow Keras PyTorch
        Model Training Evaluation Cross Validation Deployment Flask API Pipeline Git Kaggle Research 
    """,

    "HR": "Recruitment Talent Acquisition Payroll Employee Engagement Communication Performance Management",

    "DevOps Engineer": "Docker Kubernetes AWS CI CD Linux Automation Cloud Infrastructure Monitoring",

    "Web Designing": "HTML CSS JavaScript UI UX Responsive Design Bootstrap Figma",

    "Python Developer": "Python OOP Flask Django APIs Databases Backend Development",
    
    "AI Automation Engineer": """
    n8n automation workflows WhatsApp Business API Instagram Facebook automation
    CRM client management AI tools Claude OpenAI chatbots digital marketing
    """,
                               
    "Java Developer": """
    Java J2EE Spring Boot Hibernate RESTful APIs Microservices SQL MySQL 
    PostgreSQL JUnit Mockito Maven Gradle Git Agile Cloud Computing
    """
}




def role_skill_confidence(resume, role):
    skills = ROLE_SKILL_MAP[role]
    resume = clean_text(resume)
    confidence = {}

    for parent, children in skills.items():
        score = 0
        if parent in resume:
            score += 0.6
        if any(c in resume for c in children):
            score += 0.4
        confidence[parent] = round(min(score,1.0)*100,1)

    return confidence

# ---------- Utils ----------
def clean_text(text):
    doc = nlp(text.lower())
    return " ".join(
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    )

def normalize_skills(text, role=None):
    text = clean_text(text)
    skill_map = ROLE_SKILL_MAP.get(role, SKILL_MAP)
    for parent, children in SKILL_MAP.items():
        found = False

        if parent in text:
            found = True

        if any(c in text for c in children):
            found = True

        # stem handling (evaluate / evaluation / evaluating)
        if parent.startswith("evaluat") and "evaluat" in text:
            found = True

        if found and parent not in text:
            text += " " + parent

    return text

def text_similarity(a, b):
    tfidf = vectorizer.fit_transform([a, b])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def skill_match(resume, jd, role):
    resume = normalize_skills(resume)
    jd = normalize_skills(jd)

    skills = ROLE_SKILL_MAP[role]
    matched = 0
    total = 0

    for parent in skills:
        if parent in jd:
            total += 1
            if parent in resume:
                matched += 1

    return matched / total if total else 0


def keyword_overlap(resume, jd):
    return len(set(normalize_skills(resume).split()) &
               set(normalize_skills(jd).split()))

def resume_length_score(resume):
    wc = len(clean_text(resume).split())
    return 1 if 100 <= wc <= 800 else 0

def experience_score(resume):
    keywords = ["experience","intern","internship","project","worked","company"]
    text = clean_text(resume)
    return min(sum(1 for k in keywords if k in text) * 2, 10)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    raw_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += "\n" + text

    # Basic cleanup
    raw_text = raw_text.replace("\n", " ")
    raw_text = re.sub(r"\s+", " ", raw_text)

    # 🔥 CRITICAL: re-run through spaCy to rebuild broken tokens
    doc = nlp(raw_text)
    rebuilt_text = " ".join(token.text for token in doc)

    return rebuilt_text.strip()



# ---------- Advanced Features ----------
def skill_confidence(resume):
    resume = normalize_skills(resume)
    confidence = {}

    for parent, children in SKILL_MAP.items():
        score = 0
        if parent in resume:
            score += 0.6
        if any(c in resume for c in children):
            score += 0.4
        confidence[parent] = round(min(score, 1.0) * 100, 1)

    return confidence

def explain_decision(scorecard):
    reasons = []
    
    # Positive Reasons
    if scorecard["skill_match_percent"] >= 75:
        reasons.append("Strong alignment with required technical skills")
    if scorecard["experience_score"] >= 7:
        reasons.append("Relevant hands-on project and internship experience")
    if scorecard["overall_score"] >= 85:
        reasons.append("High overall ATS score")

    # Negative/Improvement Reasons (New)
    if not reasons:
        if scorecard["overall_score"] < 50:
            reasons.append("Low alignment with the specific technical stack of this role")
        if scorecard["experience_score"] < 5:
            reasons.append("Consider adding more industry-specific projects or internships")
            
    return reasons

# ---------- Core Logic ----------
def generate_scorecard(resume, role):
    jd = job_descriptions[role]

    skill = skill_match(resume, jd, role)
    keyword = keyword_overlap(resume, jd)
    exp = experience_score(resume)
    length = resume_length_score(resume)

    raw_sim = text_similarity(normalize_skills(resume), normalize_skills(jd))
    similarity = max(raw_sim, 0.45) if exp >= 6 else raw_sim

    overall = (
        skill * SCORING_CONFIG["skill_weight"] +
        min(keyword,10) * SCORING_CONFIG["keyword_weight"] +
        exp * SCORING_CONFIG["experience_weight"] +
        length * SCORING_CONFIG["length_weight"] +
        similarity * SCORING_CONFIG["similarity_weight"]
    )

    return {
        "skill_match_percent": round(skill * 100, 2),
        "keyword_match_score": keyword,
        "experience_score": exp,
        "resume_length": "Good" if length else "Needs Improvement",
        "similarity_score": round(similarity, 2),
        "overall_score": round(overall, 2)
    }

def detect_missing_skills(resume, role, is_pdf=False):
    confidence = skill_confidence(resume)
    jd_skills = normalize_skills(job_descriptions[role])

    missing = []
    for skill in ROLE_SKILL_MAP[role]:
        if confidence.get(skill, 0) >= 50:
            continue
        if is_pdf:
            continue
        if skill not in jd_skills:
            continue
        missing.append(skill)


    return missing[:10]



def detect_job_role(resume):
    resume = normalize_skills(resume)
    scores = {
        role: text_similarity(resume, normalize_skills(jd))
        for role, jd in job_descriptions.items()
    }
    best = max(scores, key=scores.get)
    return best, scores[best], scores

# ---------- Auth ----------
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        db = get_db()
        cur = db.cursor()
        cur.execute("SELECT id,password FROM users WHERE email=?",
                    (request.form["email"],))
        row = cur.fetchone()
        if row and check_password_hash(row[1], request.form["password"]):
            session["user_id"] = row[0]
            return redirect(url_for("home"))
        return render_template("auth.html", error="Invalid credentials")
    return render_template("auth.html")

@app.route("/signup", methods=["POST"])
def signup():
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute(
            "INSERT INTO users(username,email,password) VALUES(?,?,?)",
            (request.form["username"],
             request.form["email"],
             generate_password_hash(request.form["password"]))
        )
        db.commit()
    except:
        return render_template("auth.html", error="Email already exists")
    return redirect(url_for("login"))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------- Pages ----------
@app.route("/")
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ---------- Predict ----------
@app.route("/predict", methods=["POST"])
def predict():
    is_pdf = "resume_pdf" in request.files
    
    

    resume_text = (
    extract_text_from_pdf(request.files["resume_pdf"])
    if "resume_pdf" in request.files
    else request.form.get("resume_text","")
    )

    if not resume_text.strip():
        return jsonify({"error":"Resume required"}),400

    # ✅ detect role FIRST
    role, _, _ = detect_job_role(resume_text)

    job_desc_text = request.form.get("job_description", "")
    jd_analysis = None
    if job_desc_text.strip():
        jd_analysis = compare_resume_with_jd(resume_text, job_desc_text, role)

    skill_conf = role_skill_confidence(resume_text, role)

    scorecard = generate_scorecard(resume_text, role)


   # ... inside predict() after scorecard is generated ...

    # Combined decision logic to avoid overwriting
    if jd_analysis and jd_analysis["jd_similarity"] < 0.2:
        if jd_analysis["jd_similarity"] < 0.15:
            if scorecard["overall_score"] >= 80:
                decision = "Likely Shortlisted (JD is generic)"
            elif scorecard["overall_score"] >= 60:
                decision = "Borderline (JD mismatch)"
            else:
                decision = "Not a Fit for This Role"
        else:
            decision = "Not a Fit for This Role"
    else:
        # Standard scoring thresholds
        if scorecard["overall_score"] >= SCORING_CONFIG["shortlist_threshold"]:
            decision = "Likely Shortlisted"
        elif scorecard["overall_score"] >= SCORING_CONFIG["borderline_threshold"]:
            decision = "Borderline"
        else:
            decision = "Needs Improvement"

 
    analysis = {
        "detected_role": role,
        "decision": decision,
        "scorecard": scorecard,
        "missing_skills": detect_missing_skills(resume_text, role, is_pdf),


        "skill_confidence": skill_conf,

        "why_shortlisted": explain_decision(scorecard),
        "jd_analysis": jd_analysis,

        "suggestions": ["Great resume!"] if decision == "Likely Shortlisted"
                       else ["Focus on learning role-specific frameworks", "Highlight more relevant projects"]
    }

    if "user_id" in session:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO history(user_id, detected_role, score, analysis) VALUES(?,?,?,?)",
            (session["user_id"], role, scorecard["overall_score"], json.dumps(analysis))
        )
        db.commit()

    return jsonify(analysis)

def role_missing_skills(resume, role, is_pdf=False):
    if is_pdf:
        return []

    skills = ROLE_SKILL_MAP[role]
    resume = clean_text(resume)

    missing = []
    for skill in skills:
        if skill not in resume:
            missing.append(skill)

    return missing[:10]


def compare_resume_with_jd(resume, jd, role):
    resume_norm = normalize_skills(resume)
    jd_norm = normalize_skills(jd)

    skill_match_pct = skill_match(resume_norm, jd_norm, role) * 100
    keyword_score = keyword_overlap(resume_norm, jd_norm)
    similarity = text_similarity(resume_norm, jd_norm)

    missing = []
    for skill in ROLE_SKILL_MAP[role]:
        if skill in jd_norm and skill not in resume_norm:
            missing.append(skill)

    return {
        "jd_skill_match_percent": round(skill_match_pct, 2),
        "jd_keyword_score": keyword_score,
        "jd_similarity": round(similarity, 2),
        "jd_missing_skills": missing[:10]
    }


# ---------- History ----------
@app.route("/history")
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute("""
        SELECT id, detected_role, score, created_at
        FROM history WHERE user_id=?
        ORDER BY created_at DESC
    """, (session["user_id"],))
    return jsonify([
        {"id":r[0],"detected_role":r[1],"score":round(r[2],1),"created_at":r[3]}
        for r in cur.fetchall()
    ])

@app.route("/history/<int:hid>")
def get_history(hid):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT analysis, created_at FROM history WHERE id=? AND user_id=?",
        (hid, session["user_id"])
    )
    row = cur.fetchone()
    if not row:
        return jsonify({"error":"Not found"}),404
    data = json.loads(row[0])
    data["created_at"] = row[1]
    return jsonify(data)

@app.route("/history/<int:hid>", methods=["DELETE"])
def delete_history(hid):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM history WHERE id=? AND user_id=?",
                (hid, session["user_id"]))
    db.commit()
    return jsonify({"success":True})



if __name__ == "__main__":
    # 1. Initialize DB before starting
    with app.app_context():
        init_db_on_start()
        
    # 2. Use the dynamic port assigned by Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


