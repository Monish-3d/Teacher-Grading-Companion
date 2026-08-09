"""
FastAPI backend for Marks-Grader.

Wraps the existing RAG pipeline (app.py) and exposes it over HTTP so the
web frontend (in ./frontend) can call it directly:

    GET  /api/subjects        -> available subjects
    GET  /api/health          -> liveness + which subjects are warm
    POST /api/grade           -> grade a single question / answer
    POST /api/grade-sheet     -> grade a whole answer-sheet PDF (multimodal extraction)
    POST /api/generate-mcqs   -> difficulty-controlled MCQ generation

Run:
    pip install fastapi "uvicorn[standard]" python-multipart
    uvicorn server:app --port 8000        (or:  python run.py)

Then open http://127.0.0.1:8000
"""

import os
import re
import sys
import json
import tempfile
import threading
from typing import List

# On Windows, uvicorn's stdout defaults to the cp1252 codec, which cannot encode
# the emoji used in app.py's print() statements (e.g. "🔄 Pinecone index already
# exists") and would raise UnicodeEncodeError mid-request. Force UTF-8 so the
# existing pipeline's logging is harmless. Must run before `import app`.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

# Importing the pipeline module runs its module-level setup once:
# it loads the mpnet embedding model and initialises the Gemini LLM.
import app as pipeline

app = FastAPI(title="Marks-Grader API", version="1.0")

# Allow the frontend to call the API even if it is opened from a file:// origin
# or a different port during development. When served by this same app it is same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- config
# Textbooks shipped with the repo.
BUILTIN_SUBJECTS = {
    "se":   {"label": "Software Engineering",         "pdf": "subject_book.pdf"},
    "oops": {"label": "Object Oriented Programming",  "pdf": "OOP_book.pdf"},
}
UPLOAD_DIR = "uploads"          # user-uploaded textbook PDFs live here
REGISTRY_PATH = "subjects.json" # persists user-added subjects across restarts
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _load_registry():
    """Builtin subjects plus any the user has added in a previous session."""
    subjects = {k: dict(v) for k, v in BUILTIN_SUBJECTS.items()}
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                for sid, meta in json.load(f).items():
                    if sid not in BUILTIN_SUBJECTS:
                        subjects[sid] = meta
        except Exception:
            pass
    return subjects


SUBJECTS = _load_registry()


def _save_registry():
    uploaded = {k: v for k, v in SUBJECTS.items() if k not in BUILTIN_SUBJECTS}
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(uploaded, f, indent=2)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "subject"


# subject -> (semantic_store, keyword_store). Built lazily and cached so a textbook
# is only loaded / embedded once per process (mirrors Streamlit's @st.cache_resource).
_STORE_CACHE = {}
# subject -> "building" | "ready" | "error: ...". Only tracked while a fresh upload
# is being embedded; anything absent from here is treated as ready.
_STATUS = {}
# Serialises the (CPU-heavy) index builds so two requests can't build the same store.
_BUILD_LOCK = threading.Lock()


def get_stores(subject: str):
    if subject not in SUBJECTS:
        raise HTTPException(400, f"Unknown subject '{subject}'. Choose from {list(SUBJECTS)}.")
    if subject in _STORE_CACHE:
        return _STORE_CACHE[subject]
    with _BUILD_LOCK:
        if subject in _STORE_CACHE:            # built by another thread while we waited
            return _STORE_CACHE[subject]
        meta = SUBJECTS[subject]
        if not os.path.exists(meta["pdf"]):
            raise HTTPException(500, f"Textbook '{meta['pdf']}' is missing on the server.")
        raw = pipeline.load_pdf(meta["pdf"])
        chunks = pipeline.chunk_text(raw)
        _STORE_CACHE[subject] = pipeline.build_or_load_vectorstores(chunks, subject)
    return _STORE_CACHE[subject]


# --------------------------------------------------------------------------- schemas
class GradeRequest(BaseModel):
    subject: str
    question: str
    answer: str


class MCQRequest(BaseModel):
    subject: str
    topic: str
    difficulty: str = "medium"
    num_questions: int = Field(5, ge=1, le=10)


class MCQItem(BaseModel):
    question: str
    options: List[str]
    answer: str


class MCQSet(BaseModel):
    questions: List[MCQItem]


# Reuse the grader's Gemini model, but bind it to a validated MCQ schema.
# Crucially, MCQ retrieval below uses the SAME mpnet / per-subject namespace store
# as the grader, so it is consistent with how the index was actually built.
_mcq_model = pipeline.llm.with_structured_output(MCQSet)

MCQ_PROMPT = PromptTemplate(
    template=(
        "You are a {subject_label} professor writing an exam.\n"
        "Generate exactly {n} high-quality multiple-choice questions on the topic below.\n"
        "Ground them in the reference context from the textbook; use your own expertise only "
        "if the context is insufficient for the requested number of questions.\n\n"
        "Rules:\n"
        "- Each question MUST have exactly 4 options.\n"
        "- Exactly one option is correct.\n"
        "- Copy the full text of the correct option verbatim into 'answer'.\n"
        "- Calibrate the questions to the requested difficulty.\n\n"
        "Topic: {topic}\n"
        "Difficulty: {difficulty}\n"
        "Reference context:\n{context}"
    ),
    input_variables=["subject_label", "topic", "difficulty", "n", "context"],
)


# --------------------------------------------------------------------------- helpers
def _num(x, default=0.0):
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return default


def normalize_grade(result: dict, question: str, answer: str) -> dict:
    """Flatten the chain output into a stable JSON shape for the frontend."""
    return {
        "question": question,
        "answer": answer,
        "llm_score": _num(result.get("llm_score")),
        "similarity_score": _num(result.get("similarity_score")),
        "keyword_score": _num(result.get("keyword_score")),
        "final_score": _num(result.get("final_score")),
        "accuracy": _num(result.get("accuracy")),
        "feedback": result.get("feedback", ""),
    }


# --------------------------------------------------------------------------- routes
@app.get("/api/subjects")
def list_subjects():
    return [
        {"id": k, "label": v["label"],
         "status": _STATUS.get(k, "ready"),
         "builtin": k in BUILTIN_SUBJECTS}
        for k, v in SUBJECTS.items()
    ]


@app.post("/api/subjects")
async def add_subject(label: str = Form(...), file: UploadFile = File(...)):
    """Upload a textbook PDF and build a new gradable knowledge base from it.

    The (slow) embedding runs in a background thread; the client polls
    GET /api/subjects and watches this subject's 'status' flip to 'ready'.
    """
    label = label.strip()
    if not label:
        raise HTTPException(400, "A subject name is required.")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")

    base = _slugify(label)
    sid = base
    n = 2
    while sid in SUBJECTS:
        sid = f"{base}-{n}"
        n += 1

    pdf_path = os.path.join(UPLOAD_DIR, f"{sid}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(data)

    SUBJECTS[sid] = {"label": label, "pdf": pdf_path}
    _save_registry()
    _STATUS[sid] = "building"

    def _build():
        try:
            get_stores(sid)            # load -> chunk -> embed into Pinecone + BM25
            _STATUS[sid] = "ready"
        except Exception as e:         # noqa: BLE001 - surfaced to the client via status
            _STATUS[sid] = f"error: {e}"

    threading.Thread(target=_build, daemon=True).start()
    return {"id": sid, "label": label, "status": "building"}


@app.get("/api/health")
def health():
    return {"status": "ok", "subjects_warm": sorted(_STORE_CACHE.keys())}


@app.post("/api/grade")
def grade(req: GradeRequest):
    if not req.question.strip() or not req.answer.strip():
        raise HTTPException(400, "Both a question and an answer are required.")
    semantic_store, keyword_store = get_stores(req.subject)
    try:
        result = pipeline.chain.invoke({
            "question": req.question,
            "answer": req.answer,
            "subject": req.subject,
            "semantic_store": semantic_store,
            "keyword_store": keyword_store,
        })
    except Exception as e:  # noqa: BLE001 - surface a clean error to the UI
        raise HTTPException(500, f"Grading failed: {e}")
    return normalize_grade(result, req.question, req.answer)


@app.post("/api/grade-sheet")
async def grade_sheet(subject: str = Form(...), file: UploadFile = File(...)):
    semantic_store, keyword_store = get_stores(subject)

    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        qa_pairs = pipeline.extract_student_answers(tmp_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Could not read the answer sheet: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not qa_pairs:
        raise HTTPException(422, "No question-answer pairs were detected in that PDF.")

    results = []
    for qa in qa_pairs:
        q = (qa.get("question") or "").strip()
        a = (qa.get("answer") or "").strip()
        if not q and not a:
            continue
        try:
            r = pipeline.chain.invoke({
                "question": q,
                "answer": a,
                "subject": subject,
                "semantic_store": semantic_store,
                "keyword_store": keyword_store,
            })
            results.append(normalize_grade(r, q, a))
        except Exception as e:  # noqa: BLE001 - keep grading the rest of the sheet
            results.append({"question": q, "answer": a, "error": str(e)})

    scored = [r["final_score"] for r in results if "final_score" in r]
    average = round(sum(scored) / len(scored), 2) if scored else 0.0
    return {"count": len(results), "average": average, "results": results}


@app.post("/api/generate-mcqs")
def generate_mcqs(req: MCQRequest):
    if not req.topic.strip():
        raise HTTPException(400, "A topic is required.")
    semantic_store, _ = get_stores(req.subject)

    retriever = semantic_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, "filter": {"subject": req.subject}},
    )
    docs = retriever.invoke(req.topic)
    context = "\n\n".join(d.page_content for d in docs)

    prompt_str = MCQ_PROMPT.format(
        subject_label=SUBJECTS[req.subject]["label"],
        topic=req.topic,
        difficulty=req.difficulty,
        n=req.num_questions,
        context=context,
    )
    try:
        out = _mcq_model.invoke(prompt_str)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"MCQ generation failed: {e}")

    if hasattr(out, "questions"):
        questions = out.questions
        items = [{"question": q.question, "options": list(q.options), "answer": q.answer} for q in questions]
    elif isinstance(out, dict):
        items = out.get("questions", [])
    else:
        items = []

    return {"topic": req.topic, "difficulty": req.difficulty, "questions": items}


# The frontend is served from the same origin so the browser needs no CORS at all.
# This mount must come AFTER the API routes so /api/* is matched first.
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
