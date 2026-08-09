"""Convenience launcher: `python run.py` then open http://127.0.0.1:8000"""
import uvicorn

if __name__ == "__main__":
    # reload=False so the heavy pipeline (embedding model + LLM) loads only once.
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
