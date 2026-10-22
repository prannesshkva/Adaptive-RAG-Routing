import re, nltk
from fastapi import APIRouter
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

for pkg, path in [
    ('punkt', 'tokenizers/punkt'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('wordnet', 'corpora/wordnet'),
    ('stopwords', 'corpora/stopwords')
]:
    try:
        nltk.data.find(path)
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

from app.models.schemas import Query, Document

router = APIRouter(prefix="/vlrag", tags=["VLRAG"])
rag = None

class vectorless_rag:
    def __init__(self, documents):
        self.raw_text = documents
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = set()
        self.chunk_size = 200
        self.overlap = 50
        self.raw_chunks = []
        self.documents = self.preprocess(self.raw_text)
        self.model = BM25Okapi(self.documents) if self.documents else None

    def pos(self, tag):
        if tag.startswith("J"): return wordnet.ADJ
        elif tag.startswith("V"): return wordnet.VERB
        elif tag.startswith("R"): return wordnet.ADV
        else: return wordnet.NOUN

    def clean(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def preprocess(self, content):
        content = self.clean(content)
        words = content.split()
        chunks = []
        self.raw_chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(words), step):
            raw_chunk = " ".join(words[i:i + self.chunk_size])
            curr = words[i:i + self.chunk_size]
            curr_filtered = [w for w in curr if w not in self.stop_words]
            if not curr_filtered:
                curr_filtered = curr
            try:
                pos_tags = nltk.pos_tag(curr_filtered)
                processed = [self.lemmatizer.lemmatize(w, self.pos(p)) for w, p in pos_tags]
            except Exception:
                processed = curr_filtered
            chunks.append(processed)
            self.raw_chunks.append(raw_chunk)
        return chunks

    def process_query(self, query):
        query_words = self.clean(query).split()
        filtered = [w for w in query_words if w not in self.stop_words]
        if not filtered:
            filtered = query_words
        try:
            pos_tags = nltk.pos_tag(filtered)
            return [self.lemmatizer.lemmatize(w, self.pos(p)) for w, p in pos_tags]
        except Exception:
            return filtered

    def retrieve(self, query, limit=5):
        if not self.model or not self.documents:
            return []
        query_tokens = self.process_query(query)
        scores = self.model.get_scores(query_tokens)
        ranked = sorted(zip(self.raw_chunks, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:limit] if score > 0 or len(ranked) == 1]

@router.post("/init")
async def init():
    global rag
    rag = None
    return {"status": "initialized"}

@router.post("/add")
async def add(doc: Document):
    global rag
    rag = vectorless_rag(doc.text)
    return {"status": "added", "chunks": len(rag.documents)}

@router.post("/query")
async def query(q: Query):
    if rag is None:
        return {"data": []}
    results = rag.retrieve(q.message, 5)
    return {"data": results}
