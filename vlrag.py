import re,nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from fastapi import APIRouter
from pydantic import BaseModel
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

class Query(BaseModel):
    message:str
class Document(BaseModel):
    text:str

router=APIRouter(prefix="/vlrag",tags=["VLRAG"])
rag=None

class vectorless_rag:
    def __init__(self,documents):
        self.raw_text=documents
        self.lemmatizer=WordNetLemmatizer()
        self.stop_words=set(stopwords.words("english"))
        self.chunk_size=200
        self.overlap=50
        self.documents=self.preprocess(self.raw_text)
        self.model=BM25Okapi(self.documents)

    def pos(self,tag):
        if tag.startswith("J"): return wordnet.ADJ
        elif tag.startswith("V"): return wordnet.VERB
        elif tag.startswith("R"): return wordnet.ADV
        else: return wordnet.NOUN

    def clean(self,text):
        text=text.lower()
        text=re.sub(r'[^a-zA-Z0-9 ]',' ',text)
        text=re.sub(r'\s+',' ',text)
        return text.strip()

    def preprocess(self,content):
        content=self.clean(content)
        words=content.split()
        chunks=[]
        step=self.chunk_size-self.overlap
        for i in range(0,len(words),step):
            curr=words[i:i+self.chunk_size]
            curr=[w for w in curr if w not in self.stop_words]
            if not curr: continue
            pos_tags=nltk.pos_tag(curr)
            processed=[self.lemmatizer.lemmatize(w,self.pos(p)) for w,p in pos_tags]
            chunks.append(processed)
        return chunks

    def process_query(self,query):
        query=self.clean(query).split()
        query=[w for w in query if w not in self.stop_words]
        pos_tags=nltk.pos_tag(query)
        return [self.lemmatizer.lemmatize(w,self.pos(p)) for w,p in pos_tags]

    def retrieve(self,query,limit=5):
        query_tokens=self.process_query(query)
        scores=self.model.get_scores(query_tokens)
        ranked=sorted(zip(self.documents,scores),key=lambda x:x[1],reverse=True)
        return [" ".join(doc) for doc,_ in ranked[:limit]]

@router.post("/init")
async def init():
    global rag
    rag=None

@router.post("/add")
async def add(doc:Document):
    global rag
    rag=vectorless_rag(doc.text)

@router.post("/query")
async def query(q:Query):
    if rag is None:
        return {"data":["No data loaded"]}
    results=rag.retrieve(q.message,5)
    return {"data":results}