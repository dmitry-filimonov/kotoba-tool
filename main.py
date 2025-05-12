import streamlit as st
import re
import subprocess
import pandas as pd
import unicodedata
from transformers import BertTokenizer
from deep_translator import GoogleTranslator
import pykakasi
from sudachipy import dictionary as sudachi_dict, tokenizer as sudachi_tokenizer
from janome.tokenizer import Tokenizer as JTokenizer
import json

# ==== CustomTokenizer with Sudachi-based kanji segmentation & ASCII preservation ====

class CustomTokenizer:
    def __init__(self):
        self.bert = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        self.particles = [
            'は','が','を','に','で','と','も','へ','の',
            'から','まで','より','ばかり','だけ','ほど','など',
            'や','ね','よ','ぞ','ぜ'
        ]
        sudachi_obj = sudachi_dict.Dictionary().create()
        self.sudachi = sudachi_obj.tokenize
        self._is_kanji    = re.compile(r'^[一-龯]+$').fullmatch
        self._is_japanese = re.compile(r'^[ぁ-んァ-ン一-龯ー]+$').fullmatch
        self._is_ascii    = re.compile(r'^[A-Za-z0-9]+$').fullmatch

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        return re.sub(r'(ｗ|ー)\1+', r'\1', text)

    def tokenize(self, text: str) -> list[str]:
        text = self.normalize(text)
        # 1) split by script type
        tokens, buf, cur = [], "", None
        def ctype(ch):
            if re.match(r'[一-龯]', ch): return 'kanji'
            if re.match(r'[ぁ-ん]', ch): return 'hiragana'
            if re.match(r'[ァ-ン]', ch): return 'katakana'
            if re.match(r'[A-Za-z0-9]', ch): return 'ascii'
            return 'other'
        for ch in text:
            t = ctype(ch)
            if cur is None or t == cur:
                buf += ch
            else:
                tokens.append(buf)
                buf = ch
            cur = t
        if buf:
            tokens.append(buf)
        # 2) Sudachi segmentation for pure-kanji blocks
        segs = []
        for tok in tokens:
            if self._is_kanji(tok):
                segs += [m.surface() for m in self.sudachi(
                    tok,
                    mode=sudachi_tokenizer.Tokenizer.SplitMode.A
                )]
            else:
                segs.append(tok)
        # 3) final pass: keep pure Japanese or ASCII, else WordPiece
        final = []
        for tok in segs:
            if (self._is_kanji(tok)
                or self._is_japanese(tok)
                or self._is_ascii(tok)):
                final.append(tok)
            else:
                final += self.bert.tokenize(tok)
        # 4) detach particles
        out = []
        for tok in final:
            matched = False
            for p in sorted(self.particles, key=len, reverse=True):
                if tok.endswith(p) and len(tok) > len(p):
                    out += [tok[:-len(p)], p]
                    matched = True
                    break
            if not matched:
                out.append(tok)
        return [t for t in out if t.strip()]

# ==== Other tokenizers ====

def sudachi_tokenize(text: str) -> list[str]:
    sud = sudachi_dict.Dictionary().create().tokenize
    return [m.surface() for m in sud(
        text, mode=sudachi_tokenizer.Tokenizer.SplitMode.C
    )]

_janome = JTokenizer()
def janome_tokenize(text: str) -> list[str]:
    return [tok.surface for tok in _janome.tokenize(text)]

# ==== Romaji fallback ====

_kks = pykakasi.kakasi()
def fallback_romaji(tok: str) -> str:
    return "".join(item["hepburn"] for item in _kks.convert(tok))

# ==== Helpers ====

ja2en = GoogleTranslator(source="ja", target="en")

TYPE_COLOR = {
    "kanji":   "#e53935",
    "hiragana":"#43a047",
    "katakana":"#1e88e5",
    "latin":   "#757575",
    "digit":   "#fb8c00",
    "other":   "#000000",
}
def char_type(c: str) -> str:
    if re.match(r'[一-龯]', c):      return "kanji"
    if re.match(r'[ぁ-ん]', c):      return "hiragana"
    if re.match(r'[ァ-ン]', c):      return "katakana"
    if re.match(r'[A-Za-z0-9]', c):  return "latin"
    return "other"

# ==== Instantiate tokenizers ====

tokenizers = {
    "Sudachi": sudachi_tokenize,
    "Janome":  janome_tokenize,
    "Custom":  CustomTokenizer().tokenize,
}

# ==== Load Examples ====

with open("ml_curated_examples.json", encoding="utf-8") as f:
    EXAMPLES = json.load(f)

# ==== Streamlit UI ====

st.set_page_config(page_title="Japanese Tokenization Demo", layout="wide")

st.sidebar.title("About this Demo")
st.sidebar.markdown("""
This demo compares three Japanese tokenizers:
- **SudachiPy** (SplitMode.C)
- **Janome** (pure-Python)
- **Custom** (Sudachi + ASCII-preserve + BERT subwords)

Select one of the prepared sentences (varied styles). You will see:

1. **Tokenization F1** — overlap with gold tokens.  
2. **Morpheme Preservation** — recall of predefined morphemes.  
3. **Entity Preservation** — percent of named entities kept intact.  
4. **Back-translation Accuracy** — overlap between the original English and the back-translated text.

Below, each tokenizer’s segmentation is visualized with glosses or romaji, colored by script type.
""")

st.title("Japanese Tokenization Demo")

choice = st.selectbox("Select a sentence:", list(EXAMPLES.keys()))
case = EXAMPLES[choice]
jp = case["text"]
style = case["style"]
gold_tokens = set(case["targets"])
morphemes = set(case.get("morphemes", case["targets"]))
entities = set(case.get("entities", []))

st.markdown(f"**Japanese ({style}):** {jp}")

# ---- Segmentation Visualization ----
st.markdown("### Segmentation")
for name, fn in tokenizers.items():
    st.markdown(f"**{name}**:")
    toks = fn(jp)
    cols = st.columns(len(toks)) if toks else []
    for i, tkn in enumerate(toks):
        gloss = ja2en.translate(tkn) or fallback_romaji(tkn)
        color = TYPE_COLOR[char_type(tkn[0])]
        with cols[i]:
            st.markdown(f"""
                <div style="text-align:center; line-height:1.2em;">
                  <div style="font-size:10px; color:gray;">{gloss}</div>
                  <div style="font-size:18px; color:{color};">{tkn}</div>
                </div>""",
                unsafe_allow_html=True
            )

# ---- Evaluation Metrics ----
st.markdown("### Evaluation Metrics")

# Back-translation (once)
back_en = GoogleTranslator(source="ja", target="en")\
    .translate(jp).lower().split()
orig_set = set(choice.lower().split())
bt_acc = len(set(back_en) & orig_set) / len(orig_set) if orig_set else 0
st.markdown(f"**Back-translation accuracy:** {bt_acc:.1%}")

# Build table
rows = []
for name, fn in tokenizers.items():
    toks = fn(jp)
    pset = set(toks)

    # 1) Tokenization F1
    tp = len(pset & gold_tokens)
    fp = len(pset - gold_tokens)
    fn_ = len(gold_tokens - pset)
    f1 = 2*tp/(2*tp + fp + fn_) if (tp or fp or fn_) else 1.0

    # 2) Morpheme Preservation (recall)
    mp = len(pset & morphemes) / len(morphemes) if morphemes else None

    # 3) Entity Preservation (intact entities)
    ep = len(pset & entities) / len(entities) if entities else None

    rows.append({
        "Tokenizer": name,
        "F1 Score": round(f1, 2),
        "Morpheme Recall": f"{mp:.2%}" if mp is not None else "—",
        "Entity Recall": f"{ep:.2%}" if ep is not None else "—"
    })

st.dataframe(pd.DataFrame(rows))
