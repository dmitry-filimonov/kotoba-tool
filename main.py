import streamlit as st
import re
import subprocess
from translate import Translator
from transformers import BertTokenizer
from sklearn.metrics import f1_score

# === Кастомный токенизатор ===
class CustomTokenizer:
    def __init__(self):
        self.kanji_pattern = re.compile(r'([一-龯]+)')
        self.bert_tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    def tokenize(self, text):
        tokens = []
        for word in self.kanji_pattern.split(text):
            if not word:
                continue
            if self.kanji_pattern.match(word):
                # Используем BERT для контекстного разделения
                bert_tokens = self.bert_tokenizer.tokenize(word)
                tokens.extend(bert_tokens)
            else:
                tokens.append(word)
        return tokens

# === Функции для работы с MeCab ===
def mecab_tokenize(text):
    try:
        result = subprocess.run(
            ['mecab', '-Owakati'],
            input=text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return result.stdout.decode('utf-8').strip().split()
        else:
            return ["MeCab error: " + result.stderr.decode('utf-8')]
    except Exception as e:
        return [f"Error: {str(e)}"]

# === Основной код приложения ===
st.title("Japanese NLP Analyzer")

# Ввод текста
english_text = st.text_input("Enter English text:", "I study Japanese linguistics.")

if english_text:
    # Перевод на японский
    translator = Translator(to_lang="ja")
    japanese_text = translator.translate(english_text)
    st.markdown(f"**Japanese translation:** `{japanese_text}`")

    # Вкладки для результатов
    tab1, tab2, tab3 = st.tabs(["Tokenization", "Visualization", "Evaluation"])

    # Инициализация токенизаторов
    custom_tokenizer = CustomTokenizer()
    tokenizers = {
        "MeCab": mecab_tokenize,
        "Custom BERT-based": custom_tokenizer.tokenize
    }

    with tab1:
        st.header("Tokenization Comparison")
        for name, func in tokenizers.items():
            with st.expander(name, expanded=True):
                tokens = func(japanese_text)
                st.write("**Tokens:**", " | ".join(tokens))

    with tab2:
        st.header("Character Type Visualization")
        def highlight_text(text):
            highlighted = []
            for char in text:
                if re.match(r'[一-龯]', char):
                    highlighted.append(f"<span style='color:red'>{char}</span>")
                elif re.match(r'[ァ-ヺ]', char):
                    highlighted.append(f"<span style='color:blue'>{char}</span>")
                elif re.match(r'[ぁ-ゟ]', char):
                    highlighted.append(f"<span style='color:green'>{char}</span>")
                else:
                    highlighted.append(char)
            return ''.join(highlighted)

        html = highlight_text(japanese_text)
        st.markdown(html, unsafe_allow_html=True)

    with tab3:
        st.header("Model Evaluation")
        gold_file = st.file_uploader("Upload gold standard tokens (CSV)", type=['csv', 'txt'])
        
        if gold_file:
            gold_tokens = gold_file.read().decode().strip().split(',')
            st.write("**Gold tokens:**", " | ".join(gold_tokens))
            
            for name, func in tokenizers.items():
                predicted = func(japanese_text)
                f1 = f1_score(gold_tokens, predicted, average='micro')
                st.write(f"**{name} F1 Score:** {f1:.2f}")

# Инструкции по установке
with st.sidebar:
    st.header("Setup Instructions")
    st.markdown("""
    1. **For local setup:**
       ```bash
       # Установка MeCab
       sudo apt install mecab libmecab-dev mecab-ipadic-utf8
       pip install -r requirements.txt
       ```
    2. **requirements.txt:**
       ```
       streamlit
       translate
       transformers
       scikit-learn
       ```
    """)