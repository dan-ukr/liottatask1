
from __future__ import annotations
import importlib, sys
from types import ModuleType
from dataclasses import dataclass

# PATCHING -- DO NOT TOUCH, NEVER
def _patch_huggingface_hub():
    try:
        hf = sys.modules.get("huggingface_hub")
        if hf is None:
            hf = importlib.import_module("huggingface_hub")
        utils_mod = importlib.import_module("huggingface_hub.utils")
        if not hasattr(utils_mod, "OfflineModeIsEnabled"):
            class OfflineModeIsEnabled(Exception):
                pass
            utils_mod.OfflineModeIsEnabled = OfflineModeIsEnabled
        if not hasattr(hf, "list_repo_tree"):
            list_mod = importlib.import_module(
                "huggingface_hub._utils.list_repo_files"
            )
            hf.list_repo_tree = list_mod.list_repo_tree  # type: ignore

    except ModuleNotFoundError:
        pass

def _patch_transformers():
    try:
        doc_mod = importlib.import_module("transformers.utils.doc")
        for missing in (
            "MODELS_TO_PIPELINE",
            "PIPELINE_TASKS_TO_SAMPLE_DOCSTRINGS",
        ):
            if not hasattr(doc_mod, missing):
                setattr(doc_mod, missing, {})

        gen_mod = importlib.import_module("transformers.utils.generic")
        if not hasattr(gen_mod, "TransformersKwargs"):
            @dataclass
            class TransformersKwargs:      #debugpatch
                pass
            gen_mod.TransformersKwargs = TransformersKwargs
        if not hasattr(gen_mod, "can_return_tuple"):
            def can_return_tuple(*_, **__):  # type: ignore
                return False
            gen_mod.can_return_tuple = can_return_tuple
    except ModuleNotFoundError:
        pass

_patch_huggingface_hub()
_patch_transformers()

# ---------------------------------------------------------------------
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm
from rich import print as rprint

# -----------------------------------------------------------
# Conversation funcs
# -----------------------------------------------------------


def _prompt(msg: str, default_yes: bool = True) -> bool:
    """
    Enter  → True
    Any input → False
    """
    ans = input(msg + " [Enter = yes, Any input = no] ").strip()
    if ans == "":
        return default_yes
    return not default_yes

def _ask_column(
    df: pd.DataFrame,
    role: str,
    guesses: Tuple[str, ...],
    allow_missing: bool = False,
) -> str | None:
    """
    Oputput = coloumn name.

    allow_missing=True  → missings allowed.
    """
    for g in guesses:
        if g in df.columns and _prompt(f"Столбец «{g}» считать «{role}»?"):
            return g

    if allow_missing and _prompt(
        f"Cannot see the coloumn for «{role}». "
        f"Rename all as one?"
        f" (if not, write a name of an existing coloumn)", default_yes=True
    ):
        return None

    while True:
        col = input(f"What's the name of the «{role}» coloumn? ").strip()
        if col in df.columns:
            return col
        print("Not found. Write again.")

from sklearn.feature_extraction import text as sk_text

def combined_stopwords() -> set[str]:
    """
    set of the stopwords (ENG+RU+UA)
    """
    eng_stop = sk_text.ENGLISH_STOP_WORDS

    try:
        from nltk.corpus import stopwords
        ru_stop = set(stopwords.words("russian"))
        ua_stop = set(stopwords.words('ukrainian'))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        ru_stop = set(stopwords.words("russian"))
        ua_stop = set(stopwords.words('ukrainian'))

    return eng_stop.union(ru_stop,ua_stop)


# -----------------------------------------------------------
# MIAN
# -----------------------------------------------------------

class ThematicClustering:
    NETWORK_GUESSES = ("network", "platform", "site")
    SOURCE_GUESSES = ("videoid", "video_id", "post_id", "id", "source")
    CONTENT_GUESSES = ("text", "transcript", "content", "full_text")

    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []
        self.stopwords: set[str] = set()
        self.allow_multi = False
        self.extra_stopwords = set()
        self.topic_labels = None
        self.topic_probs  = None
        self.topic_table  = None
        self.topic_names  = None
        self.vectorizer   = None

       # step 1 - loading
    def load_datasets(self) -> None:
        """Interactive input"""
        import os

        while True:
            raw = input("Enter the path to the file (Enter = break): ").strip()
            if raw == "":
                if self.frames:
                    break 
                print("We need at least 1 file, repeat")
                continue

            path = Path(raw).expanduser().resolve()

            # pathcheck
            if path.is_dir():
                print("That's a directory, not a file:", path)
                print("Files:\n", *os.listdir(path), sep=" • ")
                continue
            if not path.is_file():
                print("File not found:", path)
                continue

            # reading the file
            try:
                if path.suffix.lower() in (".xlsx", ".xls"):
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(path, sep=None, engine="python")
            except Exception as e:
                print("Unable to read:", e)
                continue

            print("✔ uploaded rows:", len(df))

            # choise of coloumns
            net_col = _ask_column(
                df, "Network", self.NETWORK_GUESSES, allow_missing=True
            )
            src_col = _ask_column(
                df, "Source", self.SOURCE_GUESSES, allow_missing=False
            )
            text_col = _ask_column(
                df, "Content", self.CONTENT_GUESSES, allow_missing=False
            )
            if src_col is None or text_col is None:
                print("Файл пропущен — нет обязательных столбцов.")
                continue
            # filling/naming
            if net_col is None:
                val = input(
                    "No 'Network' coloumn "
                    "How to name the network? "
                ).strip() or "unknown"
                df["network"] = val
            else:
                df.rename(columns={net_col: "network"}, inplace=True)

            df.rename(columns={src_col: "source_id", text_col: "content"},
                      inplace=True)
            self.frames.append(df[["network", "source_id", "content"]])

        self.data = pd.concat(self.frames, ignore_index=True)
        print("\nRows:", len(self.data))
        print(self.data.head())

    # Cleaning
    def clean(self) -> None:
        before = len(self.data)
        self.data.dropna(subset=["content"], inplace=True)
        after = len(self.data)
        rprint(
            f"[yellow]Deleted {before - after:,} rows without content.[/yellow]")

    #choise of the model
    
    def pick_model(self) -> None:
        """
        Benching LDA, NMF, BERTopic by C_v coherence.
        """
        import re, nltk, gensim
        from sklearn.feature_extraction.text import (
            CountVectorizer, TfidfVectorizer
        )
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        from bertopic import BERTopic
        from gensim.models.coherencemodel import CoherenceModel
        from rich import print as rprint
        rprint(
            f"[yellow]Choosing the best model[/yellow]")
        texts_raw = self.data["content"].astype(str).tolist()

        tokenizer = re.compile(r"\b\w\w+\b", flags=re.I)
        docs_tok = [[t.lower() for t in tokenizer.findall(t)] for t in texts_raw]
        id2word   = gensim.corpora.Dictionary(docs_tok)
        corpus    = [id2word.doc2bow(d) for d in docs_tok]

        stop = combined_stopwords()
        scores = {}
        topics_dict = {}

        # ---------- LDA ------------------------------------------------
        vec = CountVectorizer(max_df=0.9, stop_words=list(stop))
        X   = vec.fit_transform(texts_raw)

        lda = LatentDirichletAllocation(
            n_components=10, random_state=0, learning_method="batch"
        ).fit(X)

        # top-10 words
        lda_topics = [
            [vec.get_feature_names_out()[i] for i in comp.argsort()[-10:][::-1]]
            for comp in lda.components_
        ]
        cm_lda = CoherenceModel(
            topics=lda_topics, texts=docs_tok,
            dictionary=id2word, coherence="c_v"
        )
        scores["LDA"] = cm_lda.get_coherence()
        topics_dict["LDA"] = lda_topics

        # ---------- NMF ------------------------------------------------
        tfidf = TfidfVectorizer(max_df=0.9, stop_words=list(stop))
        X2    = tfidf.fit_transform(texts_raw)

        nmf = NMF(n_components=10, random_state=0).fit(X2)

        nmf_topics = [
            [tfidf.get_feature_names_out()[i] for i in comp.argsort()[-10:][::-1]]
            for comp in nmf.components_
        ]
        cm_nmf = CoherenceModel(
            topics=nmf_topics, texts=docs_tok,
            dictionary=id2word, coherence="c_v"
        )
        scores["NMF"] = cm_nmf.get_coherence()
        topics_dict["NMF"] = nmf_topics

        # ---------- BERTopic ------------------------------------------
        topic_model = BERTopic(
            language="multilingual",
            calculate_probabilities=True,
            verbose=False
        )
        topic_model.fit(texts_raw)

        bert_topics = [
            [w for w, _ in topic_model.get_topic(tid)[:10]]
            for tid in topic_model.get_topics().keys()
            if tid != -1 and len(topic_model.get_topic(tid)) > 0
        ]

        if len(bert_topics) >= 2:
            cm_bert = CoherenceModel(
                topics=bert_topics,
                texts=docs_tok,
                dictionary=id2word,
                coherence="c_v"
            )
            scores["BERTopic"] = cm_bert.get_coherence()
        else:
            scores["BERTopic"] = float("-inf")

        rprint("[bold]C_v coherence (↑ better)[/bold]")
        for m, sc in scores.items():
            rprint(f"{m:8}: {sc:.4f}")

        self.model_name = max(scores, key=scores.get)
        rprint(f"[bold green]Chosen model: {self.model_name}[/bold green]")

   # Modelling
    def build_topics(self) -> None:
        from rich import print as rprint
        resp = _prompt(
            "Do u agree with the model choise?"
            "\n   Enter — yes"
            "\n   Any input - choose manually",
            default_yes=True
        )

        if resp == True:                            
            print("✔ Working with the preassigned model.")
        else:                                
            while True:
                choice = input(
                    "\nChoose the model:"
                    "\n   1 — BERTopic"
                    "\n   2 — LDA"
                    "\n   3 — NMF"
                    "\nYour choise: "
                ).strip()

                if choice == "1":
                    self.model_name = "BERTopic"
                    print("✅BERTopic")
                    break
                elif choice == "2":
                    self.model_name = "LDA"
                    print("✅LDA")
                    break
                elif choice == "3":
                    self.model_name = "NMF"
                    print("✅NMF")
                    break
                else:
                    print("❌ Incorrect - please enter 1, 2 or 3")

        if self.model_name in {"LDA", "BERTopic"}:
            raw = _prompt("Can content relate to more than one topic?", default_yes=True)
            self.allow_multi = (raw is True)
        else:
            self.allow_multi = False

        # preperation
        import numpy as np, re
        from sklearn.feature_extraction.text import (
            CountVectorizer, TfidfVectorizer
        )
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from bertopic import BERTopic
        from rich import print as rprint

        texts = self.data["content"].astype(str).tolist()
        stop = list(combined_stopwords() | self.extra_stopwords)
        k_range = range(3, 26)

        model = self.model_name
        soft  = self.allow_multi
        rprint(f"[bold]Building the topics:[/bold] {model} | "
            f"{'soft' if soft else 'hard'}")

        # ── 3. LDA ──────────────────────────────────────────────────
        if model == "LDA":
            self.vectorizer = CountVectorizer(max_df=0.9, stop_words=stop)
            X = self.vectorizer.fit_transform(texts)

            best_k, best_perp, best_mdl = None, 1e9, None
            for k in k_range:
                mdl = LatentDirichletAllocation(
                    n_components=k, random_state=0
                ).fit(X)
                p = mdl.perplexity(X)
                if p < best_perp:
                    best_k, best_perp, best_mdl = k, p, mdl

            self.model = best_mdl
            theta = best_mdl.transform(X)
            self.topic_probs = theta

            if soft: 
                self.topic_labels = [
                    list(np.where(row > 0.05)[0]) for row in theta
                ]
            else:
                self.topic_labels = theta.argmax(axis=1)

            rprint(f"LDA k={best_k}, perplexity={best_perp:.1f}")

        # ── 4. BERTopic ─────────────────────────────────────────────
        elif model == "BERTopic":
            tm = BERTopic(
                language="multilingual",
                calculate_probabilities=True,
                verbose=False
            )
            labels, probs = tm.fit_transform(texts)

            self.model = tm
            self.topic_probs = probs

            if soft:
                self.topic_labels = [
                    list(np.where(p > 0.05)[0]) for p in probs
                ]
            else:
                self.topic_labels = probs.argmax(axis=1)

            rprint(f"Topics obtained: {len(set(labels))}")

        # ── 5.  NMF + KMeans  (always hard) ─────────────────────────
        else:                          
            self.vectorizer = TfidfVectorizer(max_df=0.9, stop_words=stop)
            X = self.vectorizer.fit_transform(texts)

            best_k, best_s, W_best = None, -1, None
            for k in k_range:
                nmf = NMF(n_components=k, random_state=0)
                W = nmf.fit_transform(X)
                s = silhouette_score(W, W.argmax(axis=1))
                if s > best_s:
                    best_k, best_s, W_best = k, s, W
            nmf_best = NMF(n_components=best_k, random_state=0)
            W_best   = nmf_best.fit_transform(X)

            km = KMeans(n_clusters=best_k, random_state=0, n_init=10)
            self.topic_labels = km.fit_predict(W_best)
            self.nmf        = nmf_best        
            self.kmeans     = km       
            self.model_name = "NMF"         
            rprint(f"NMF+KMeans k={best_k}, silhouette={best_s:.3f}")

        if soft:
            self.data["topic"] = [
                ", ".join(map(str, lbls)) if lbls else ""
                for lbls in self.topic_labels
            ]
        else:
            self.data["topic"] = self.topic_labels

        rprint("[green]Clustering ended sucessfully.[/green]")

    def show_topics(self, topn: int = 20) -> None:
        """
        Gives out the table «topic | top_words».
        """
        import numpy as np, pandas as pd
        from IPython.display import display

        rows = []

        # ── 1. BERTopic ───────────────────────────────────────────
        if self.model_name == "BERTopic":
            topic_ids = sorted(
                t for t in self.model.get_topics().keys() if t != -1
            )
            for tid in topic_ids:
                words = [w for w, _ in self.model.get_topic(tid)[:topn]]
                rows.append({"topic": tid, "top_words": " ".join(words)})

        # ── 2. LDA ────────────────────────────────────────────────
        elif self.model_name == "LDA":
            vocab  = self.vectorizer.get_feature_names_out()
            comps  = self.model.components_   
            for tid, row in enumerate(comps):
                idx = row.argsort()[-topn:][::-1]
                rows.append({"topic": tid,
                            "top_words": " ".join(vocab[idx])})

        # ── 3. NMF ────────────────────────────────────────────────
        elif self.model_name == "NMF":
            vocab = self.vectorizer.get_feature_names_out()

            if hasattr(self, "nmf"):
                comps = self.nmf.components_
            else:
                print("⚠️  self.nmf not found - redo step 5")
                return

            for tid, row in enumerate(comps):
                idx = row.argsort()[-topn:][::-1]
                rows.append({"topic": tid,
                            "top_words": " ".join(vocab[idx])})

        self.topic_table = pd.DataFrame(rows)
        display(self.topic_table)

    # refining stopwords
    def refine_stopwords(self) -> None:
        """
        1) Gives out the top words table.
        2) Asks if there are trash words in the top.
        • If yes - list of the trash words as input
        • Adds them to stopwords
        • Redoes build_topics() and show_topics()
        """
        while True:
            ans = _prompt(
            'Are there any top words you would like to delete (trash words)?',
                default_yes=False
            )
            if ans == 'n':
                break
            extra = input(
                "Enter the additional stopwords (using ',' no spaces): "
            ).strip()
            if not extra:
                break

            self.extra_stopwords.update(w.strip() for w in extra.split(',') if w.strip())
            self.build_topics()
            self.show_topics()

    # naming the topics
    def name_topics(self) -> None:
        """
        Interactive naming of the topics
        """
        from IPython.display import display
        display(self.topic_table)
        if not _prompt('Do you want to manually name the topics using top words?', default_yes=True):
            self.topic_names = {t: 'Topic ' + str(t) for t in self.topic_table.topic}
            return

        self.topic_names = {}
        for t, words in zip(self.topic_table.topic, self.topic_table.top_words):
            print('\nTpp-words', t, '→', words)
            name = input('Enter the name of the topic ' + str(t) + ': ').strip()
            self.topic_names[t] = name if name else 'Topic ' + str(t)

        import pandas as pd
        renamed = pd.DataFrame({
            'topic': self.topic_table.topic,
            'name':  [self.topic_names[t] for t in self.topic_table.topic],
            'top_words': self.topic_table.top_words
        })
        display(renamed)

   # Export
    def export(self, outfile: str = "thematic_clusters.xlsx") -> None:
        """
        Exports excel and json
        """
        import re, json
        from pathlib import Path
        import pandas as pd
        rows = []
        for tid, words in self.topic_table[["topic", "top_words"]].values:
            if self.allow_multi:
                mask = self.data["topic"].str.contains(rf"\b{tid}\b")
            else:
                mask = self.data["topic"] == tid
            rows.append(
                {
                    "topic": tid,
                    "top_words": words,
                    "count": int(mask.sum()),
                }
            )
        summary = pd.DataFrame(rows)

        writer = pd.ExcelWriter(outfile, engine="openpyxl")
        summary.to_excel(writer, sheet_name="summary", index=False)
        base_cols = ["topic"]
        optional_cols = [c for c in ["network", "source_id"] if c in self.data.columns]
        use_cols = base_cols + optional_cols + ["content"]

        def make_snippet(text, n=20):
            tokens = re.findall(r"\w+", str(text))[:n]
            return " ".join(tokens)

        for _, row in summary.iterrows():
            tid = int(row["topic"])
            title = str(self.topic_names.get(tid, "")).strip()
            sheet_name = (title if title else "T" + str(tid))[:31]

            if self.allow_multi:
                mask = self.data["topic"].str.contains(rf"\b{tid}\b")
            else:
                mask = self.data["topic"] == tid
            df_t = self.data.loc[mask, use_cols].copy()
            if df_t.empty:
                continue

            df_t.rename(columns={"content": "snippet"}, inplace=True)
            df_t["snippet"] = df_t["snippet"].apply(make_snippet)
            df_t.to_excel(writer, sheet_name=sheet_name, index=False)

        writer.close()

        json_path = Path(outfile).with_suffix(".json")
        with open(json_path, "w", encoding="utf8") as fp:
            json.dump(summary.to_dict(orient="records"), fp,
                    ensure_ascii=False, indent=2)

        rprint(f"[bold green]Exported:[/bold green]  "
            f"{outfile}  и  {json_path.name}")
        
    def run(self) -> None:
        self.load_datasets()
        self.clean()
        self.pick_model()
        self.build_topics()
        self.show_topics()
        self.refine_stopwords()
        self.name_topics()
        self.export()

if __name__ == "__main__":
    wizard = ThematicClustering()
    wizard.run()
