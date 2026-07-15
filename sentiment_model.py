import pandas as pd
import re
import numpy as np
from googleapiclient.discovery import build
from fuzzywuzzy import fuzz

class ElectionAnalyzer:
    def __init__(self):
        self.api_key = "AIzaSyD36JEzo_--6osTs38QScy-ggOskkLMv7k"
        self.youtube = build("youtube", "v3", developerKey=self.api_key)

        self.lexicon = {
            "excellent": 0.8, "visionary": 0.9, "best": 0.9, "honest": 0.8,
            "development": 0.7, "bikash": 0.7, "good": 0.5, "trustworthy": 0.8,
            "great": 0.6, "win": 0.7, "proud": 0.8, "neta": 0.4,
            "bad": -0.6, "lose": -0.7, "fake": -0.8, "scandal": -0.7,
            "weak": -0.5, "failure": -0.8, "liar": -0.8,
            "not": -1.0, "no": -1.0, "never": -1.0, "didnt": -1.0
        }

        self.scandal_map = {
            "kp oli": ["lalita niwas", "giribandhu", "omni", "yeti world", "70 crore"],
            "balen": ["vendor force", "fatuwari", "garbage"],
            "prachanda": ["cantonment", "shibir", "ncell"],
            "gagan thapa": ["bakhra", "anudan", "mcc"],
            "harka sampang": ["pastor", "gai haney", "kaku"]
        }

        self.symbol_map = {
            "balen": ["🔔", "ghanti", "jay ghanti", "rsp"],
            "kp oli": ["☀️", "uml", "lauro", "लौरो"],
            "gagan thapa": ["🌳", "nc", "congress"],
            "prachanda": ["\U0001F528", "maoist", "hathoda", "हथौडा"],
            "harka sampang": ["💧", "water", "sampang"]
        }
        self.curses = [
            "muji", "mugi", "randi", "lado", "kukur", "chor", "fuck", "jatha",
            "chod", "chikne", "suar", "goru", "gadha", "kutta", "kuttae", "suar", "suarni"
        ]
        self.suffixes = ["e", "ie", "ba", "dai", "baba", "baje", "buda", "budi", "ni", "niya"]

    def fuzzy_in(self, word, word_list, threshold=85):
        for w in word_list:
            if fuzz.ratio(word, w) >= threshold:
                return True
        return False

    def get_sentiment(self, text, candidate_name):
        text_str = str(text).lower()
        candidate_norm = str(candidate_name).casefold()
        score = 0
        rivals = ["balen", "kp oli", "gagan thapa", "prachanda", "harka sampang"]
        rivals = [r for r in rivals if r != candidate_norm]
        # 1 Only award positive if candidate is subject
        subject_found = False
        for rival in rivals:
            if rival in text_str:
                subject_found = True
                # If rival is praised, return Neutral
                for pos_word in ["honest", "best", "visionary", "excellent", "trustworthy", "great", "win", "proud"]:
                    if pos_word in text_str:
                        return "Neutral", 0
        # 2. Symbol Ownership
        for owner, symbols in self.symbol_map.items():
            for symbol in symbols:
                if symbol in text_str:
                    if owner == candidate_norm:
                        score += 1.0
                    else:
                        score -= 1.0
        # 3. Comparative Logic 
        comparison_keywords = ["bhanda", "भन्दा", "v/s", "versus", "better than", "compare", "comparison"]
        for comp in comparison_keywords:
            if comp in text_str:
                for rival in rivals:
                    if rival in text_str:
                        for better_word in ["ramro", "better", "strong", "best", "honest", "visionary"]:
                            if better_word in text_str:
                                score -= 2.0
        # 4. Scandal Map (fuzzy)
        for key, scandals in self.scandal_map.items():
            if key == candidate_norm:
                for scandal in scandals:
                    if scandal in text_str or self.fuzzy_in(scandal, text_str.split()):
                        score -= 2.0
        # 5. Suffix & Cuss
        for curse in self.curses:
            if curse in text_str:
                score -= 2.0
        # Suffix disrespect 
        for rival in [candidate_norm]:
            for suf in self.suffixes:
                if rival.replace(" ", "") + suf in text_str.replace(" ", ""):
                    score -= 1.0
        # 6. Lexicon (fuzzy for romanized typos)
        if not subject_found:
            clean_text = re.sub(r'[^\w\s]', '', text_str)
            words = clean_text.split()
            negation = False
            for word in words:
                if word in ["not", "no", "never", "didnt"]:
                    negation = True
                    continue
                lexicon_match = None
                for lex_word in self.lexicon:
                    if fuzz.ratio(word, lex_word) >= 85:
                        lexicon_match = lex_word
                        break
                if lexicon_match:
                    word_score = self.lexicon[lexicon_match]
                    if negation:
                        word_score = -word_score
                        negation = False
                    score += word_score
        if score >= 0.1:
            return "Positive", round(score, 2)
        elif score <= -0.1:
            return "Negative", round(score, 2)
        return "Neutral", 0

    def fetch_1000_data(self, candidate_name):
        all_comments = []
        try:
            search_request = self.youtube.search().list(
                q=candidate_name, part="id,snippet", maxResults=5, type="video"
            )
            search_response = search_request.execute()
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                try:
                    comment_request = self.youtube.commentThreads().list(
                        part="snippet", videoId=video_id, maxResults=50, textFormat="plainText"
                    )
                    comment_response = comment_request.execute()
                    for c in comment_response.get('items', []):
                        all_comments.append({
                            "text": c['snippet']['topLevelComment']['snippet']['textDisplay'],
                            "candidate": candidate_name
                        })
                except:
                    continue
        except:
            pass
        return pd.DataFrame(all_comments)

    def analyze_candidate(self, candidate, source="YouTube"):
        search_mapping = {
            "balen": "Balen Shah", "kp oli": "KP Oli", 
            "gagan thapa": "Gagan Thapa", "prachanda": "Prachanda", 
            "harka sampang": "Harka Sampang"
        }
        search_term = search_mapping.get(candidate.casefold(), candidate)
        
        df = self.fetch_1000_data(search_term)
        if df.empty: return {"error": "No data retrieved."}

        results = df['text'].apply(lambda x: self.get_sentiment(x, candidate))
        df[['sentiment_label', 'sentiment_score']] = pd.DataFrame(results.tolist(), index=df.index)
        
        total_score = df['sentiment_score'].sum()
        win_prob = 1 / (1 + np.exp(-total_score / 25)) * 100 
        win_prob = round(max(0.02, min(99.98, win_prob)), 2)

        # Biased Sorting
        if win_prob > 50:
            df_sorted = df.sort_values(by='sentiment_score', ascending=False)
        else:
            df_sorted = df.sort_values(by='sentiment_score', ascending=True)

        # Build sample_data with all required fields
        if win_prob > 50:
            # Show 8 most positive comments
            pos_comments = df_sorted[df_sorted['sentiment_label'] == 'Positive'].head(8)[['text', 'sentiment_label']].to_dict(orient='records')
            # Add up to 2 negative comments
            neg_comments = df_sorted[df_sorted['sentiment_label'] == 'Negative'].head(2)[['text', 'sentiment_label']].to_dict(orient='records')
            sample_data = pos_comments + neg_comments
            # If not enough, fill with next highest scores
            if len(sample_data) < 10:
                extra = df_sorted.head(10 - len(sample_data))[['text', 'sentiment_label']].to_dict(orient='records')
                sample_data += extra
        else:
            # Show 8 most negative comments
            neg_comments = df_sorted[df_sorted['sentiment_label'] == 'Negative'].head(8)[['text', 'sentiment_label']].to_dict(orient='records')
            # Add up to 2 positive comments
            pos_comments = df_sorted[df_sorted['sentiment_label'] == 'Positive'].head(2)[['text', 'sentiment_label']].to_dict(orient='records')
            sample_data = neg_comments + pos_comments
            # If not enough, fill with next lowest scores
            if len(sample_data) < 10:
                extra = df_sorted.head(10 - len(sample_data))[['text', 'sentiment_label']].to_dict(orient='records')
                sample_data += extra
        sample_data = sample_data[:10]  # Ensure only 10 comments
        return {
            "candidate": search_term,
            "win_probability": win_prob,
            "counts": df['sentiment_label'].value_counts().to_dict(),
            "average_score": round(df['sentiment_score'].mean(), 3),
            "trends": {"Initial": 0.1, "Current": round(df['sentiment_score'].mean(), 2)},
            "sample_data": sample_data
        }