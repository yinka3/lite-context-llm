from typing import Dict, List, Optional, Tuple, Union
import spacy
import json
from _types import TopicExtractionResult, RelationReport, SentimentResult
from spacy.language import Language
from spacy.tokens import Doc, Token
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NLP_MODEL = None
try:
    NLP_MODEL = spacy.load("en_coreference_web_trf")
    print("✅ Loaded SpaCy with dedicated coreference model")
    
    test_doc = NLP_MODEL("John went to the store. He bought milk.")
    if hasattr(test_doc._, 'coref_clusters'):
        print(f"✅ Coreference working - found {len(test_doc._.coref_clusters)} clusters")
        
        for i, cluster in enumerate(test_doc._.coref_clusters):
            mentions = [mention.text for mention in cluster.mentions]
            print(f"  Cluster {i}: {mentions}")
    else:
        raise Exception("Coreference extension not found")
        
except OSError as e:
    print(f"❌ Failed to load en_coreference_web_trf model")
    print("Install with: python -m spacy download en_coreference_web_trf")
    raise OSError("Coreference model not available")
except Exception as e:
    print(f"❌ Coreference setup failed: {e}")
    raise


class SpacyModel:

    def __init__(self):

        self.nlp: Language = NLP_MODEL
        self.vader: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()

        try:
            with open("./vocab.json" ,"r") as file:
                self.vocab: Dict[str, Union[List, Dict]] = json.load(file)
        except Exception:
            raise Exception("Could not load in vocab.")
        
        self.nlp_docs: Dict[str, Doc] = {}
        for cat, doc in self.vocab.get("seed_concepts").items():
            self.nlp_docs[cat] = self.nlp(" ".join(doc))

    def _resolve_coreference(self, doc: Doc) -> Dict[int, Token]:
        resolved_map: Dict[int, Token] = {}
        
        try:
            if not hasattr(doc._, 'coref_clusters'):
                raise AttributeError("Coreference clusters not found - coref pipeline not loaded properly")
            
            clusters = doc._.coref_clusters
            print(f"Found {len(clusters)} coreference clusters")
            
            for cluster in clusters:
                main = cluster.main
                print(f"Cluster main: '{main.text}' mentions: {[m.text for m in cluster.mentions]}")
                
                for mention in cluster.mentions:
                    if mention.root.pos_ == "PRON":
                        resolved_map[mention.root.i] = main
                        print(f"Resolved pronoun '{mention.root.text}' -> '{main.text}'")
                        
        except Exception as e:
            print(f"❌ Coreference resolution failed: {e}")
            print("This will impact topic relationship detection")
            # Re-raise since you need this functionality
            raise
            
        return resolved_map
    
    def _adj_to_state(self, adj: Token) -> Optional[Tuple[str, str]]:
        STATE_ADJECTIVE: List[str] = self.vocab.get("state_adjectives")        

        if adj.lemma_ in STATE_ADJECTIVE:
            return ("condition", adj.lemma_)

        match_category: str = ""
        max_similarity: float = 0.0

        for cat, seed_doc in self.nlp_docs.items():
            similarity = adj.similarity(seed_doc)
            if similarity > max_similarity:
                max_similarity = similarity
                match_category = cat
        
        if max_similarity > 0.60:
            return (match_category, adj.lemma_)

        return None

    def extract_topics(self, text: str) -> TopicExtractionResult:
        
        doc = self.nlp(text)
        resolved_map = self._resolve_coreference(doc)
        STATE_VERBS = self.vocab.get("state_verbs")
        INTENT_MAP = self.vocab.get("intent_map")

        result = TopicExtractionResult()
        result.entities = {ent.text: ent.label_ for ent in doc.ents}


        root_token = next((token for token in doc if token.dep_ == "ROOT"), None)

        if root_token:
            intent: str = INTENT_MAP.get(root_token.lemma_)
            if intent:
                result.intent = intent
                
        for token in doc:
            target_token = None
            if token.pos_ in ["VERB", "ADJ"]:

                for child in token.children:
                    if child.dep_ in ["dobj", "nsubjpass", "nsubj"]:
                        target_token = child
                        break
                
                if not target_token and token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                    target_token = token.head
                
                if not target_token:
                    continue

                if target_token.i in resolved_map:
                    topic_noun = resolved_map[target_token.i].root.text
                else:
                    topic_noun = target_token.lemma_
                
                if token.lemma_ in STATE_VERBS:
                    state_verb: List[str] = STATE_VERBS[token.lemma_]
                    cat: str = state_verb[0]
                    value: str = state_verb[1]
                    if topic_noun not in result.state_info:
                        result.state_info[topic_noun] = {}
                    result.state_info[topic_noun][cat] = value

                if token.pos_ == "ADJ":
                    states = self._adj_to_state(token)

                    if states:
                        cat, value = states
                        if topic_noun not in result.state_info:
                            result.state_info[topic_noun] = {}
                        result.state_info[topic_noun][cat] = value
            
        if doc.noun_chunks:
            result.main_topic = max(doc.noun_chunks, key=len).text
            
        return result


    def topics_are_related(self, text1: str, text2: str) -> RelationReport:
        """
        Check if texts are contextually related, including state changes
        Returns: (are_related, confidence_score, common_elements)
        """        
        topics1: TopicExtractionResult = self.extract_topics(text1)
        topics2: TopicExtractionResult = self.extract_topics(text2)

        report: RelationReport = RelationReport()
                
        if topics1.state_info and topics2.state_info:
            keys1 = set(topics1.state_info.keys())
            keys2 = set(topics2.state_info.keys())
            shared_topics = keys1 & keys2
            if shared_topics:
                report.found_connections.append("state_overlap")
                report.common_elements.update(shared_topics)
                report.confidence_score += 0.35
        
        m_topic1 = topics1.main_topic
        m_topic2 = topics2.main_topic

        if m_topic1 and m_topic2:
            if m_topic1 in m_topic2 or m_topic2 in m_topic1:
                report.found_connections.append("main_topic_overlap")
                report.common_elements.add(m_topic1 if len(m_topic1) < len(m_topic2) else m_topic2)
                report.confidence_score += 0.25
        
        if topics1.state_info:
            for topic in topics1.state_info.keys():
                if topic.lower() in text2.lower() and topic not in report.common_elements:
                    report.found_connections.append("topic_continuation")
                    report.common_elements.add(topic)
                    report.confidence_score += 0.15
                    break
    
        if topics2.state_info:
            for topic in topics2.state_info.keys():
                if topic.lower() in text1.lower() and topic not in report.common_elements:
                    report.found_connections.append("topic_continuation")
                    report.common_elements.add(topic)
                    report.confidence_score += 0.15
                    break
        

        if topics1.entities and topics2.entities:
            entities1 = set(topics1.entities.keys())
            entities2 = set(topics2.entities.keys())

            shared_entities = entities1 & entities2
            if shared_entities:
                report.found_connections.append("entity_overlap")
                report.common_elements.update(shared_entities)
                entity_score = min(0.20, 0.1 * len(shared_entities))
                report.confidence_score += entity_score
        
        if topics1.intent and topics2.intent:
            if topics1.intent == topics2.intent and topics1.intent != "unknown":
                report.found_connections.append("similar_intent")
                report.confidence_score += 0.1
        
        doc2 = self.nlp(text2)
    
        has_early_pronoun = False
        for token in doc2[:min(5, len(doc2))]:
            if token.pos_ == "PRON" and token.dep_ in ["nsubj", "nsubjpass"]:
                has_early_pronoun = True
                break
        
        if has_early_pronoun:
            if report.common_elements:
                report.found_connections.append("pronoun_continuation_strong")
                report.confidence_score += 0.15
            else:
                report.found_connections.append("pronoun_continuation_weak")
                report.confidence_score += 0.1
        
        if report.confidence_score > 0.2:
            report.is_related = True

        report.confidence_score = min(report.confidence_score, 1.0)
        report.common_elements = report.common_elements
        
        return report
    

    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Use VADER for more accurate sentiment analysis"""
        scores = self.vader.polarity_scores(text)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            label = "positive"
        elif compound_score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        
        return SentimentResult(score=compound_score, label=label, 
                               positive_count=scores['pos'],
                               negative_count=scores['neg'])


    def get_sentiment_emoji(self, label: str) -> str:
        """Convert sentiment label to emoji for display"""
        emoji_map = {
            "positive": "😊",
            "negative": "😤", 
            "neutral": "😐"
        }
        return emoji_map.get(label, "😐")
