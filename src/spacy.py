import spacy
import json
from spacy.tokens import Doc, Token, Span

class SpacyModel:

    def __init__(self):

        try:
            nlp_model = spacy.load("en_core_web_md")
            print("SpaCy loaded successfully")
        except:
            print("SpaCy not loaded. Install with: python -m spacy download en_core_web_md")
            nlp_model = None

        self.nlp = nlp_model
        self.coref = self.nlp.add_pipe("experimental_coref")

        try:
            with open("./vocab.json" ,"r") as file:
                self.vocab = json.load(file)
        except Exception:
            raise Exception("Could not load in vocab.")
        

    def _resolve_coreference(self, doc: Doc):
        resolved_map = {}

        for cluster in doc._.coref_clusters:
            main = cluster.main
            for mention in cluster.mentions:
                if mention.root.pos_ == "PRON":
                    resolved_map[mention.root.i] = main
        
        return resolved_map
    
    def _adj_to_state(self, adj: Token):
        STATE_ADJECTIVE = self.vocab.get("state_adjectives")
        SEED_CONCEPTS = self.vocab.get("seed_concepts")

        if adj.lemma_ in STATE_ADJECTIVE:
            return ("condition", adj.lemma_)

        match_category = None
        max_similarity = 0

        for cat, seed_doc in SEED_CONCEPTS.items():
            clump = self.nlp(" ".join(seed_doc))
            similarity = adj.similarity(clump)
            if similarity > max_similarity:
                max_similarity = similarity
                match_category = cat
        
        if max_similarity > 0.60:
            return (match_category, adj.lemma_)

        return None

    def extract_topics(self, text: str):
        
        doc = self.nlp(text)
        resolved_map = self._resolve_coreference(doc)
        STATE_VERBS = self.vocab.get("state_verbs")
        INTENT_MAP = self.vocab.get("intent_map")

        result = {
            "intent": "unknown",
            "main_topic": None,
            "entities": {ent.text: ent.label_ for ent in doc.ents},
            "state_info": {}
        }

        root_token = next((token for token in doc if token.dep_ == "ROOT"), None)

        if root_token:
            intent = INTENT_MAP.get(root_token.lemma_)
            if intent:
                result["intent"]  = intent
                
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
                    cat, value = STATE_VERBS[token.lemma_]
                    if topic_noun not in result["state_info"]:
                        result["state_info"][topic_noun] = {}
                    result["state_info"][topic_noun][cat] = value

                if token.pos_ == "ADJ":
                    states = self._adj_to_state(token)

                    if states:
                        cat, value = STATE_VERBS[token.lemma_]
                        if topic_noun not in result["state_info"]:
                            result["state_info"][topic_noun] = {}
                        result["state_info"][topic_noun][cat] = value
            
            if doc.noun_chunks:
                result["main_topic"] = max(doc.noun_chunks, key=len).text
            
        return result


    def topics_are_related(self, text1: str, text2: str) -> tuple[bool, float, set]:
        """
        Check if texts are contextually related, including state changes
        Returns: (are_related, confidence_score, common_elements)
        """        
        topics1 = self.extract_topics(text1)
        topics2 = self.extract_topics(text2)

        report = {
            "is_related": False,
            "confidence_score": 0.0,
            "common_elements": set(),
            "found_connections": []
        }
                
        if topics1.get("state_info") and topics2.get("state_info"):
            keys1 = set(topics1["state_info"].keys())
            keys2 = set(topics2["state_info"].keys())
            shared_topics = keys1 & keys2
            if shared_topics:
                report["found_connections"].append("state_overlap")
                report["common_elements"].update(shared_topics)
                report["confidence_score"] += 0.5
        
        m_topic1 = topics1.get("main_topic")
        m_topic2 = topics2.get("main_topic")

        if m_topic1 and m_topic2:
            if m_topic1 in m_topic2 or m_topic2 in m_topic1:
                report["found_connections"].append("main_topic_overlap")
                report["common_elements"].add(m_topic1 if len(m_topic1) < len(m_topic2) else m_topic2)
                report["confidence_score"] += 0.3
        

        if topics1.get("entities") and topics2.get("entities"):
            entities1 = set(topics1["entities"].keys())
            entities2 = set(topics2["entities"].keys())

            shared_entities = entities1 & entities2
            if shared_entities:
                report["found_connections"].append("entity_overlap")
                report["common_elements"].update(shared_entities)
                report["confidence_score"] += 0.2
        
            if report["confidence_score"] > 0.25:
                report["is_related"] = True

        report["confidence_score"] = min(report["confidence_score"], 1.0)
        report["common_elements"] = list(report["common_elements"])
        return report
