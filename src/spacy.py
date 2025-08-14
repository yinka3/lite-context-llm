import spacy

try:
    nlp_model = spacy.load("en_core_web_md")
    print("SpaCy loaded successfully")
except:
    print("SpaCy not loaded. Install with: python -m spacy download en_core_web_md")
    nlp_model = None


class SpacyModel:

    def __init__(self):
        self.nlp = nlp_model
        self.nlp.add_pipe("experimental_coref")
        

    def extract_topics(self, text: str) -> dict:
        """Extract comprehensive topic information with state awareness"""
        if not self.nlp:
            return {"keywords": set()}
        
        doc = self.nlp(text.lower())
        
        topics = {
            "nouns": set(),
            "verbs": set(),
            "entities": set(),
            "keywords": set(),
            "pronouns": set(),
            "context": set(),
            "states": set()
        }
        
        # Define state verbs for tracking changes
        STATE_VERBS = {
            "break": "broken", "fix": "working", "repair": "working",
            "damage": "broken", "improve": "better", "worsen": "worse",
            "start": "active", "stop": "inactive", "finish": "complete",
            "open": "open", "close": "closed", "work": "working",
            "buy": "owned", "sell": "sold", "get": "obtained"
        }
        
        # Extract nouns and their states
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                topics["nouns"].add(token.lemma_)
                topics["keywords"].add(token.lemma_)
                
                # Check for adjectives describing the noun
                for child in token.children:
                    if child.pos_ == "ADJ":
                        topics["states"].add(f"{token.lemma_}:{child.lemma_}")
                        topics["context"].add(f"{token.lemma_}_state")
                
                # Store verb-noun relationships
                if token.head.pos_ == "VERB":
                    topics["context"].add(f"{token.head.lemma_}_{token.lemma_}")
        
        # Extract verbs and their context
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                topics["verbs"].add(token.lemma_)
                
                # Check if this verb indicates a state change
                if token.lemma_ in STATE_VERBS:
                    # Find what object is changing state
                    for child in token.children:
                        if child.dep_ in ["nsubjpass", "dobj", "nsubj"] and child.pos_ == "NOUN":
                            topics["states"].add(f"{child.lemma_}:{STATE_VERBS[token.lemma_]}")
                            topics["context"].add(f"{child.lemma_}_state_change")
                            topics["keywords"].add(child.lemma_)
                
                # Handle phrasal verbs (broke up, gave up, etc.)
                for child in token.children:
                    if child.dep_ == "prt":  # Particle
                        phrase = f"{token.lemma_}_{child.text}"
                        topics["verbs"].add(phrase)
                        topics["keywords"].add(phrase)
                        topics["context"].add(token.lemma_)  # Base verb for context
                
                # Mark as important if it's the root verb
                if token.dep_ == "ROOT":
                    topics["keywords"].add(token.lemma_)
        
        # Track pronouns for continuation
        for token in doc:
            if token.pos_ == "PRON":
                topics["pronouns"].add(token.text.lower())
                if token.dep_ in ["nsubj", "nsubjpass"]:
                    topics["context"].add("has_subject_pronoun")
        
        # Extract subject-verb-object relationships
        for token in doc:
            if token.dep_ == "ROOT":  # Main verb
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.lemma_ if child.pos_ != "PRON" else child.text.lower()
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        obj = child.lemma_
                
                if subject:
                    topics["context"].add(f"subj_{subject}")
                if obj:
                    topics["context"].add(f"obj_{obj}")
                if subject and obj:
                    topics["context"].add(f"{subject}_{token.lemma_}_{obj}")
        
        # Extract named entities
        for ent in doc.ents:
            topics["entities"].add(ent.text.lower())
            topics["keywords"].add(ent.text.lower())
            topics["context"].add(f"{ent.label_}_{ent.text.lower()}")
        
        # Extract noun chunks for better context
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:  # Multi-word phrases
                topics["context"].add(chunk.text.lower())
        
        return topics


    def topics_are_related(self, text1: str, text2: str) -> tuple[bool, float, set]:
        """
        Check if texts are contextually related, including state changes
        Returns: (are_related, confidence_score, common_elements)
        """
        if not self.nlp:
            return (True, 0.5, set())
        
        topics1 = self.extract_topics(text1)
        topics2 = self.extract_topics(text2)
        
        common_elements = set()
        
        # CHECK 1: Same object, different states (car broken → car fixed)
        objects_with_states1 = {s.split(':')[0] for s in topics1.get("states", set()) if ':' in s}
        objects_with_states2 = {s.split(':')[0] for s in topics2.get("states", set()) if ':' in s}
        
        shared_objects_different_states = objects_with_states1 & objects_with_states2
        if shared_objects_different_states:
            common_elements.update(f"state_change:{obj}" for obj in shared_objects_different_states)
            return (True, 0.8, common_elements)
        
        # CHECK 2: State change context for same objects
        state_contexts1 = {c.split('_')[0] for c in topics1.get("context", set()) if "_state" in c}
        state_contexts2 = {c.split('_')[0] for c in topics2.get("context", set()) if "_state" in c}
        
        if state_contexts1 & state_contexts2:
            common_elements.update(state_contexts1 & state_contexts2)
            return (True, 0.7, common_elements)
        
        # CHECK 3: Pronoun continuation
        if "has_subject_pronoun" in topics2.get("context", set()):
            # If message 2 starts with pronouns and shares any nouns/entities with message 1
            if (topics1["nouns"] & topics2["nouns"]) or (topics1["entities"] & topics2["entities"]):
                common_elements.add("pronoun_continuation")
                return (True, 0.65, common_elements)
        
        # CHECK 4: Shared subjects or objects
        subjects1 = {c.split('_')[1] for c in topics1.get("context", set()) if c.startswith("subj_")}
        subjects2 = {c.split('_')[1] for c in topics2.get("context", set()) if c.startswith("subj_")}
        objects1 = {c.split('_')[1] for c in topics1.get("context", set()) if c.startswith("obj_")}
        objects2 = {c.split('_')[1] for c in topics2.get("context", set()) if c.startswith("obj_")}
        
        shared_subjects = subjects1 & subjects2
        shared_objects = objects1 & objects2
        
        if shared_subjects or shared_objects:
            common_elements.update(shared_subjects | shared_objects)
            return (True, 0.6, common_elements)
        
        # CHECK 5: Calculate weighted overlap for general similarity
        scores = {
            "nouns": 0.35,      # What the text is about
            "keywords": 0.30,   # Most important terms
            "entities": 0.20,   # Specific names/places
            "verbs": 0.10,      # Actions being discussed
            "context": 0.05     # Additional context
        }
        
        total_score = 0
        
        for category, weight in scores.items():
            set1 = topics1.get(category, set())
            set2 = topics2.get(category, set())
            
            overlap = set1 & set2
            if overlap:
                union = set1 | set2
                if union:
                    similarity = len(overlap) / len(union)
                    total_score += similarity * weight
                    common_elements.update(overlap)
        
        # BOOST: If they share important nouns, increase score
        shared_nouns = topics1["nouns"] & topics2["nouns"]
        if shared_nouns:
            total_score += 0.2 * len(shared_nouns)
            common_elements.update(shared_nouns)
        
        # Decision threshold
        are_related = total_score > 0.15 or len(common_elements) >= 2
        
        # If using spaCy medium/large model, also check semantic similarity
        if not are_related and not common_elements:
            try:
                doc1 = self.nlp(text1)
                doc2 = self.nlp(text2)
                if doc1.has_vector and doc2.has_vector:
                    semantic_similarity = doc1.similarity(doc2)
                    if semantic_similarity > 0.7:
                        return (True, semantic_similarity, {"high_semantic_similarity"})
            except:
                pass
        
        return (are_related, total_score, common_elements)


