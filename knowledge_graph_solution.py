import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import ollama
import time
import spacy
from difflib import get_close_matches
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Entity:
    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT
    properties: Dict = field(default_factory=dict)
    canonical_name: str = ""  # For disambiguation

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.name


@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 0.9


@dataclass
class PersonalityProfile:
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    behaviors: List[str] = field(default_factory=list)
    emotional_states: List[str] = field(default_factory=list)
    evidence_count: int = 1  # How many text segments contributed to this profile


# ============================================================================
# SYNTHETIC DATA GENERATOR
# ============================================================================


class SyntheticDataGenerator:
    """Generate realistic synthetic documents for testing"""

    @staticmethod
    def generate_narrative_document() -> str:
        return """
    Sarah Chen, a seasoned software engineer at TechCorp, has always been known 
    for her innovative thinking and creative problem-solving. She eagerly embraces 
    new technologies and often proposes unconventional solutions during team meetings. 
    Her colleagues describe her as enthusiastic, outgoing, and always willing to help.
    This high-openness trait makes her a key player in brainstorming for 'Project Nova'.

    Last month, Sarah led a critical project to redesign the company's authentication 
    system. Despite the tight deadline, she remained calm and organized, meticulously 
    planning every phase of the implementation. Her attention to detail and systematic 
    approach impressed both her manager, David Martinez, and the executive team.
    David Martinez, who also manages the QA team, noted her high conscientiousness in his report.

    However, Sarah sometimes struggles with criticism. During a code review session 
    in San Francisco headquarters, when her colleague John Smith pointed out a potential 
    security vulnerability, she became visibly upset and defensive. John Smith, who is 
    officially part of the remote team in the Seattle office, is known for his blunt, 
    analytical feedback. A new QA lead, Ben Carter, was also in the meeting. Ben is 
    highly introverted but extremely detail-oriented, and he later privately sent Sarah 
    a file with three more minor bugs. This combination made Sarah feel overwhelmed.

    Later, she acknowledged the feedback from both John and Ben. She worked collaboratively 
    with John Smith to fix the major issue and coordinated with Ben Carter on the minor 
    patches, demonstrating her ability to manage emotions. David Martinez praised all three
    for their professionalism on Project Nova.

    Sarah is an active member of the AI Ethics Committee at TechCorp, a group chaired by 
    David Martinez. In this committee, she passionately advocates for responsible AI 
    development, often collaborating with legal expert, Amina Al-Jamil. She frequently 
    volunteers to mentor junior developers and organizes community workshops on machine 
    learning basics. Her empathetic nature and genuine concern for others' growth make 
    her a beloved figure in the tech community in San Francisco.

    In her personal life, Sarah tends to worry about work-life balance and sometimes 
    experiences anxiety before major presentations. This neuroticism is something she 
    discusses openly with her mentor, Amina Al-Jamil, who helps her channel that
    energy into meticulous preparation.
    """

    @staticmethod
    def generate_business_document() -> str:
        return """
    The merger between GlobalTech Industries and InnovateSoft was announced in 
    January 2024, marking a significant shift in the enterprise software landscape. 
    CEO Jennifer Liu of GlobalTech praised the deal, stating it would create 
    unprecedented opportunities for innovation. The deal was valued at $5 billion.

    The negotiation process, led by Chief Financial Officer Robert Kim of GlobalTech,
    took nearly eight months. Robert demonstrated exceptional patience and analytical 
    thinking throughout the complex discussions. He worked closely with the legal team 
    at 'Sullivan & Cromwell' to finalize the terms.

    Jennifer Liu's leadership style contrasts sharply with Robert's methodical approach. 
    She is known for her bold, risk-taking decisions and charismatic personality. 
    During investor calls, she exudes confidence and inspires stakeholders with her 
    ambitious vision for the future. Some critics, however, describe her as impulsive 
    and occasionally dismissive of dissenting opinions. She immediately pushed for 
    'Project Fusion', the aggressive integration plan.

    The merger created tension among employees at both companies. Many InnovateSoft 
    developers expressed anxiety about job security and cultural changes. Team lead 
    Maria Santos, from the InnovateSoft side, attempted to address these concerns 
    by hosting open forums and encouraging transparent communication. Her compassionate 
    approach and active listening helped ease the transition. Robert Kim collaborated
    with Maria Santos to develop a new retention bonus package.

    Meanwhile, GlobalTech's CTO Alex Thompson remained skeptical about the integration 
    timeline for Project Fusion. Known for his pessimistic but realistic assessments, Alex 
    raised concerns about technical incompatibilities. His infrastructure team, based in 
    the Berlin office, submitted a 50-page report outlining risks. Alex Thompson reported
    directly to Jennifer Liu, which led to several tense executive meetings. Dr. Emily Vance,
    the former CTO of InnovateSoft and now a board advisor, publicly backed Alex's concerns,
    creating a new alliance. Maria Santos also found herself collaborating with Alex's
    team to align developer environments.
    """

    @staticmethod
    def generate_social_document() -> str:
        return """
    The annual charity gala in New York City brought together an eclectic mix of 
    personalities from the nonprofit sector. The event was a joint effort between the
    Hope Foundation and a new partner, the 'Oceanic Preservation Trust'. Emma Watson,
    director of the Hope Foundation, greeted every guest with genuine warmth and
    enthusiasm. Her infectious energy and extroverted nature made everyone feel welcome.

    Michael Chen, a reserved but thoughtful board member from the Oceanic Preservation Trust, 
    spent most of the evening in quiet conversations with small groups. He listened more 
    than he spoke, carefully considering each person's perspective. His introverted 
    personality doesn't diminish his impact; colleagues respect his deep, analytical 
    thinking. He was seen in a long, quiet discussion with Emma Watson about a future
    joint project.

    The event's keynote speaker, Dr. Patricia Johnson, delivered an emotionally 
    powerful speech about mental health advocacy. Her openness about her own struggles 
    with anxiety resonated deeply with the audience. Patricia's neurotic tendencies, 
    which she openly discusses, fuel her passion. She founded the Mindful Living Institute,
    which receives funding from the Hope Foundation. Earlier, Dr. Johnson co-hosted a
    panel with Emma Watson titled 'Empathy in Advocacy'.

    During the fundraising auction, tension arose when businessman Carlos Rodriguez 
    aggressively outbid other participants. His competitive and sometimes abrasive 
    behavior rubbed many attendees the wrong way. Carlos Rodriguez is the CEO of 
    'Rodriguez Imports'. Emma quietly pulled him aside later, diplomatically explaining 
    the event's collaborative spirit. Carlos, though initially resistant, eventually 
    apologized and made an additional generous donation to the Mindful Living Institute.

    The evening concluded with impromptu performances. Jazz musician Lisa Park's 
    spontaneous saxophone solo captivated everyone. Lisa Park, who is a personal
    friend of Michael Chen, is known for her creative, free-spirited nature.
    She often tells people she lives by intuition rather than rigid plans. This 
    high openness to experience makes her performances unpredictable yet magical.
    Michael had personally asked her to attend, hoping she would play.
    """


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def find_entity_match(
    name: str, entity_list: list, cutoff: float = 0.8
) -> Optional[str]:
    """Improved fuzzy matching with higher threshold"""
    matches = get_close_matches(name, entity_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
    """Retry decorator with exponential backoff"""

    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"⚠️ Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
        return None

    return wrapper


# ----------------------------
# Normalization / merging utils
# ----------------------------
ORG_SUFFIXES = [
    r"\bInc\b",
    r"\bInc\.\b",
    r"\bLtd\b",
    r"\bLtd\.\b",
    r"\bLLC\b",
    r"\bCo\b",
    r"\bCompany\b",
    r"\bCorporation\b",
    r"\bCorp\b",
    r"\bIndustries\b",
    r"\bSystems\b",
    r"\bTechnologies\b",
    r"\bGroup\b",
    r"\bHoldings\b",
]

# Common role or title words that should not be standalone PERSON entities
ROLE_KEYWORDS = {
    "ceo",
    "cto",
    "cfo",
    "coo",
    "chief",
    "officer",
    "director",
    "manager",
    "president",
    "founder",
    "chairman",
    "executive",
    "analyst",
    "engineer",
    "leader",
    "consultant",
    "advisor",
    "professor",
}


def normalize_org_name(name: str) -> str:
    """Normalize organization names by removing common prefixes, suffixes, quotes, and extra whitespace."""
    n = name.strip()

    # Remove leading articles: 'The', 'A', 'An'
    n = re.sub(r"^(the|a|an)\s+", "", n, flags=re.IGNORECASE)

    # Remove commas, asterisks, single quotes, and extra punctuation
    n = re.sub(r"[,'\*]+", " ", n)

    # Remove common suffix words
    for suf in ORG_SUFFIXES:
        n = re.sub(rf"\s*{suf}\b", "", n, flags=re.IGNORECASE)

    # Collapse multiple spaces
    n = re.sub(r"\s+", " ", n).strip()

    return n


def title_person_finder(text: str) -> List[str]:
    """Find tokens like 'CTO Alex Thompson' or 'Dr. Patricia Johnson' and return 'Alex Thompson' etc."""
    matches = []
    # cover titles and honorifics often followed by a name (two-part name captured)
    pattern = re.compile(
        r"\b(?:CEO|CTO|CFO|Chief|Dr|Mr|Ms|Mrs|Professor)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
    )
    for m in pattern.finditer(text):
        matches.append(m.group(1).strip())
    return matches


# ============================================================================
# LLM RELATIONSHIP EXTRACTOR
# ============================================================================


class LLMRelationshipExtractor:
    RELATION_TYPES = [
        "works_for",
        "manages",
        "reports_to",
        "collaborates_with",
        "founded",
        "member_of",
        "located_in",
        "employed_at",
        "leads",
        "participates_in",
        "owns",
        "invested_in",
    ]

    def __init__(self, model_name: str = "gemma3:12b", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature

    def create_prompt(self, text: str) -> str:
        """Improved prompt with few-shot examples and explicit relation types"""
        return f"""You are a knowledge graph extraction assistant. Extract relationships from text.

Valid relation types: {', '.join(self.RELATION_TYPES)}

Examples:
Text: "Sarah works at Google as an engineer."
Output: [{{"source": "Sarah", "target": "Google", "relation": "works_for", "confidence": 0.95}}]

Text: "John manages the sales team at Microsoft."
Output: [{{"source": "John", "target": "Microsoft", "relation": "works_for", "confidence": 0.9}}, 
         {{"source": "John", "target": "sales team", "relation": "manages", "confidence": 0.95}}]

Now extract relationships from:
{text}

Return ONLY valid JSON array with no additional text:
[{{"source": "Entity1", "target": "Entity2", "relation": "relation_type", "confidence": 0.0-1.0}}]
"""

    @retry_with_backoff
    def _call_llm(self, prompt: str) -> str:
        """LLM call with retry logic"""
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]

    def extract_relationships(self, text: str, entity_list: list) -> List[Relationship]:
        """Extract relationships with fuzzy matching and validation"""
        try:
            prompt = self.create_prompt(text)
            output = self._call_llm(prompt)

            # Extract JSON with improved pattern matching
            json_match = re.search(r"\[.*\]", output, re.DOTALL)
            if not json_match:
                print(f"⚠️ No JSON array found in LLM output")
                return []

            raw = json.loads(json_match.group())
            relationships = []

            for r in raw:
                # Fuzzy match entities
                src_match = find_entity_match(r.get("source", ""), entity_list)
                tgt_match = find_entity_match(r.get("target", ""), entity_list)

                if src_match and tgt_match and src_match != tgt_match:
                    relation = r.get("relation", "related_to")
                    # Validate relation type
                    if relation not in self.RELATION_TYPES:
                        relation = "related_to"

                    relationships.append(
                        Relationship(
                            source=src_match,
                            target=tgt_match,
                            relation=relation,
                            confidence=float(r.get("confidence", 0.8)),
                        )
                    )

            return relationships

        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return []
        except Exception as e:
            print(f"⚠️ Error extracting relationships: {e}")
            return []


# ============================================================================
# ENTITY & PERSONALITY EXTRACTION
# ============================================================================
class LLMSimulator:
    nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def extract_entities(text: str) -> List[Entity]:
        """Enhanced entity extraction with normalization, title capture, and merging"""

        # 1. Find persons with titles first (e.g., "CTO Alex Thompson" -> "Alex Thompson")
        titled_persons = set(title_person_finder(text))  # e.g., {"Alex Thompson"}

        doc = LLMSimulator.nlp(text)
        raw_entities: List[Entity] = []
        entity_counts = Counter()

        # 2. Add titled persons immediately as high-priority PERSONs
        for person_name in titled_persons:
            raw_entities.append(
                Entity(
                    name=person_name, entity_type="PERSON", properties={"mentions": 0}
                )
            )
            entity_counts[person_name] += 1

        # 3. Collect spaCy NER outputs
        for ent in doc.ents:
            ent_text = ent.text.strip()

            # Skip if this is already a titled person
            if ent_text in titled_persons:
                entity_counts[ent_text] += 1
                continue

            # Skip if entity contains a titled person
            is_container_for_person = any(
                person in ent_text for person in titled_persons
            )
            if is_container_for_person:
                for person in titled_persons:
                    if person in ent_text:
                        entity_counts[person] += 1
                continue

            # Determine entity type
            etype = "OTHER"
            if ent.label_ == "PERSON":
                etype = "PERSON"
            elif ent.label_ == "ORG":
                etype = "ORGANIZATION"
            elif ent.label_ in {"GPE", "LOC"}:
                etype = "LOCATION"
            elif ent.label_ == "EVENT":
                etype = "EVENT"

            entity_counts[ent_text] += 1
            raw_entities.append(
                Entity(name=ent_text, entity_type=etype, properties={"mentions": 0})
            )

        # 4. Normalize organization names and filter out known title keywords
        TITLE_KEYWORDS = {
            "cto",
            "ceo",
            "cfo",
            "coo",
            "founder",
            "director",
            "president",
            "officer",
        }

        for e in raw_entities:
            if e.entity_type == "ORGANIZATION":
                e.name = normalize_org_name(e.name)
                if e.name.lower() in TITLE_KEYWORDS:
                    e.entity_type = "OTHER"
            # Ensure titled persons are PERSON
            elif e.name in titled_persons:
                e.entity_type = "PERSON"

        # 5. Build canonical mapping for PERSONs
        persons = [e.name for e in raw_entities if e.entity_type == "PERSON"]
        canonical_map = {}
        first_name_map = {}  # first-name -> full name

        for p in persons:
            # Longer candidate containing p
            candidates = [
                q for q in persons if q.lower() != p.lower() and p.lower() in q.lower()
            ]
            canonical_map[p] = (
                max(candidates, key=lambda x: len(x)) if candidates else p
            )
            if len(p.split()) > 1:
                first_name_map[p.split()[0].lower()] = p

        # Merge single-token PERSONs (like "Emma") to full canonical names
        for e in raw_entities:
            if len(e.name.split()) == 1 and e.name.lower() in first_name_map:
                e.entity_type = "PERSON"
                canonical_map[e.name] = first_name_map[e.name.lower()]

        # 6. Apply canonical names and aggregate mentions
        final_entities: Dict[str, Entity] = {}
        for e in raw_entities:
            canonical = e.name
            if e.entity_type == "PERSON":
                canonical = canonical_map.get(e.name, e.name)
            elif e.entity_type == "ORGANIZATION":
                canonical = normalize_org_name(e.name) or e.name

            if canonical not in final_entities:
                ent_copy = Entity(
                    name=canonical,
                    entity_type=e.entity_type,
                    properties={"mentions": entity_counts.get(e.name, 0)},
                    canonical_name=canonical,
                )
                final_entities[canonical] = ent_copy
            else:
                final_entities[canonical].properties["mentions"] += entity_counts.get(
                    e.name, 0
                )
                # PERSON type overrides others
                if (
                    e.entity_type == "PERSON"
                    and final_entities[canonical].entity_type != "PERSON"
                ):
                    final_entities[canonical].entity_type = "PERSON"

        # 7. Filter out false-positive PERSON roles
        NON_PERSON_WORDS = {"advocacy", "foundation", "committee", "initiative"}
        final_entities = {
            name: e
            for name, e in final_entities.items()
            if not (
                e.entity_type == "PERSON"
                and (
                    any(word.lower() in ROLE_KEYWORDS for word in e.name.split())
                    or e.name.lower() in NON_PERSON_WORDS
                )
                and len(e.name.split()) <= 2
            )
        }

        return list(final_entities.values())

    @staticmethod
    def extract_affiliation_relationships(
        text: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Expanded affiliation extraction with broader patterns and fuzzy matching"""
        relationships = []

        # Expanded patterns covering job titles, memberships, and leadership
        affiliation_patterns = [
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works? at|employed at|employee of|joined|part of|from)\s+([A-Z][a-zA-Z&\s]+)",
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:CEO|CTO|CFO|COO|director|head|leader|manager|founder|chair|president|executive)\s+(?:of|at)\s+([A-Z][a-zA-Z&\s]+)",
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:member|representative|delegate)\s+(?:of|at)\s+([A-Z][a-zA-Z&\s]+)",
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:leads?|heads?|manages?)\s+([A-Z][a-zA-Z&\s]+)",
            r"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|started|created|established)\s+([A-Z][a-zA-Z&\s]+)",
        ]

        entity_names = [e.name for e in entities]
        persons = [e.name for e in entities if e.entity_type == "PERSON"]
        orgs = [e.name for e in entities if e.entity_type == "ORGANIZATION"]

        for pattern in affiliation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person = find_entity_match(match.group(1).strip(), persons, cutoff=0.6)
                org = find_entity_match(match.group(2).strip(), orgs, cutoff=0.6)
                if person and org and person != org:
                    relation_type = (
                        "founded"
                        if re.search(
                            r"founded|started|created|established", match.group(0), re.I
                        )
                        else (
                            "leads"
                            if re.search(r"lead|head|manage", match.group(0), re.I)
                            else "works_for"
                        )
                    )
                    relationships.append(
                        Relationship(
                            source=person,
                            target=org,
                            relation=relation_type,
                            confidence=0.85,
                        )
                    )

        # Optional: also link persons and orgs appearing together in same paragraph
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            para_persons = [p for p in persons if p in para]
            para_orgs = [o for o in orgs if o in para]
            for p in para_persons:
                for o in para_orgs:
                    relationships.append(
                        Relationship(p, o, "affiliated_with", confidence=0.6)
                    )

        return relationships

    @staticmethod
    def extract_context_windows(
        text: str, person_name: str, window_size: int = 3
    ) -> List[str]:
        """Extract sentences around each mention of a person"""
        sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
        contexts = []

        for i, sent in enumerate(sentences):
            if person_name.lower() in sent.lower():
                start = max(0, i - window_size)
                end = min(len(sentences), i + window_size + 1)
                context = " ".join(sentences[start:end])
                contexts.append(context)

        return contexts

    @staticmethod
    @retry_with_backoff
    def infer_personality(
        text: str, person_name: str, llm_model: str = "gemma3:12b"
    ) -> PersonalityProfile:
        """Improved personality inference with context aggregation"""
        contexts = LLMSimulator.extract_context_windows(text, person_name)

        if not contexts:
            return PersonalityProfile(0.5, 0.5, 0.5, 0.5, 0.5)

        # Aggregate evidence from all contexts
        context_text = "\n\n".join(contexts[:3])  # Use up to 3 contexts

        prompt = f"""Analyze the personality of {person_name} based on these descriptions.

Big Five Personality Traits (rate 0.0-1.0):
- Openness: creativity, curiosity, open to new experiences
- Conscientiousness: organized, responsible, disciplined
- Extraversion: outgoing, energetic, sociable
- Agreeableness: cooperative, compassionate, trusting
- Neuroticism: anxious, emotionally unstable, sensitive to stress

Text about {person_name}:
{context_text}

Return JSON (no additional text):
{{
  "openness": 0.0-1.0,
  "conscientiousness": 0.0-1.0,
  "extraversion": 0.0-1.0,
  "agreeableness": 0.0-1.0,
  "neuroticism": 0.0-1.0,
  "behaviors": ["behavior1", "behavior2", "behavior3"],
  "emotions": ["emotion1", "emotion2", "emotion3"]
}}
"""

        try:
            response = ollama.chat(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            content = response["message"]["content"]

            # Extract JSON
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                return PersonalityProfile(0.5, 0.5, 0.5, 0.5, 0.5)

            data = json.loads(json_match.group())

            # Validate trait ranges
            def clamp(val, min_val=0.0, max_val=1.0):
                return max(min_val, min(max_val, float(val)))

            return PersonalityProfile(
                openness=clamp(data.get("openness", 0.5)),
                conscientiousness=clamp(data.get("conscientiousness", 0.5)),
                extraversion=clamp(data.get("extraversion", 0.5)),
                agreeableness=clamp(data.get("agreeableness", 0.5)),
                neuroticism=clamp(data.get("neuroticism", 0.5)),
                behaviors=data.get("behaviors", [])[:5],
                emotional_states=data.get("emotions", [])[:5],
                evidence_count=len(contexts),
            )

        except Exception as e:
            print(f"⚠️ Error inferring personality for {person_name}: {e}")
            return PersonalityProfile(0.5, 0.5, 0.5, 0.5, 0.5)


# ============================================================================
# KNOWLEDGE GRAPH BUILDER
# ============================================================================


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.personalities: Dict[str, PersonalityProfile] = {}
        self.llm = LLMSimulator()
        self.llm_relationship_extractor = LLMRelationshipExtractor()

    def _add_relationship(self, rel: Relationship):
        """Add relationship with deduplication and confidence aggregation"""
        # Check if edge already exists
        existing_edges = self.graph.get_edge_data(rel.source, rel.target)

        if existing_edges:
            # Check if same relation type exists
            for key, data in existing_edges.items():
                if data.get("relation") == rel.relation:
                    # Update confidence (average)
                    old_conf = data.get("confidence", 0.5)
                    new_conf = (old_conf + rel.confidence) / 2
                    self.graph[rel.source][rel.target][key]["confidence"] = new_conf
                    return

        # Add new edge
        self.graph.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation,
            confidence=rel.confidence,
            **rel.properties,
        )
        self.relationships.append(rel)

    def _extract_cooccurrence_relationships(
        self, text: str, entities: List[Entity]
    ) -> List[Relationship]:
        """Enhanced co-occurrence linking across a wider 3-sentence window"""
        doc = self.llm.nlp(text)
        relationships = []

        # Split sentences and map them to entities they contain
        sentences = [sent.text for sent in doc.sents]
        sent_entities = (
            []
        )  # This is a list of lists: [ [EntA, EntB], [EntB, EntC], ... ]
        for sent in sentences:
            contained = [e for e in entities if e.name in sent]
            sent_entities.append(contained)

        # More aggressive co-occurrence: link entities in a 3-sentence window [i-1, i, i+1]
        for i in range(len(sent_entities)):
            # 1. Define the window indices
            start_idx = max(0, i - 1)
            end_idx = min(len(sent_entities), i + 2)  # (i+1) + 1 for slice

            # 2. Collect all unique entities in this window
            # *** FIX: Use a dict to store unique entities by name, as Entity object is unhashable ***
            window_entities_dict = {}
            for j in range(start_idx, end_idx):
                for entity in sent_entities[j]:
                    if entity.name not in window_entities_dict:
                        window_entities_dict[entity.name] = entity

            window_entities_list = list(
                window_entities_dict.values()
            )  # This is now a list of unique Entity objects

            # 3. Create links between all combinations in this window
            for e1, e2 in itertools.combinations(window_entities_list, 2):
                # if e1.name == e2.name: # This check is redundant now but harmless
                #     continue

                # Broaden to all meaningful pairs (PERSON, ORG, EVENT, LOCATION)
                if e1.entity_type in {
                    "PERSON",
                    "ORGANIZATION",
                    "EVENT",
                    "LOCATION",
                } and e2.entity_type in {"PERSON", "ORGANIZATION", "EVENT", "LOCATION"}:

                    # Add relationship with a fixed medium confidence
                    relationships.append(
                        Relationship(
                            source=e1.name,
                            target=e2.name,
                            relation="co_occurs",
                            confidence=0.5,  # Fixed confidence for 3-sentence window
                        )
                    )
        return relationships

    def process_document(self, text: str):
        """Improved document processing pipeline"""
        print("STEP 1: Entity Extraction")
        entities = self.llm.extract_entities(text)

        # Convert list to dict keyed by canonical name to ensure graph nodes are unique
        for entity in entities:
            key = (
                entity.canonical_name
                if getattr(entity, "canonical_name", None)
                else entity.name
            )
            # If entity exists, merge mention counts and other properties
            if key in self.entities:
                # sum mentions if available
                prev = self.entities[key]
                prev.properties["mentions"] = prev.properties.get(
                    "mentions", 0
                ) + entity.properties.get("mentions", 0)
            else:
                self.entities[key] = Entity(
                    name=key,
                    entity_type=entity.entity_type,
                    properties=entity.properties,
                    canonical_name=key,
                )

            # add/update graph node with canonical key
            self.graph.add_node(
                key,
                entity_type=self.entities[key].entity_type,
                **self.entities[key].properties,
            )
        print(f"✓ Extracted {len(entities)} entities")

        print("\nSTEP 2: Relationship Extraction")
        entity_names = list(self.entities.keys())

        # 2a. LLM-based relationships
        print("  - LLM extraction...")
        llm_rels = self.llm_relationship_extractor.extract_relationships(
            text, entity_names
        )
        for rel in llm_rels:
            self._add_relationship(rel)
        print(f"  ✓ Found {len(llm_rels)} LLM relationships")

        # 2b. Pattern-based affiliation extraction
        print("  - Affiliation extraction...")
        affiliation_rels = self.llm.extract_affiliation_relationships(text, entities)
        for rel in affiliation_rels:
            self._add_relationship(rel)
        print(f"  ✓ Found {len(affiliation_rels)} affiliations")

        # 2c. Co-occurrence relationships
        print("  - Co-occurrence analysis...")
        cooccur_rels = self._extract_cooccurrence_relationships(text, entities)
        for rel in cooccur_rels:
            self._add_relationship(rel)
        print(f"  ✓ Found {len(cooccur_rels)} co-occurrences")

        print(f"  → Total edges: {self.graph.number_of_edges()}")

        print("\nSTEP 3: Personality Inference")
        person_entities = [
            self.entities[k]
            for k in self.entities
            if self.entities[k].entity_type == "PERSON"
        ]

        # Parallel personality inference
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self.llm.infer_personality, text, person.name
                ): person.name
                for person in person_entities
            }

            for future in as_completed(futures):
                person_name = futures[future]
                try:
                    profile = future.result()
                    self.personalities[person_name] = profile

                    # Update graph node
                    self.graph.nodes[person_name].update(
                        {
                            "openness": profile.openness,
                            "conscientiousness": profile.conscientiousness,
                            "extraversion": profile.extraversion,
                            "agreeableness": profile.agreeableness,
                            "neuroticism": profile.neuroticism,
                            "behaviors": profile.behaviors,
                            "emotions": profile.emotional_states,
                        }
                    )
                    print(f"  ✓ {person_name}")
                except Exception as e:
                    print(f"  ✗ {person_name}: {e}")

        print(f"✓ Inferred {len(self.personalities)} personality profiles")

        # Validate personality consistency
        self._validate_personalities()

    def _validate_personalities(self):
        """Cross-validate personality traits against behaviors/emotions"""
        for name, profile in self.personalities.items():
            # Check high neuroticism alignment
            if profile.neuroticism > 0.7:
                negative_emotions = {
                    "anxious",
                    "worried",
                    "upset",
                    "nervous",
                    "stressed",
                }
                has_negative = any(
                    any(neg in emotion.lower() for neg in negative_emotions)
                    for emotion in profile.emotional_states
                )
                if not has_negative:
                    print(
                        f"  ⚠️ Validation warning: {name} has high neuroticism "
                        f"but no negative emotions detected"
                    )

            # Check high extraversion alignment
            if profile.extraversion > 0.7:
                social_behaviors = {"outgoing", "sociable", "talkative", "energetic"}
                has_social = any(
                    any(soc in behavior.lower() for soc in social_behaviors)
                    for behavior in profile.behaviors
                )
                if not has_social:
                    print(
                        f"  ⚠️ Validation warning: {name} has high extraversion "
                        f"but no social behaviors detected"
                    )

    def export_to_json(self, filename: str = "knowledge_graph.json"):
        """Export graph with all data"""
        data = {
            "entities": [asdict(e) for e in self.entities.values()],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "confidence": r.confidence,
                    "properties": r.properties,
                }
                for r in self.relationships
            ],
            "personalities": {
                name: asdict(profile) for name, profile in self.personalities.items()
            },
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Exported to {filename}")

    def get_statistics(self) -> Dict:
        """Compute graph statistics"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_persons": len(
                [e for e in self.entities.values() if e.entity_type == "PERSON"]
            ),
            "num_orgs": len(
                [e for e in self.entities.values() if e.entity_type == "ORGANIZATION"]
            ),
            "num_locations": len(
                [e for e in self.entities.values() if e.entity_type == "LOCATION"]
            ),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values())
            / max(1, self.graph.number_of_nodes()),
        }

    def visualize(self, filename: str = "knowledge_graph.png") -> None:
        """Visualize the knowledge graph"""
        plt.figure(figsize=(16, 12))

        # Define colors by entity type
        color_map = {
            "PERSON": "#1f77b4",  # blue
            "ORGANIZATION": "#ff7f0e",  # orange
            "LOCATION": "#2ca02c",  # green
            "EVENT": "#d62728",  # red
            "CONCEPT": "#9467bd",  # purple
            "OTHER": "#8c564b",  # brown (distinct from grey)
        }

        node_colors = [
            color_map.get(self.graph.nodes[node].get("entity_type", "OTHER"), "#CCCCCC")
            for node in self.graph.nodes()
        ]

        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos, node_color=node_colors, node_size=3000, alpha=0.9
        )

        # Draw edges with varying opacity based on confidence
        for u, v, data in self.graph.edges(data=True):
            confidence = data.get("confidence", 0.5)
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=[(u, v)],
                edge_color="gray",
                arrows=True,
                arrowsize=15,
                alpha=confidence * 0.7,
                width=1.5,
            )

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight="bold")

        # Draw edge labels (only for high confidence)
        high_conf_edges = {
            (u, v): data["relation"]
            for u, v, data in self.graph.edges(data=True)
            if data.get("confidence", 0) > 0.7
        }
        nx.draw_networkx_edge_labels(self.graph, pos, high_conf_edges, font_size=7)

        # Legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=color, label=etype) for etype, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc="upper left", framealpha=0.9)

        plt.title(
            "Knowledge Graph with Personality Modeling", fontsize=16, fontweight="bold"
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Saved visualization to {filename}")
        plt.close()


# ============================================================================
# EVALUATOR
# ============================================================================


class KnowledgeGraphEvaluator:
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg = kg_builder

    def evaluate_completeness(self) -> Dict:
        """Measure extraction completeness"""
        stats = self.kg.get_statistics()

        persons = [e for e in self.kg.entities.values() if e.entity_type == "PERSON"]
        persons_with_personality = len(self.kg.personalities)

        completeness_score = persons_with_personality / max(1, len(persons))

        # Average evidence count
        avg_evidence = sum(
            p.evidence_count for p in self.kg.personalities.values()
        ) / max(1, len(self.kg.personalities))

        return {
            "personality_coverage": completeness_score,
            "entities_extracted": stats["num_nodes"],
            "relationships_extracted": stats["num_edges"],
            "avg_evidence_per_person": avg_evidence,
            "score": completeness_score,
        }

    def evaluate_consistency(self) -> Dict:
        """Check for consistency in personality traits"""
        inconsistencies = []

        for name, profile in self.kg.personalities.items():
            # Check trait ranges
            traits = {
                "openness": profile.openness,
                "conscientiousness": profile.conscientiousness,
                "extraversion": profile.extraversion,
                "agreeableness": profile.agreeableness,
                "neuroticism": profile.neuroticism,
            }

            for trait_name, value in traits.items():
                if value < 0 or value > 1:
                    inconsistencies.append(
                        f"{name}: {trait_name} out of range ({value})"
                    )

            # Check high neuroticism alignment
            if profile.neuroticism > 0.7:
                negative_emotions = {
                    "anxious",
                    "worried",
                    "upset",
                    "nervous",
                    "stressed",
                    "tense",
                }
                has_negative = any(
                    any(neg in emotion.lower() for neg in negative_emotions)
                    for emotion in profile.emotional_states
                )
                if not has_negative:
                    inconsistencies.append(
                        f"{name}: High neuroticism ({profile.neuroticism:.2f}) without negative emotions"
                    )

            # Check high extraversion alignment
            if profile.extraversion > 0.7:
                social_indicators = {
                    "outgoing",
                    "sociable",
                    "talkative",
                    "energetic",
                    "enthusiastic",
                }
                has_social = any(
                    any(soc in behavior.lower() for soc in social_indicators)
                    for behavior in profile.behaviors
                )
                if not has_social:
                    inconsistencies.append(
                        f"{name}: High extraversion ({profile.extraversion:.2f}) without social behaviors"
                    )

            # Check low extraversion (introversion) alignment
            if profile.extraversion < 0.3:
                introverted_indicators = {
                    "reserved",
                    "quiet",
                    "introverted",
                    "thoughtful",
                }
                has_introverted = any(
                    any(intro in behavior.lower() for intro in introverted_indicators)
                    for behavior in profile.behaviors
                )
                if not has_introverted:
                    inconsistencies.append(
                        f"{name}: Low extraversion ({profile.extraversion:.2f}) without introverted behaviors"
                    )

        consistency_score = 1.0 - (
            len(inconsistencies) / max(1, len(self.kg.personalities) * 3)
        )
        consistency_score = max(0, consistency_score)

        return {
            "consistency_score": consistency_score,
            "inconsistencies_found": len(inconsistencies),
            "inconsistencies": inconsistencies,
        }

    def evaluate_graph_quality(self) -> Dict:
        """Evaluate structural properties of the graph"""
        G = self.kg.graph

        if G.number_of_nodes() == 0:
            return {
                "num_components": 0,
                "connectivity_ratio": 0,
                "avg_clustering": 0,
                "graph_density": 0,
                "avg_confidence": 0,
            }

        # Connected components
        weakly_connected = nx.number_weakly_connected_components(G)
        largest_component = len(max(nx.weakly_connected_components(G), key=len))
        connectivity_ratio = largest_component / G.number_of_nodes()

        # Convert to undirected for clustering
        undirected_G = nx.Graph(G.to_undirected())
        avg_clustering = (
            nx.average_clustering(undirected_G)
            if undirected_G.number_of_nodes() > 0
            else 0
        )

        # Graph density
        graph_density = nx.density(nx.Graph(G))

        # Average confidence of relationships
        confidences = [data.get("confidence", 0.5) for _, _, data in G.edges(data=True)]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        return {
            "num_components": weakly_connected,
            "connectivity_ratio": connectivity_ratio,
            "avg_clustering": avg_clustering,
            "graph_density": graph_density,
            "avg_confidence": avg_confidence,
        }

    def evaluate_relationship_quality(self) -> Dict:
        """Evaluate relationship extraction quality"""
        G = self.kg.graph

        # Count relationships by type
        relation_counts = defaultdict(int)
        for _, _, data in G.edges(data=True):
            rel_type = data.get("relation", "unknown")
            relation_counts[rel_type] += 1

        # High confidence relationships
        high_conf = sum(
            1 for _, _, data in G.edges(data=True) if data.get("confidence", 0) > 0.8
        )
        low_conf = sum(
            1 for _, _, data in G.edges(data=True) if data.get("confidence", 0) < 0.5
        )

        total_edges = G.number_of_edges()

        return {
            "total_relationships": total_edges,
            "unique_relation_types": len(relation_counts),
            "relation_type_distribution": dict(relation_counts),
            "high_confidence_ratio": high_conf / max(1, total_edges),
            "low_confidence_count": low_conf,
        }

    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        completeness = self.evaluate_completeness()
        consistency = self.evaluate_consistency()
        quality = self.evaluate_graph_quality()
        rel_quality = self.evaluate_relationship_quality()
        stats = self.kg.get_statistics()

        report = f"""
========================================================================
                    KNOWLEDGE GRAPH EVALUATION REPORT
========================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. GRAPH STATISTICS
------------------------------------------------------------------------
Total Nodes:              {stats['num_nodes']}
Total Edges:              {stats['num_edges']}
Person Entities:          {stats['num_persons']}
Organization Entities:    {stats['num_orgs']}
Location Entities:        {stats['num_locations']}
Graph Density:            {stats['density']:.3f}
Average Degree:           {stats['avg_degree']:.2f}

2. COMPLETENESS METRICS
------------------------------------------------------------------------
Personality Coverage:     {completeness['personality_coverage']:.2%}
Entities Extracted:       {completeness['entities_extracted']}
Relationships Extracted:  {completeness['relationships_extracted']}
Avg Evidence/Person:      {completeness['avg_evidence_per_person']:.1f}
Overall Score:            {completeness['score']:.2%}

3. CONSISTENCY METRICS
------------------------------------------------------------------------
Consistency Score:        {consistency['consistency_score']:.2%}
Inconsistencies Found:    {consistency['inconsistencies_found']}
"""

        if consistency["inconsistencies"]:
            report += "\nDetected Inconsistencies:\n"
            for inc in consistency["inconsistencies"][:5]:
                report += f"  • {inc}\n"
            if len(consistency["inconsistencies"]) > 5:
                report += f"  ... and {len(consistency['inconsistencies']) - 5} more\n"

        report += f"""
4. GRAPH QUALITY METRICS
------------------------------------------------------------------------
Connected Components:     {quality['num_components']}
Connectivity Ratio:       {quality['connectivity_ratio']:.2%}
Average Clustering:       {quality['avg_clustering']:.3f}
Graph Density:            {quality['graph_density']:.3f}
Avg Relationship Conf:    {quality['avg_confidence']:.3f}

5. RELATIONSHIP QUALITY
------------------------------------------------------------------------
Total Relationships:      {rel_quality['total_relationships']}
Unique Relation Types:    {rel_quality['unique_relation_types']}
High Confidence (>0.8):   {rel_quality['high_confidence_ratio']:.2%}
Low Confidence (<0.5):    {rel_quality['low_confidence_count']}

Relation Type Distribution:
"""

        for rel_type, count in sorted(
            rel_quality["relation_type_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            report += f"  • {rel_type:20s}: {count:3d}\n"

        report += """
6. PERSONALITY ANALYSIS
------------------------------------------------------------------------
"""

        for name, profile in sorted(self.kg.personalities.items()):
            report += f"\n{name}:\n"
            report += f"  Openness:          {profile.openness:.2f}\n"
            report += f"  Conscientiousness: {profile.conscientiousness:.2f}\n"
            report += f"  Extraversion:      {profile.extraversion:.2f}\n"
            report += f"  Agreeableness:     {profile.agreeableness:.2f}\n"
            report += f"  Neuroticism:       {profile.neuroticism:.2f}\n"
            report += f"  Evidence Count:    {profile.evidence_count}\n"
            if profile.behaviors:
                report += f"  Behaviors:         {', '.join(profile.behaviors[:3])}\n"
            if profile.emotional_states:
                report += (
                    f"  Emotions:          {', '.join(profile.emotional_states[:3])}\n"
                )

        report += "\n========================================================================\n"

        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution pipeline"""
    print("=" * 70)
    print("  IMPROVED KNOWLEDGE GRAPH CONSTRUCTION WITH PERSONALITY MODELING")
    print("=" * 70)

    generator = SyntheticDataGenerator()
    documents = [
        ("narrative", generator.generate_narrative_document()),
        ("business", generator.generate_business_document()),
        ("social", generator.generate_social_document()),
    ]

    for doc_name, doc_text in documents:
        print(f"\n{'='*70}")
        print(f"  Processing: {doc_name.upper()} DOCUMENT")
        print(f"{'='*70}\n")

        kg_builder = KnowledgeGraphBuilder()

        try:
            kg_builder.process_document(doc_text)
            kg_builder.export_to_json(f"kg_{doc_name}.json")
            kg_builder.visualize(f"kg_{doc_name}.png")

            evaluator = KnowledgeGraphEvaluator(kg_builder)
            report = evaluator.generate_report()
            print("\n" + report)

            with open(f"evaluation_{doc_name}.txt", "w") as f:
                f.write(report)

            print(f"✓ {doc_name.upper()} document processing completed successfully\n")

        except Exception as e:
            print(f"✗ Error processing {doc_name} document: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
