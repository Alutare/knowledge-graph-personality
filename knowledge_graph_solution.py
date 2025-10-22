"""
Knowledge Graph Construction with Personality Modeling
Using Ollama Gemma 3 LLM for relationship extraction
"""

import json
import re
from typing import Dict, List
from dataclasses import dataclass, asdict
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import ollama
import time
import spacy
from difflib import get_close_matches
import itertools

# ============================================================================ 
# DATA STRUCTURES
# ============================================================================

@dataclass
class Entity:
    name: str
    entity_type: str  # PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class Relationship:
    source: str
    target: str
    relation: str
    properties: Dict = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}

@dataclass
class PersonalityProfile:
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float
    behaviors: List[str] = None
    emotional_states: List[str] = None
    
    def __post_init__(self):
        if self.behaviors is None:
            self.behaviors = []
        if self.emotional_states is None:
            self.emotional_states = []

# ============================================================================ 
# SYNTHETIC DATA GENERATOR
# ============================================================================

class SyntheticDataGenerator:
    """Generate realistic synthetic documents for testing"""
    
    @staticmethod
    def generate_narrative_document() -> str:
        """Generate a narrative document with rich personality indicators"""
        return """
Sarah Chen, a seasoned software engineer at TechCorp, has always been known 
for her innovative thinking and creative problem-solving. She eagerly embraces 
new technologies and often proposes unconventional solutions during team meetings. 
Her colleagues describe her as enthusiastic, outgoing, and always willing to help.

Last month, Sarah led a critical project to redesign the company's authentication 
system. Despite the tight deadline, she remained calm and organized, meticulously 
planning every phase of the implementation. Her attention to detail and systematic 
approach impressed both her manager, David Martinez, and the executive team.

However, Sarah sometimes struggles with criticism. During a code review session 
in San Francisco headquarters, when her colleague John Smith pointed out a potential 
security vulnerability, she became visibly upset and defensive. Later, she acknowledged 
the feedback and worked collaboratively with John to fix the issue, demonstrating her 
ability to manage emotions and maintain professional relationships.

Sarah is an active member of the AI Ethics Committee at TechCorp, where she 
passionately advocates for responsible AI development. She frequently volunteers 
to mentor junior developers and organizes community workshops on machine learning 
basics. Her empathetic nature and genuine concern for others' growth make her a 
beloved figure in the tech community.

In her personal life, Sarah tends to worry about work-life balance and sometimes 
experiences anxiety before major presentations. Despite these challenges, she 
maintains a positive outlook and actively seeks feedback to improve her public 
speaking skills.
"""
    
    @staticmethod
    def generate_business_document() -> str:
        """Generate a business-focused document"""
        return """
The merger between GlobalTech Industries and InnovateSoft was announced in 
January 2024, marking a significant shift in the enterprise software landscape. 
CEO Jennifer Liu of GlobalTech praised the deal, stating it would create 
unprecedented opportunities for innovation.

The negotiation process, led by Chief Financial Officer Robert Kim, took nearly 
eight months. Robert demonstrated exceptional patience and analytical thinking 
throughout the complex discussions. He meticulously reviewed every financial 
projection and legal document, never rushing to conclusions despite pressure 
from board members.

Jennifer Liu's leadership style contrasts sharply with Robert's methodical approach. 
She is known for her bold, risk-taking decisions and charismatic personality. 
During investor calls, she exudes confidence and inspires stakeholders with her 
ambitious vision for the future. Some critics, however, describe her as impulsive 
and occasionally dismissive of dissenting opinions.

The merger created tension among employees at both companies. Many InnovateSoft 
developers expressed anxiety about job security and cultural changes. Team lead 
Maria Santos attempted to address these concerns by hosting open forums and 
encouraging transparent communication. Her compassionate approach and active 
listening helped ease the transition.

Meanwhile, GlobalTech's CTO Alex Thompson remained skeptical about the integration 
timeline. Known for his pessimistic but realistic assessments, Alex raised concerns 
about technical incompatibilities between the two platforms. His thorough analysis, 
while sometimes seen as overly cautious, ultimately prevented several costly mistakes.
"""
    
    @staticmethod
    def generate_social_document() -> str:
        """Generate a document focused on social interactions"""
        return """
The annual charity gala in New York City brought together an eclectic mix of 
personalities from the nonprofit sector. Emma Watson, director of the Hope 
Foundation, greeted every guest with genuine warmth and enthusiasm. Her infectious 
energy and extroverted nature made everyone feel welcome and valued.

Michael Chen, a reserved but thoughtful board member, spent most of the evening 
in quiet conversations with small groups. He listened more than he spoke, carefully 
considering each person's perspective before offering his insights. His introverted 
personality doesn't diminish his impact; colleagues respect his deep, analytical 
thinking and reliable judgment.

The event's keynote speaker, Dr. Patricia Johnson, delivered an emotionally 
powerful speech about mental health advocacy. Her openness about her own struggles 
with anxiety resonated deeply with the audience. Patricia's neurotic tendencies, 
which she openly discusses, fuel her passion for supporting others facing similar 
challenges. She founded the Mindful Living Institute to provide resources for 
emotional wellness.

During the fundraising auction, tension arose when businessman Carlos Rodriguez 
aggressively outbid other participants, seeming to prioritize winning over the 
cause itself. His competitive and sometimes abrasive behavior rubbed many attendees 
the wrong way. Emma quietly pulled him aside later, diplomatically explaining the 
event's collaborative spirit. Carlos, though initially resistant, eventually 
apologized and made an additional generous donation.

The evening concluded with impromptu performances by local artists. Jazz musician 
Lisa Park's spontaneous saxophone solo captivated everyone. Her creative, 
free-spirited nature shines through her music, and she often tells people she 
lives by intuition rather than rigid plans. This openness to experience makes 
her performances unpredictable yet magical.
"""

def find_entity_match(name, entity_list, cutoff=0.6):
    matches = get_close_matches(name, entity_list, n=1, cutoff=cutoff)
    return matches[0] if matches else None

# ============================================================================ 
# LLM RELATIONSHIP EXTRACTOR (Two-pass)
# ============================================================================

class LLMRelationshipExtractor:
    def __init__(self, model_name="gemma3:12b"):
        self.model_name = model_name

    def create_prompt(self, text: str) -> str:
        return f"""
You are a knowledge graph extraction assistant.
Extract relationships from the following text.

Return only valid JSON in this format:

[
  {{ "source": "Entity1", "target": "Entity2", "relation": "Relation", "confidence": 0.9 }},
  ...
]

Text:
{text}
"""

    def extract_relationships(self, text: str, entity_list: list) -> list:
        """Extract relationships for entities in entity_list with fuzzy matching"""
        try:
            prompt = self.create_prompt(text)
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            output = response['message']['content']

            # Extract JSON from response
            start = output.find('[')
            end = output.rfind(']')
            if start == -1 or end == -1:
                print(f"⚠️ Model output not JSON: {output}")
                return []

            cleaned_json = output[start:end+1]
            raw = json.loads(cleaned_json)

            relationships = []
            for r in raw:
                src_match = find_entity_match(r['source'], entity_list)
                tgt_match = find_entity_match(r['target'], entity_list)
                if src_match and tgt_match:
                    relationships.append(
                        Relationship(
                            source=src_match,
                            target=tgt_match,
                            relation=r.get('relation') or r.get('relation_type', ''),
                            properties={'confidence': r.get('confidence', 0.9)}
                        )
                    )

            return relationships

        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return []

# ============================================================================ 
# LLM SIMULATOR FOR ENTITY & PERSONALITY 
# ============================================================================

class LLMSimulator:
    nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def extract_entities(text: str) -> List[Entity]:
        doc = LLMSimulator.nlp(text)
        entities = []
        for ent in doc.ents:
            etype = 'OTHER'
            if ent.label_ in {'PERSON'}:
                etype = 'PERSON'
            elif ent.label_ in {'ORG'}:
                etype = 'ORGANIZATION'
            elif ent.label_ in {'GPE', 'LOC'}:
                etype = 'LOCATION'
            entities.append(Entity(name=ent.text, entity_type=etype, properties={"mentions": text.count(ent.text)}))
        # Remove duplicates
        seen = set()
        unique_entities = []
        for e in entities:
            if e.name not in seen:
                seen.add(e.name)
                unique_entities.append(e)
        return unique_entities

    @staticmethod
    def infer_personality(text: str, person_name: str, llm_model="gemma3:12b") -> PersonalityProfile:
        prompt = f"""
Given this passage: "{text}".
Based on descriptions of {person_name}, infer their Big Five personality traits (0-1) and list up to 5 behaviors and 5 emotional states.
Return JSON like:
{{
  "openness": 0.7,
  "conscientiousness": 0.9,
  "extraversion": 0.8,
  "agreeableness": 0.6,
  "neuroticism": 0.3,
  "behaviors": ["mentors", "analyzes"],
  "emotions": ["enthusiastic", "calm"]
}}
"""
        response = ollama.chat(model=llm_model, messages=[{"role":"user","content":prompt}])
        content = response['message']['content']
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1:
            return PersonalityProfile(0.5,0.5,0.5,0.5,0.5)
        data = json.loads(content[start:end+1])
        return PersonalityProfile(
            openness=data.get('openness',0.5),
            conscientiousness=data.get('conscientiousness',0.5),
            extraversion=data.get('extraversion',0.5),
            agreeableness=data.get('agreeableness',0.5),
            neuroticism=data.get('neuroticism',0.5),
            behaviors=data.get('behaviors',[]),
            emotional_states=data.get('emotions',[])
        )

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

    def filter_relationships(raw, entity_list):
        filtered = []
        for r in raw:
            if r['source'] in entity_list and r['target'] in entity_list:
                filtered.append(
                    Relationship(
                        source=r['source'],
                        target=r['target'],
                        relation_type=r.get('relation', ''),  # map 'relation' → 'relation_type'
                        properties={"confidence": r.get("confidence", 0.9)}  # store confidence here
                    )
                )
        return filtered

    def process_document(self, text: str):
        print("STEP 1: Entity Extraction")
        entities = self.llm.extract_entities(text)
        for entity in entities:
            self.entities[entity.name] = entity
            self.graph.add_node(entity.name, type=entity.entity_type, **entity.properties)
        print(f"Extracted {len(entities)} entities")

        print("STEP 2: Relationship Extraction (LLM + co-occurrence + affiliations)")
        entity_names = list(self.entities.keys())

        # 2a. LLM relationships with fuzzy matching
        raw_relationships = self.llm_relationship_extractor.extract_relationships(text, entity_names)
        for rel in raw_relationships:
            self.graph.add_edge(
                rel.source,
                rel.target,
                type=rel.relation,
                **rel.properties
            )

        # 2b. Co-occurrence relationships in the same sentence
        sentences = re.split(r'[.!?]', text)
        for sent in sentences:
            sent_entities = [e.name for e in entities if e.name in sent]
            for a, b in itertools.combinations(sent_entities, 2):
                self.graph.add_edge(a, b, type="co_occurs", confidence=0.8)

        # 2c. Affiliation edges (PERSON -> ORG/EVENT)
        for person in [e for e in entities if e.entity_type == "PERSON"]:
            for other in entities:
                if other.entity_type in ["ORGANIZATION", "EVENT"]:
                    self.graph.add_edge(person.name, other.name, type="affiliated_with", confidence=0.9)

        print(f"Total edges after enhancement: {self.graph.number_of_edges()}")

        print("STEP 3: Personality Inference")
        for entity in entities:
            if entity.entity_type == "PERSON":
                profile = self.llm.infer_personality(text, entity.name)
                self.personalities[entity.name] = profile
                self.graph.nodes[entity.name].update({
                    'openness': profile.openness,
                    'conscientiousness': profile.conscientiousness,
                    'extraversion': profile.extraversion,
                    'agreeableness': profile.agreeableness,
                    'neuroticism': profile.neuroticism,
                    'behaviors': profile.behaviors,
                    'emotions': profile.emotional_states
                })
        print(f"Inferred personalities for {len(self.personalities)} persons")

    def export_to_json(self, filename: str="knowledge_graph.json"):
        data = {
            'entities': [asdict(e) for e in self.entities.values()],
            'relationships': [asdict(r) for r in self.relationships],
            'personalities': {name: asdict(profile) for name, profile in self.personalities.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {filename}")

    def get_statistics(self) -> Dict:
        """Compute graph statistics"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_persons': len([e for e in self.entities.values() 
                               if e.entity_type == "PERSON"]),
            'num_orgs': len([e for e in self.entities.values() 
                            if e.entity_type == "ORGANIZATION"]),
            'num_locations': len([e for e in self.entities.values() 
                                 if e.entity_type == "LOCATION"]),
            'density': nx.density(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes())
        }
    
    def visualize(self, filename: str = "knowledge_graph.png") -> None:
        """Visualize the knowledge graph"""
        plt.figure(figsize=(16, 12))
        
        # Define colors by entity type
        color_map = {
            'PERSON': '#FF6B6B',
            'ORGANIZATION': '#4ECDC4',
            'LOCATION': '#95E1D3',
            'EVENT': '#FFA07A',
            'CONCEPT': '#DDA0DD'
        }
        
        node_colors = [
            color_map.get(self.graph.nodes[node].get('type', 'CONCEPT'), '#CCCCCC')
            for node in self.graph.nodes()
        ]
        
        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=3000,
            alpha=0.9
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.5,
            width=2
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=9,
            font_weight='bold'
        )
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'type')
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels,
            font_size=7
        )
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=etype)
            for etype, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title("Knowledge Graph with Personality Modeling", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {filename}")

# ============================================================================ 
# EVALUATOR
# ============================================================================

class KnowledgeGraphEvaluator:
    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg = kg_builder

    def evaluate_completeness(self) -> Dict:
        """
        Measure how complete the extraction is
        """
        stats = self.kg.get_statistics()
        
        # Check if persons have personality profiles
        persons = [e for e in self.kg.entities.values() if e.entity_type == "PERSON"]
        persons_with_personality = len(self.kg.personalities)
        
        completeness_score = persons_with_personality / max(1, len(persons))
        
        return {
            'personality_coverage': completeness_score,
            'entities_extracted': stats['num_nodes'],
            'relationships_extracted': stats['num_edges'],
            'score': completeness_score
        }
    
    def evaluate_consistency(self) -> Dict:
        """
        Check for consistency in personality traits
        """
        inconsistencies = []
        
        for name, profile in self.kg.personalities.items():
            # Check trait ranges
            traits = [
                profile.openness, profile.conscientiousness,
                profile.extraversion, profile.agreeableness,
                profile.neuroticism
            ]
            
            if any(t < 0 or t > 1 for t in traits):
                inconsistencies.append(f"{name}: Trait out of range")
            
            # Check if high neuroticism aligns with negative emotions
            if profile.neuroticism > 0.7:
                negative_emotions = {'anxious', 'worried', 'upset', 'nervous'}
                if not any(e in profile.emotional_states for e in negative_emotions):
                    inconsistencies.append(
                        f"{name}: High neuroticism without negative emotions"
                    )
        
        consistency_score = 1.0 - (len(inconsistencies) / max(1, len(self.kg.personalities)))
        
        return {
            'consistency_score': consistency_score,
            'inconsistencies_found': len(inconsistencies),
            'inconsistencies': inconsistencies
        }
    
    def evaluate_graph_quality(self) -> Dict:
        """
        Evaluate structural properties of the graph
        """
        G = self.kg.graph

        # Connected components
        if G.number_of_nodes() > 0:
            weakly_connected = nx.number_weakly_connected_components(G)
            largest_component = len(max(nx.weakly_connected_components(G), key=len))
            connectivity_ratio = largest_component / G.number_of_nodes()
        else:
            weakly_connected = 0
            connectivity_ratio = 0

        # Convert MultiDiGraph -> simple undirected graph
        undirected_G = nx.Graph(G.to_undirected())

        # Average clustering
        avg_clustering = (
            nx.average_clustering(undirected_G)
            if undirected_G.number_of_nodes() > 0
            else 0
        )

        # Graph density (using simple graph)
        graph_density = nx.density(nx.Graph(G))

        return {
            'num_components': weakly_connected,
            'connectivity_ratio': connectivity_ratio,
            'avg_clustering': avg_clustering,
            'graph_density': graph_density
        }

    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        completeness = self.evaluate_completeness()
        consistency = self.evaluate_consistency()
        quality = self.evaluate_graph_quality()
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
Overall Score:            {completeness['score']:.2%}

3. CONSISTENCY METRICS
------------------------------------------------------------------------
Consistency Score:        {consistency['consistency_score']:.2%}
Inconsistencies Found:    {consistency['inconsistencies_found']}

4. GRAPH QUALITY METRICS
------------------------------------------------------------------------
Connected Components:     {quality['num_components']}
Connectivity Ratio:       {quality['connectivity_ratio']:.2%}
Average Clustering:       {quality['avg_clustering']:.3f}
Graph Density:            {quality['graph_density']:.3f}

5. PERSONALITY ANALYSIS
------------------------------------------------------------------------
"""
        
        for name, profile in self.kg.personalities.items():
            report += f"\n{name}:\n"
            report += f"  Openness:          {profile.openness:.2f}\n"
            report += f"  Conscientiousness: {profile.conscientiousness:.2f}\n"
            report += f"  Extraversion:      {profile.extraversion:.2f}\n"
            report += f"  Agreeableness:     {profile.agreeableness:.2f}\n"
            report += f"  Neuroticism:       {profile.neuroticism:.2f}\n"
            report += f"  Behaviors:         {', '.join(profile.behaviors[:3])}\n"
            report += f"  Emotions:          {', '.join(profile.emotional_states[:3])}\n"
        
        report += "\n========================================================================\n"
        
        return report

# ============================================================================ 
# MAIN EXECUTION
# ============================================================================

def main():
    generator = SyntheticDataGenerator()
    documents = [
        ("narrative", generator.generate_narrative_document()),
        ("business", generator.generate_business_document()),
        ("social", generator.generate_social_document())
    ]

    for doc_name, doc_text in documents:
        print(f"\n{'='*60}\nProcessing: {doc_name.upper()} DOCUMENT\n{'='*60}\n")
        kg_builder = KnowledgeGraphBuilder()
        kg_builder.process_document(doc_text)
        kg_builder.export_to_json(f"kg_{doc_name}.json")
        kg_builder.visualize(f"kg_{doc_name}.png")
        evaluator = KnowledgeGraphEvaluator(kg_builder)
        report = evaluator.generate_report()
        print(report)
        with open(f"evaluation_{doc_name}.txt","w") as f:
            f.write(report)

    print("\nPipeline completed successfully.")

if __name__=="__main__":
    main()
