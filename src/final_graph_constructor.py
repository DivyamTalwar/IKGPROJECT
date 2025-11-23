from neo4j import GraphDatabase
import json
import os

NEO4J_URI = "neo4j+s://9d1a9c1e.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "tOcmGKJfmOAknCqy9Dd1yIFKUI4V_Suj_6slLFolWR4"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE = os.path.join(SCRIPT_DIR, "phase2_extraction_COMPLETE_50_triples.json")

print("NEO4J AURA KNOWLEDGE GRAPH LOADER")
print(f"\nLoading JSON data from: {JSON_FILE}")

try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data['entities'])} entities and {len(data['triples'])} triples")
except FileNotFoundError:
    print(f"ERROR: Could not find file: {JSON_FILE}")
    print(f"\nMake sure 'phase2_extraction_COMPLETE_50_triples.json' is in the same folder as this script!")
    print(f"   Current folder: {SCRIPT_DIR}")
    exit(1)

print(f"\Connecting to Neo4j Aura...")
print(f"   URI: {NEO4J_URI}")
print(f"   User: {NEO4J_USER}")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    driver.verify_connectivity()
    print("Connected successfully!")
except Exception as e:
    print(f"Connection failed: {e}")
    print("\nPlease check your internet connection and Neo4j Aura instance!")
    driver.close()
    exit(1)

print("\nClearing existing data...")
with driver.session(database="neo4j") as session:
    result = session.run("MATCH (n) DETACH DELETE n")
    summary = result.consume()
    print("Database cleared")

print(f"\nCreating {len(data['entities'])} nodes...")
node_count = 0
with driver.session(database="neo4j") as session:
    for i, entity in enumerate(data['entities'], 1):
        try:
            session.run("""
                CREATE (n {
                    id: $id,
                    name: $name,
                    type: $type,
                    role: $role,
                    mentions: $mentions,
                    context: $context
                })
            """, 
            id=entity['entity_id'],
            name=entity['text'],
            type=entity['type'],
            role=entity.get('role', entity.get('category', '')),
            mentions=entity.get('mentions', 0),
            context=entity.get('context', '')
            )
            node_count += 1
            
            if i % 10 == 0:
                print(f"   Created {i}/{len(data['entities'])} nodes...")
        except Exception as e:
            print(f"  Error creating node {entity['entity_id']}: {e}")

print(f"Successfully created {node_count} nodes!")

print(f"\nCreating {len(data['triples'])} relationships...")
rel_count = 0
with driver.session(database="neo4j") as session:
    for i, triple in enumerate(data['triples'], 1):
        try:
            predicate = triple['predicate'].replace('-', '_').replace(' ', '_').replace('/', '_')
            
            session.run(f"""
                MATCH (a {{id: $subject_id}})
                MATCH (b {{id: $object_id}})
                CREATE (a)-[r:{predicate} {{
                    confidence: $confidence,
                    date: $date,
                    context: $context,
                    source_text: $source_text,
                    category: $category
                }}]->(b)
            """,
            subject_id=triple['subject_id'],
            object_id=triple['object_id'],
            confidence=triple['confidence'],
            date=triple.get('date', ''),
            context=triple.get('context', ''),
            source_text=triple.get('source_text', ''),
            category=triple.get('category', '')
            )
            rel_count += 1
            
            if i % 10 == 0:
                print(f"  Created {i}/{len(data['triples'])} relationships...")
        except Exception as e:
            print(f"   Error creating relationship {triple['triple_id']}: {e}")

print(f"Successfully created {rel_count} relationships!")

print("\nVerifying Knowledge Graph...")
with driver.session(database="neo4j") as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    final_node_count = result.single()['count']
    print(f" Total Nodes: {final_node_count}")
    
    result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
    final_rel_count = result.single()['count']
    print(f"   Total Relationships: {final_rel_count}")
    
    result = session.run("""
        MATCH (n) 
        RETURN n.type as type, count(n) as count 
        ORDER BY count DESC 
        LIMIT 5
    """)
    print(f"\n Node Type Distribution:")
    for record in result:
        print(f"      • {record['type']}: {record['count']}")

driver.close()

print("KNOWLEDGE GRAPH LOADED SUCCESSFULLY! 🔥🔥🔥")
print("="*60)
print(f"\n{final_node_count} nodes created")
print(f"{final_rel_count} relationships created")
print(f"\nView your graph at: https://console.neo4j.io")
print(f"   Database: neo4j")
print(f"   Instance: 9d1a9c1e")
