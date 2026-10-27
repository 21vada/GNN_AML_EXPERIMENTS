import pandas as pd
from neo4j import GraphDatabase

# Update with your Neo4j credentials if different
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")

print("Connecting to Neo4j and extracting temporal data...")

driver = GraphDatabase.driver(URI, auth=AUTH)
with driver.session() as session:
    query = """
    MATCH (s:Account)-[r:TRANSFERRED]->(rec:Account)
    RETURN s.id AS source, 
           rec.id AS target, 
           r.amount AS amount, 
           r.timestamp AS timestamp, 
           s.created_at AS source_created,   // The new property we just backfilled
           rec.created_at AS target_created, // The new property we just backfilled
           r.is_laundering AS label
    """
    result = session.run(query)
    # Convert directly into a pandas DataFrame
    df = pd.DataFrame([record.data() for record in result])

driver.close()

# Save locally to be uploaded to Colab
export_path = "transactions_temporal.csv"
df.to_csv(export_path, index=False)

print(f"✅ Export complete! {len(df)} transactions saved to {export_path}")