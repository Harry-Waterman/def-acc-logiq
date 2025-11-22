#!/usr/bin/env python3
import os
import json
import sys
import time
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

def wait_for_neo4j(uri, user, password, max_retries=30, delay=2):
    """Wait for Neo4j to be ready"""
    print("Waiting for Neo4j to be ready...")
    for i in range(max_retries):
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            driver.close()
            print("Neo4j is ready!")
            return True
        except (ServiceUnavailable, Exception) as e:
            if i < max_retries - 1:
                print(f"Neo4j not ready yet, retrying... ({i+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"ERROR: Neo4j did not become ready in time: {e}")
                return False
    return False

def seed_dashboard():
    # Get environment variables
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
    neo4j_user = os.getenv('NEO4J_USER')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    neo4j_db = os.getenv('NEO4J_DB', 'neo4j')
    dashboard_path = os.getenv('DASHBOARD_PATH', '/seed/habour_overview.json')
    
    if not neo4j_user or not neo4j_password:
        print("ERROR: NEO4J_USER and NEO4J_PASSWORD must be set")
        sys.exit(1)
    
    # Wait for Neo4j to be ready
    if not wait_for_neo4j(neo4j_uri, neo4j_user, neo4j_password):
        sys.exit(1)
    
    # Read dashboard JSON
    print(f"Reading dashboard from {dashboard_path}...")
    try:
        with open(dashboard_path, 'r') as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in dashboard file: {e}")
        sys.exit(1)
    
    # Connect to Neo4j
    print("Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    except Exception as e:
        print(f"ERROR: Failed to connect to Neo4j: {e}")
        sys.exit(1)
    
    # Insert dashboard
    print("Seeding dashboard...")
    try:
        with driver.session(database=neo4j_db) as session:
            # NeoDash stores dashboards as _Neodash_Dashboard nodes
            # NeoDash stores dashboards as JSON strings, not nested objects
            # Store the dashboard JSON as a string property
            # Use UUID as the unique identifier if available
            dashboard_uuid = dashboard_json.get("uuid")
            dashboard_title = dashboard_json.get("title", "Harbour Dashboard")
            # Ensure the JSON is properly formatted (compact, no extra whitespace)
            dashboard_json_str = json.dumps(dashboard_json, separators=(',', ':'))
            
            # Verify the JSON has required structure
            if "pages" not in dashboard_json:
                print("WARNING: Dashboard JSON missing 'pages' property")
            if not isinstance(dashboard_json.get("pages"), list):
                print("WARNING: Dashboard 'pages' is not a list")
            
            if dashboard_uuid:
                # Use UUID as the merge key (most reliable)
                # NeoDash expects the dashboard JSON in the 'content' property
                # Store in 'content' as primary, and also in 'dashboard' and 'json' for compatibility
                result = session.run(
                    """
                    MERGE (d:_Neodash_Dashboard {uuid: $uuid})
                    SET d.title = $title,
                        d.uuid = $uuid,
                        d.content = $content,
                        d.updatedAt = datetime()
                    RETURN d.title as title, d.uuid as uuid, labels(d) as labels, 
                           d.content IS NOT NULL as has_content,
                           size(d.content) as content_size
                    """,
                    uuid=dashboard_uuid,
                    title=dashboard_title,
                    content=dashboard_json_str
                )
            else:
                # Fallback to title if no UUID
                result = session.run(
                    """
                    MERGE (d:_Neodash_Dashboard {title: $title})
                    SET d.title = $title,
                        d.content = $content,
                        d.updatedAt = datetime()
                    RETURN d.title as title, d.uuid as uuid, labels(d) as labels, 
                           d.content IS NOT NULL as has_content,
                           size(d.content) as content_size
                    """,
                    title=dashboard_title,
                    content=dashboard_json_str
                )
            
            record = result.single()
            if record:
                labels = record.get('labels', [])
                has_content = record.get('has_content', False)
                content_size = record.get('content_size', 0)
                print(f"✓ Dashboard '{record['title']}' (UUID: {record.get('uuid', 'N/A')}) created/updated successfully.")
                print(f"  Labels: {', '.join(labels)}")
                print(f"  Content property stored: {has_content} ({content_size} characters)")
                if 'Dashboard' in labels and '_Neodash_Dashboard' not in labels:
                    print("  ⚠ WARNING: Node has Dashboard label but not _Neodash_Dashboard!")
                elif '_Neodash_Dashboard' in labels:
                    print("  ✓ Correct label _Neodash_Dashboard is present")
                
                # Verify the stored JSON can be parsed
                if dashboard_uuid:
                    verify_result = session.run("""
                        MATCH (d:_Neodash_Dashboard {uuid: $uuid})
                        RETURN d.content as content_json
                    """, uuid=dashboard_uuid)
                else:
                    verify_result = session.run("""
                        MATCH (d:_Neodash_Dashboard {title: $title})
                        RETURN d.content as content_json
                    """, title=dashboard_title)
                
                verify_record = verify_result.single()
                if verify_record and verify_record['content_json']:
                    try:
                        parsed = json.loads(verify_record['content_json'])
                        if 'pages' in parsed and isinstance(parsed['pages'], list):
                            print(f"  ✓ Dashboard JSON is valid and contains {len(parsed['pages'])} page(s)")
                        else:
                            print("  ⚠ WARNING: Dashboard JSON missing 'pages' array")
                    except json.JSONDecodeError as e:
                        print(f"  ⚠ WARNING: Dashboard JSON is not valid JSON: {e}")
                else:
                    print("  ⚠ WARNING: Content property is NULL or empty")
            else:
                print("WARNING: Dashboard operation completed but no record returned.")
    except Exception as e:
        print(f"ERROR: Failed to seed dashboard: {e}")
        import traceback
        traceback.print_exc()
        driver.close()
        sys.exit(1)
    finally:
        driver.close()

if __name__ == "__main__":
    seed_dashboard()

