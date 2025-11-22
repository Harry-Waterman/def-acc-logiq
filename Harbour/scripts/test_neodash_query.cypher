// Test query to check if dashboard is stored correctly for NeoDash
// Run this in Neo4j Browser to verify the dashboard structure

// 1. Check if dashboard exists
MATCH (d:_Neodash_Dashboard)
RETURN d.uuid as uuid, 
       d.title as title,
       d.dashboard IS NOT NULL as has_dashboard_prop,
       d.json IS NOT NULL as has_json_prop,
       labels(d) as labels

// 2. Check the dashboard JSON structure
MATCH (d:_Neodash_Dashboard)
RETURN d.uuid as uuid,
       d.title as title,
       size(d.dashboard) as dashboard_size,
       substring(d.dashboard, 0, 100) as dashboard_preview

// 3. Try to parse and verify the JSON structure
MATCH (d:_Neodash_Dashboard)
WITH d, d.dashboard as dashboard_json
WHERE dashboard_json IS NOT NULL
RETURN d.uuid as uuid,
       d.title as title,
       apoc.convert.fromJsonMap(dashboard_json).title as parsed_title,
       apoc.convert.fromJsonMap(dashboard_json).version as parsed_version,
       size(apoc.convert.fromJsonMap(dashboard_json).pages) as pages_count

// 4. What NeoDash likely queries (simulated)
// NeoDash probably does something like:
MATCH (d:_Neodash_Dashboard {uuid: $uuid})
RETURN d.dashboard as dashboard
// or
MATCH (d:_Neodash_Dashboard)
WHERE d.uuid = $uuid OR d.title = $title
RETURN d.dashboard as dashboard, d.json as json

