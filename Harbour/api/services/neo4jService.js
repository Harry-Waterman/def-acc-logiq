import neo4j from 'neo4j-driver';
import crypto from 'crypto';

// Neo4j connection configuration
const NEO4J_URI = process.env.NEO4J_URI || 'bolt://neo4j:7687';
const NEO4J_USERNAME = process.env.NEO4J_USERNAME || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || 'password';

// Create Neo4j driver
const driver = neo4j.driver(
  NEO4J_URI,
  neo4j.auth.basic(NEO4J_USERNAME, NEO4J_PASSWORD)
);

/**
 * Extract domain from email address
 */
function extractDomain(email) {
  if (!email || typeof email !== 'string') return null;
  const parts = email.split('@');
  return parts.length === 2 ? parts[1] : null;
}

/**
 * Extract domain from URL
 */
function extractDomainFromUrl(url) {
  if (!url || typeof url !== 'string') return null;
  try {
    const urlObj = new URL(url);
    return urlObj.hostname.replace(/^www\./, '');
  } catch (e) {
    // If URL parsing fails, try simple extraction
    const match = url.match(/https?:\/\/(?:www\.)?([^\/]+)/);
    return match ? match[1] : null;
  }
}

/**
 * Create or get a node (merges if exists)
 */
async function mergeNode(session, label, properties, uniqueKey) {
  // Filter out null/undefined values and the unique key
  const propsToSet = {};
  Object.keys(properties).forEach(key => {
    if (key !== uniqueKey && properties[key] !== null && properties[key] !== undefined) {
      propsToSet[key] = properties[key];
    }
  });
  
  // Build SET clause for properties to update
  const setProps = Object.keys(propsToSet)
    .map(key => `n.${key} = $${key}`)
    .join(', ');
  
  const query = `
    MERGE (n:${label} {${uniqueKey}: $${uniqueKey}})
    ${setProps ? `SET ${setProps}` : ''}
    RETURN n
  `;
  
  const params = {
    [uniqueKey]: properties[uniqueKey],
    ...propsToSet
  };
  
  const result = await session.run(query, params);
  
  return result.records[0]?.get('n');
}

/**
 * Create relationship between two nodes
 */
async function createRelationship(
  session,
  fromLabel,
  fromKey,
  fromValue,
  relationshipType,
  toLabel,
  toKey,
  toValue
) {
  const query = `
    MATCH (from:${fromLabel} {${fromKey}: $fromValue})
    MATCH (to:${toLabel} {${toKey}: $toValue})
    MERGE (from)-[r:${relationshipType}]->(to)
    RETURN r
  `;
  
  await session.run(query, {
    fromValue,
    toValue
  });
}

/**
 * Generate a unique identifier for an email based on its content
 */
function generateEmailId(from, to, dateTime) {
  const toArray = Array.isArray(to) ? to : [to];
  const toStr = toArray.filter(Boolean).sort().join(',');
  const content = `${from || ''}|${toStr}|${dateTime || ''}`;
  return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
}

/**
 * Main function to create email graph from JSON payload
 */
export async function createEmailGraph(emailData) {
  const session = driver.session();
  
  try {
    // Extract data from payload
    const {
      from,
      to,
      dateTime,
      urls = [],
      flags = [],
      score,
      installationId,
      userId
    } = emailData;

    // Validate required fields
    if (!from) {
      throw new Error('from is required');
    }

    // Convert dateTime to ISO string for Neo4j (or use current time if not provided)
    const emailDateTime = dateTime ? dateTime : new Date().toISOString();
    
    // Generate unique identifier for email
    const emailId = generateEmailId(from, to, emailDateTime);
    
    // Create Email node
    const emailNode = await mergeNode(
      session,
      'Email',
      {
        id: emailId,
        dateTime: emailDateTime
      },
      'id'
    );

    // Process FROM address
    if (from) {
      const fromDomain = extractDomain(from);
      
      // Create FROM Address
      await mergeNode(session, 'Address', { email: from }, 'email');
      
      // Create relationship Email -> FROM -> Address
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'FROM',
        'Address',
        'email',
        from
      );
      
      // Create Domain and link if domain exists
      if (fromDomain) {
        await mergeNode(session, 'Domain', { name: fromDomain }, 'name');
        await createRelationship(
          session,
          'Address',
          'email',
          from,
          'HAS_DOMAIN',
          'Domain',
          'name',
          fromDomain
        );
      }
    }

    // Process TO addresses (can be array or single)
    const toAddresses = Array.isArray(to) ? to : [to];
    for (const toAddr of toAddresses) {
      if (!toAddr) continue;
      
      const toDomain = extractDomain(toAddr);
      
      // Create TO Address
      await mergeNode(session, 'Address', { email: toAddr }, 'email');
      
      // Create relationship Email -> TO -> Address
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'TO',
        'Address',
        'email',
        toAddr
      );
      
      // Create Domain and link if domain exists
      if (toDomain) {
        await mergeNode(session, 'Domain', { name: toDomain }, 'name');
        await createRelationship(
          session,
          'Address',
          'email',
          toAddr,
          'HAS_DOMAIN',
          'Domain',
          'name',
          toDomain
        );
      }
    }

    // Process URLs
    const urlArray = Array.isArray(urls) ? urls : [];
    for (const url of urlArray) {
      if (!url) continue;
      
      const urlDomain = extractDomainFromUrl(url);
      
      // Create URL node
      await mergeNode(session, 'Url', { url }, 'url');
      
      // Create relationship Email -> CONTAINS_URL -> Url
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'CONTAINS_URL',
        'Url',
        'url',
        url
      );
      
      // Create Domain and link if domain exists
      if (urlDomain) {
        await mergeNode(session, 'Domain', { name: urlDomain }, 'name');
        await createRelationship(
          session,
          'Url',
          'url',
          url,
          'HAS_DOMAIN',
          'Domain',
          'name',
          urlDomain
        );
      }
    }

    // Process Flags (array of strings)
    const flagArray = Array.isArray(flags) ? flags : [];
    for (const flag of flagArray) {
      if (!flag || typeof flag !== 'string') continue;
      
      // Create Flag node
      await mergeNode(
        session,
        'Flag',
        { type: flag },
        'type'
      );
      
      // Create relationship Email -> HAS_FLAG -> Flag
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'HAS_FLAG',
        'Flag',
        'type',
        flag
      );
    }

    // Process Score (single number)
    if (score !== undefined && score !== null && typeof score === 'number') {
      // Create Score node
      const scoreQuery = `
        MERGE (s:Score {value: $value})
        ON CREATE SET s.value = $value
        ON MATCH SET s.value = $value
        WITH s
        MATCH (e:Email {id: $emailId})
        MERGE (e)-[r:HAS_Score]->(s)
        RETURN r
      `;
      await session.run(scoreQuery, {
        emailId,
        value: score
      });
    }

    // Process Installation ID
    if (installationId) {
      await mergeNode(
        session,
        'installationId',
        { id: installationId },
        'id'
      );
      
      // Create relationship Email -> OWNER -> installationId
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'OWNER',
        'installationId',
        'id',
        installationId
      );
    }

    // Process User ID and link to Installation
    if (userId && installationId) {
      await mergeNode(session, 'userId', { id: userId }, 'id');
      
      // Create relationship userId -> INSTALLED_BY -> installationId
      await createRelationship(
        session,
        'userId',
        'id',
        userId,
        'INSTALLED_BY',
        'installationId',
        'id',
        installationId
      );
    }

    return {
      emailId,
      status: 'created',
      timestamp: new Date().toISOString()
    };
    
  } finally {
    await session.close();
  }
}

/**
 * Close Neo4j driver connection
 */
export async function closeDriver() {
  await driver.close();
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
  console.log('Closing Neo4j driver...');
  await closeDriver();
  process.exit(0);
});

