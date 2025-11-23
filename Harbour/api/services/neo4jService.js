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
 * Extract email from recipient (handles both object and string formats)
 */
function extractRecipientEmail(recipient) {
  if (!recipient) return null;
  if (typeof recipient === 'object' && recipient !== null) {
    return recipient.email || null;
  }
  if (typeof recipient === 'string') {
    // Check if it's an email format (contains @)
    if (recipient.includes('@')) {
      return recipient;
    }
    // Otherwise it's a display name, return null
    return null;
  }
  return null;
}

/**
 * Extract display name from recipient (handles both object and string formats)
 */
function extractRecipientDisplayName(recipient) {
  if (!recipient) return null;
  if (typeof recipient === 'object' && recipient !== null) {
    return recipient.displayName || null;
  }
  if (typeof recipient === 'string') {
    // If it's an email format, return null (no display name)
    if (recipient.includes('@')) {
      return null;
    }
    // Otherwise it's a display name
    return recipient;
  }
  return null;
}

/**
 * Generate a unique identifier for an email based on its content
 */
function generateEmailId(sender, recipients, dateTime) {
  const recipientArray = Array.isArray(recipients) ? recipients : [recipients];
  // Extract emails from recipients (handles both object and string formats)
  const recipientEmails = recipientArray
    .map(extractRecipientEmail)
    .filter(Boolean)
    .sort();
  const recipientStr = recipientEmails.join(',');
  const content = `${sender || ''}|${recipientStr}|${dateTime || ''}`;
  return crypto.createHash('sha256').update(content).digest('hex').substring(0, 16);
}

/**
 * Main function to create email graph from JSON payload
 */
export async function createEmailGraph(emailData) {
  const session = driver.session();
  
  try {
    // Extract data from nested payload structure
    const {
      classification = {},
      email = {},
      installationId,
      userId
    } = emailData;

    // Extract classification data
    const score = classification?.score;
    const flags = classification?.flags || [];

    // Validate email object exists first
    if (!email) {
      throw new Error('email object is required in payload');
    }

    // Debug logging to help diagnose issues
    console.log('Received email object:', JSON.stringify(email, null, 2).substring(0, 1000));

    // Extract email data - validate sender structure
    if (!email.sender) {
      throw new Error(
        `email.sender is required. Received email structure: ${JSON.stringify(email).substring(0, 500)}`
      );
    }

    let sender;
    let displayName;
    
    // Handle both object and string formats for sender
    if (typeof email.sender === 'object' && email.sender !== null) {
      sender = email.sender.email;
      displayName = email.sender.displayName;
      console.log('Extracted sender from object:', sender);
    } else if (typeof email.sender === 'string') {
      sender = email.sender;
      console.log('Extracted sender from string:', sender);
    } else {
      console.log('Unexpected sender type:', typeof email.sender, email.sender);
    }
    
    const recipients = email.recipients || [];
    const dateTime = email.sentTime;
    const urls = email.urls || [];
    const attachments = email.attachments || [];

    // Validate required fields with detailed error messages
    if (!sender) {
      // Provide helpful error message with what we actually received
      throw new Error(
        `email.sender.email is required. ` +
        `Received sender: ${JSON.stringify(email.sender)}, ` +
        `Full email object: ${JSON.stringify(email).substring(0, 500)}`
      );
    }

    // Ensure recipients is an array
    const recipientList = Array.isArray(recipients) ? recipients : (recipients ? [recipients] : []);

    // Convert dateTime to ISO string for Neo4j (or use current time if not provided)
    const emailDateTime = dateTime ? dateTime : new Date().toISOString();
    
    // Generate unique identifier for email
    const emailId = generateEmailId(sender, recipientList, emailDateTime);
    
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

    // Process FROM address (sender)
    if (sender) {
      const fromDomain = extractDomain(sender);
      
      // Create FROM Address
      await mergeNode(session, 'Address', { email: sender }, 'email');
      
      // Create DisplayName node if available
      if (displayName) {
        await mergeNode(session, 'DisplayName', { name: displayName }, 'name');
        
        // Create relationship Address -> HAS_DISPLAY_NAME -> DisplayName
        await createRelationship(
          session,
          'Address',
          'email',
          sender,
          'HAS_DISPLAY_NAME',
          'DisplayName',
          'name',
          displayName
        );
      }
      
      // Create relationship Email -> FROM -> Address
      await createRelationship(
        session,
        'Email',
        'id',
        emailId,
        'FROM',
        'Address',
        'email',
        sender
      );
      
      // Create Domain and link if domain exists
      if (fromDomain) {
        await mergeNode(session, 'Domain', { name: fromDomain }, 'name');
        await createRelationship(
          session,
          'Address',
          'email',
          sender,
          'HAS_DOMAIN',
          'Domain',
          'name',
          fromDomain
        );
      }
    }

    // Process TO addresses (recipients) - can be array or single
    // Handle both object and string formats for recipients
    const toAddresses = Array.isArray(recipientList) ? recipientList : [recipientList];
    for (const recipient of toAddresses) {
      if (!recipient) continue;
      
      let toEmail;
      let toDisplayName;
      
      // Handle both object and string formats for recipient
      if (typeof recipient === 'object' && recipient !== null) {
        toEmail = recipient.email;
        toDisplayName = recipient.displayName;
        console.log('Extracted recipient from object:', toEmail);
      } else if (typeof recipient === 'string') {
        // Check if it's an email format (contains @)
        if (recipient.includes('@')) {
          toEmail = recipient;
          toDisplayName = null;
        } else {
          // It's a display name without email
          toEmail = null;
          toDisplayName = recipient;
        }
        console.log('Extracted recipient from string:', toEmail || toDisplayName);
      } else {
        console.log('Unexpected recipient type:', typeof recipient, recipient);
        continue;
      }
      
      // Skip if we don't have at least an email or display name
      if (!toEmail && !toDisplayName) continue;
      
      // If we have an email, process it as an Address
      if (toEmail) {
        const toDomain = extractDomain(toEmail);
        
        // Create TO Address
        await mergeNode(session, 'Address', { email: toEmail }, 'email');
        
        // Create DisplayName node if available
        if (toDisplayName) {
          await mergeNode(session, 'DisplayName', { name: toDisplayName }, 'name');
          
          // Create relationship Address -> HAS_DISPLAY_NAME -> DisplayName
          await createRelationship(
            session,
            'Address',
            'email',
            toEmail,
            'HAS_DISPLAY_NAME',
            'DisplayName',
            'name',
            toDisplayName
          );
        }
        
        // Create relationship Email -> TO -> Address
        await createRelationship(
          session,
          'Email',
          'id',
          emailId,
          'TO',
          'Address',
          'email',
          toEmail
        );
        
        // Create Domain and link if domain exists
        if (toDomain) {
          await mergeNode(session, 'Domain', { name: toDomain }, 'name');
          await createRelationship(
            session,
            'Address',
            'email',
            toEmail,
            'HAS_DOMAIN',
            'Domain',
            'name',
            toDomain
          );
        }
      } else if (toDisplayName) {
        // If we only have a display name (no email), create DisplayName node
        // but we can't create an Address without an email, so just create the DisplayName
        await mergeNode(session, 'DisplayName', { name: toDisplayName }, 'name');
        // Note: We don't create a TO relationship here since we need an Address node
        // which requires an email. This is a limitation but handles the edge case.
        console.log('Recipient has display name but no email:', toDisplayName);
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

    // Process Score (can be string or number)
    if (score !== undefined && score !== null) {
      // Convert string score to number if needed
      const scoreValue = typeof score === 'string' ? parseFloat(score) : score;
      if (!isNaN(scoreValue) && typeof scoreValue === 'number') {
        // Create Score node
        const scoreQuery = `
          MERGE (s:Score {value: $value})
          ON CREATE SET s.value = $value
          ON MATCH SET s.value = $value
          WITH s
          MATCH (e:Email {id: $emailId})
          MERGE (e)-[r:HAS_SCORE]->(s)
          RETURN r
        `;
        await session.run(scoreQuery, {
          emailId,
          value: scoreValue
        });
      }
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

