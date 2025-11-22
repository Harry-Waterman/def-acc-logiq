// Test data for Harbour Neo4j database
// This script creates sample data matching the database schema

// Clear existing data
MATCH (n) DETACH DELETE n;

// Create Domains
CREATE (d1:Domain {name: 'example.com'})
CREATE (d2:Domain {name: 'suspicious-site.net'})
CREATE (d3:Domain {name: 'legitimate-company.org'})
CREATE (d4:Domain {name: 'phishing-attack.com'})
CREATE (d5:Domain {name: 'trusted-service.io'});

// Create Addresses
CREATE (a1:Address {email: 'sender@example.com'})
CREATE (a2:Address {email: 'recipient@legitimate-company.org'})
CREATE (a3:Address {email: 'phisher@suspicious-site.net'})
CREATE (a4:Address {email: 'user@trusted-service.io'})
CREATE (a5:Address {email: 'victim@example.com'});

// Create DisplayNames
CREATE (dn1:DisplayName {name: 'John Sender'})
CREATE (dn2:DisplayName {name: 'Legitimate Company Support'})
CREATE (dn3:DisplayName {name: 'Security Alert'})
CREATE (dn4:DisplayName {name: 'Trusted Service Team'})
CREATE (dn5:DisplayName {name: 'Jane Victim'});

// Link Addresses to DisplayNames
MATCH (a:Address {email: 'sender@example.com'}), (dn:DisplayName {name: 'John Sender'})
CREATE (a)-[:HAS_DISPLAY_NAME]->(dn);

MATCH (a:Address {email: 'recipient@legitimate-company.org'}), (dn:DisplayName {name: 'Legitimate Company Support'})
CREATE (a)-[:HAS_DISPLAY_NAME]->(dn);

MATCH (a:Address {email: 'phisher@suspicious-site.net'}), (dn:DisplayName {name: 'Security Alert'})
CREATE (a)-[:HAS_DISPLAY_NAME]->(dn);

MATCH (a:Address {email: 'user@trusted-service.io'}), (dn:DisplayName {name: 'Trusted Service Team'})
CREATE (a)-[:HAS_DISPLAY_NAME]->(dn);

MATCH (a:Address {email: 'victim@example.com'}), (dn:DisplayName {name: 'Jane Victim'})
CREATE (a)-[:HAS_DISPLAY_NAME]->(dn);

// Link Addresses to Domains
MATCH (a:Address {email: 'sender@example.com'}), (d:Domain {name: 'example.com'})
CREATE (a)-[:HAS_DOMAIN]->(d);

MATCH (a:Address {email: 'recipient@legitimate-company.org'}), (d:Domain {name: 'legitimate-company.org'})
CREATE (a)-[:HAS_DOMAIN]->(d);

MATCH (a:Address {email: 'phisher@suspicious-site.net'}), (d:Domain {name: 'suspicious-site.net'})
CREATE (a)-[:HAS_DOMAIN]->(d);

MATCH (a:Address {email: 'user@trusted-service.io'}), (d:Domain {name: 'trusted-service.io'})
CREATE (a)-[:HAS_DOMAIN]->(d);

MATCH (a:Address {email: 'victim@example.com'}), (d:Domain {name: 'example.com'})
CREATE (a)-[:HAS_DOMAIN]->(d);

// Create URLs
CREATE (u1:Url {url: 'https://example.com/verify-account'})
CREATE (u2:Url {url: 'http://suspicious-site.net/steal-data'})
CREATE (u3:Url {url: 'https://legitimate-company.org/login'})
CREATE (u4:Url {url: 'https://phishing-attack.com/fake-login'})
CREATE (u5:Url {url: 'https://trusted-service.io/dashboard'});

// Link URLs to Domains
MATCH (u:Url {url: 'https://example.com/verify-account'}), (d:Domain {name: 'example.com'})
CREATE (u)-[:HAS_DOMAIN]->(d);

MATCH (u:Url {url: 'http://suspicious-site.net/steal-data'}), (d:Domain {name: 'suspicious-site.net'})
CREATE (u)-[:HAS_DOMAIN]->(d);

MATCH (u:Url {url: 'https://legitimate-company.org/login'}), (d:Domain {name: 'legitimate-company.org'})
CREATE (u)-[:HAS_DOMAIN]->(d);

MATCH (u:Url {url: 'https://phishing-attack.com/fake-login'}), (d:Domain {name: 'phishing-attack.com'})
CREATE (u)-[:HAS_DOMAIN]->(d);

MATCH (u:Url {url: 'https://trusted-service.io/dashboard'}), (d:Domain {name: 'trusted-service.io'})
CREATE (u)-[:HAS_DOMAIN]->(d);

// Create Flags
CREATE (f1:Flag {type: 'PHISHING'})
CREATE (f2:Flag {type: 'SPAM'})
CREATE (f3:Flag {type: 'SAFE'})
CREATE (f4:Flag {type: 'SUSPICIOUS'})
CREATE (f5:Flag {type: 'LEGITIMATE'});

// Create Scores
CREATE (s1:Score {value: 0.95})
CREATE (s2:Score {value: 0.65})
CREATE (s3:Score {value: 0.15})
CREATE (s4:Score {value: 0.80})
CREATE (s5:Score {value: 0.10});

// Create Installation IDs and User IDs
CREATE (i1:installationId {id: 'inst-001'})
CREATE (i2:installationId {id: 'inst-002'})
CREATE (i3:installationId {id: 'inst-003'});

CREATE (u1:userId {id: 'user-001'})
CREATE (u2:userId {id: 'user-002'})
CREATE (u3:userId {id: 'user-003'});

// Link Users to Installations
MATCH (u:userId {id: 'user-001'}), (i:installationId {id: 'inst-001'})
CREATE (u)-[:INSTALLED_BY]->(i);

MATCH (u:userId {id: 'user-002'}), (i:installationId {id: 'inst-002'})
CREATE (u)-[:INSTALLED_BY]->(i);

MATCH (u:userId {id: 'user-003'}), (i:installationId {id: 'inst-003'})
CREATE (u)-[:INSTALLED_BY]->(i);

// Create Emails
// Email IDs are generated from sender|recipients|dateTime hash
// e1: sender@example.com | recipient@legitimate-company.org | 2024-01-15T10:30:00Z
CREATE (e1:Email {
    id: 'a1b2c3d4e5f6g7h8',
    dateTime: datetime('2024-01-15T10:30:00Z')
})
// e2: phisher@suspicious-site.net | victim@example.com | 2024-01-16T14:20:00Z
CREATE (e2:Email {
    id: 'b2c3d4e5f6g7h8i9',
    dateTime: datetime('2024-01-16T14:20:00Z')
})
// e3: user@trusted-service.io | recipient@legitimate-company.org | 2024-01-17T09:15:00Z
CREATE (e3:Email {
    id: 'c3d4e5f6g7h8i9j0',
    dateTime: datetime('2024-01-17T09:15:00Z')
})
// e4: phisher@suspicious-site.net | victim@example.com | 2024-01-18T16:45:00Z
CREATE (e4:Email {
    id: 'd4e5f6g7h8i9j0k1',
    dateTime: datetime('2024-01-18T16:45:00Z')
})
// e5: sender@example.com | recipient@legitimate-company.org | 2024-01-19T11:00:00Z
CREATE (e5:Email {
    id: 'e5f6g7h8i9j0k1l2',
    dateTime: datetime('2024-01-19T11:00:00Z')
});

// Link Emails to Addresses (FROM)
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (a:Address {email: 'sender@example.com'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (a:Address {email: 'phisher@suspicious-site.net'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (a:Address {email: 'user@trusted-service.io'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (a:Address {email: 'phisher@suspicious-site.net'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (a:Address {email: 'sender@example.com'})
CREATE (e)-[:FROM]->(a);

// Link Emails to Addresses (TO)
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (a:Address {email: 'victim@example.com'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (a:Address {email: 'victim@example.com'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

// Link Emails to URLs
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (u:Url {url: 'https://example.com/verify-account'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (u:Url {url: 'http://suspicious-site.net/steal-data'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (u:Url {url: 'https://legitimate-company.org/login'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (u:Url {url: 'https://phishing-attack.com/fake-login'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (u:Url {url: 'https://trusted-service.io/dashboard'})
CREATE (e)-[:CONTAINS_URL]->(u);

// Link Emails to Flags
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (f:Flag {type: 'SUSPICIOUS'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (f:Flag {type: 'PHISHING'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (f:Flag {type: 'LEGITIMATE'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (f:Flag {type: 'PHISHING'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (f:Flag {type: 'SAFE'})
CREATE (e)-[:HAS_FLAG]->(f);

// Link Emails to Scores
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (s:Score {value: 0.80})
CREATE (e)-[:HAS_SCORE]->(s);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (s:Score {value: 0.95})
CREATE (e)-[:HAS_SCORE]->(s);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (s:Score {value: 0.15})
CREATE (e)-[:HAS_SCORE]->(s);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (s:Score {value: 0.95})
CREATE (e)-[:HAS_SCORE]->(s);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (s:Score {value: 0.10})
CREATE (e)-[:HAS_SCORE]->(s);

// Link Emails to Installation IDs
MATCH (e:Email {id: 'a1b2c3d4e5f6g7h8'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {id: 'b2c3d4e5f6g7h8i9'}), (i:installationId {id: 'inst-002'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {id: 'c3d4e5f6g7h8i9j0'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {id: 'd4e5f6g7h8i9j0k1'}), (i:installationId {id: 'inst-003'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {id: 'e5f6g7h8i9j0k1l2'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

// Return summary
RETURN 'Test data created successfully!' AS message,
       count{(n:Email)} AS emails,
       count{(n:Address)} AS addresses,
       count{(n:DisplayName)} AS displayNames,
       count{(n:Domain)} AS domains,
       count{(n:Url)} AS urls,
       count{(n:Flag)} AS flags,
       count{(n:Score)} AS scores,
       count{(n:installationId)} AS installations,
       count{(n:userId)} AS users;

