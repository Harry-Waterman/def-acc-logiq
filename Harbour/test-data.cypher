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
CREATE (f1:Flag {type: 'PHISHING', severity: 'HIGH'})
CREATE (f2:Flag {type: 'SPAM', severity: 'MEDIUM'})
CREATE (f3:Flag {type: 'SAFE', severity: 'LOW'})
CREATE (f4:Flag {type: 'SUSPICIOUS', severity: 'MEDIUM'})
CREATE (f5:Flag {type: 'LEGITIMATE', severity: 'LOW'});

// Create Scores
CREATE (s1:Score {value: 0.95, category: 'PHISHING_RISK'})
CREATE (s2:Score {value: 0.65, category: 'SPAM_RISK'})
CREATE (s3:Score {value: 0.15, category: 'PHISHING_RISK'})
CREATE (s4:Score {value: 0.80, category: 'PHISHING_RISK'})
CREATE (s5:Score {value: 0.10, category: 'PHISHING_RISK'});

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
CREATE (e1:Email {
    dateTime: datetime('2024-01-15T10:30:00Z'),
    subject: 'Verify Your Account',
    messageId: 'email-001'
})
CREATE (e2:Email {
    dateTime: datetime('2024-01-16T14:20:00Z'),
    subject: 'Urgent: Update Required',
    messageId: 'email-002'
})
CREATE (e3:Email {
    dateTime: datetime('2024-01-17T09:15:00Z'),
    subject: 'Welcome to Our Service',
    messageId: 'email-003'
})
CREATE (e4:Email {
    dateTime: datetime('2024-01-18T16:45:00Z'),
    subject: 'Security Alert',
    messageId: 'email-004'
})
CREATE (e5:Email {
    dateTime: datetime('2024-01-19T11:00:00Z'),
    subject: 'Monthly Newsletter',
    messageId: 'email-005'
});

// Link Emails to Addresses (FROM)
MATCH (e:Email {messageId: 'email-001'}), (a:Address {email: 'sender@example.com'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {messageId: 'email-002'}), (a:Address {email: 'phisher@suspicious-site.net'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {messageId: 'email-003'}), (a:Address {email: 'user@trusted-service.io'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {messageId: 'email-004'}), (a:Address {email: 'phisher@suspicious-site.net'})
CREATE (e)-[:FROM]->(a);

MATCH (e:Email {messageId: 'email-005'}), (a:Address {email: 'sender@example.com'})
CREATE (e)-[:FROM]->(a);

// Link Emails to Addresses (TO)
MATCH (e:Email {messageId: 'email-001'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {messageId: 'email-002'}), (a:Address {email: 'victim@example.com'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {messageId: 'email-003'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {messageId: 'email-004'}), (a:Address {email: 'victim@example.com'})
CREATE (e)-[:TO]->(a);

MATCH (e:Email {messageId: 'email-005'}), (a:Address {email: 'recipient@legitimate-company.org'})
CREATE (e)-[:TO]->(a);

// Link Emails to URLs
MATCH (e:Email {messageId: 'email-001'}), (u:Url {url: 'https://example.com/verify-account'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {messageId: 'email-002'}), (u:Url {url: 'http://suspicious-site.net/steal-data'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {messageId: 'email-003'}), (u:Url {url: 'https://legitimate-company.org/login'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {messageId: 'email-004'}), (u:Url {url: 'https://phishing-attack.com/fake-login'})
CREATE (e)-[:CONTAINS_URL]->(u);

MATCH (e:Email {messageId: 'email-005'}), (u:Url {url: 'https://trusted-service.io/dashboard'})
CREATE (e)-[:CONTAINS_URL]->(u);

// Link Emails to Flags
MATCH (e:Email {messageId: 'email-001'}), (f:Flag {type: 'SUSPICIOUS'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {messageId: 'email-002'}), (f:Flag {type: 'PHISHING'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {messageId: 'email-003'}), (f:Flag {type: 'LEGITIMATE'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {messageId: 'email-004'}), (f:Flag {type: 'PHISHING'})
CREATE (e)-[:HAS_FLAG]->(f);

MATCH (e:Email {messageId: 'email-005'}), (f:Flag {type: 'SAFE'})
CREATE (e)-[:HAS_FLAG]->(f);

// Link Emails to Scores
MATCH (e:Email {messageId: 'email-001'}), (s:Score {value: 0.80})
CREATE (e)-[:HAS_Score]->(s);

MATCH (e:Email {messageId: 'email-002'}), (s:Score {value: 0.95})
CREATE (e)-[:HAS_Score]->(s);

MATCH (e:Email {messageId: 'email-003'}), (s:Score {value: 0.15})
CREATE (e)-[:HAS_Score]->(s);

MATCH (e:Email {messageId: 'email-004'}), (s:Score {value: 0.95})
CREATE (e)-[:HAS_Score]->(s);

MATCH (e:Email {messageId: 'email-005'}), (s:Score {value: 0.10})
CREATE (e)-[:HAS_Score]->(s);

// Link Emails to Installation IDs
MATCH (e:Email {messageId: 'email-001'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {messageId: 'email-002'}), (i:installationId {id: 'inst-002'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {messageId: 'email-003'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {messageId: 'email-004'}), (i:installationId {id: 'inst-003'})
CREATE (e)-[:OWNER]->(i);

MATCH (e:Email {messageId: 'email-005'}), (i:installationId {id: 'inst-001'})
CREATE (e)-[:OWNER]->(i);

// Return summary
RETURN 'Test data created successfully!' AS message,
       count{(n:Email)} AS emails,
       count{(n:Address)} AS addresses,
       count{(n:Domain)} AS domains,
       count{(n:Url)} AS urls,
       count{(n:Flag)} AS flags,
       count{(n:Score)} AS scores,
       count{(n:installationId)} AS installations,
       count{(n:userId)} AS users;

