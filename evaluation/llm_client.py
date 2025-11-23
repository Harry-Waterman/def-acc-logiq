import requests
import json
import re
import time
from typing import Dict, List, Any, Optional

# System prompt from popup.ts
SYSTEM_PROMPT = """
SYSTEM:
You are an email-security classifier.
You will receive the following fields of an email:
from_address, subject, recipients, attachment_names, urls, body.

Your job is to output a single JSON object containing:

{
  "score": "0-100",
  "reasons": []
}

Your rules:
	1.	The score must be a number from 0 to 100.
	2.	Only include items in "reasons" if the score is 50 or higher.
	3.	If the score is below 50, "reasons" must be an empty array ([]).
	4.	When reasons are included, they must come only from this list:
	•	"Sender address doesn't match display name”
	•	“Generic Greetings”
	•	“Urgent or threatening language”
	•	“Suspicious urls”
	•	“suspicious attachment names unrelated to the email subject or body”
	•	“spelling and grammar mistakes”
	•	“too good to be true offers”
	•	“requests for personal information”
	5.	Do not invent new reasons.
	6.	Only include reasons that actually appear in the email content.
	7.	Do not output explanations outside the JSON.
	8.	Do not output placeholder text.
	9.	Output only the JSON. No extra text.

Few-shot examples (to anchor behaviour)

Example A (score below 50 → empty reasons array)

Email content summary: harmless internal update
Output format to copy:

{
  "score": "12",
  "reasons": []
}

Example B (score above 50 → reasons list required)

Email content summary: classic phishing asking for bank login
Output format to copy:

{
  "score": "87",
  "reasons": [
    "Suspicious urls",
    "requests for personal information"
  ]
}
Now classify this email:
"""

class LLMClient:
    def __init__(self, api_url: str = "http://localhost:1234/v1", api_key: str = "", model: str = "local-model", temperature: float = 0.1, retry_on_rate_limit: bool = False):
        self.api_url = api_url.rstrip('/')
        if not self.api_url.endswith('/chat/completions'):
             self.api_url += '/chat/completions'
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.retry_on_rate_limit = retry_on_rate_limit

    def classify_email(self, email_data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Sends email data to the LLM for classification.
        
        Args:
            email_data: Dict containing 'sender', 'subject', 'body', 'urls', etc.
            timeout: Timeout in seconds for the request.
        
        Returns:
            Dict with 'score' (int) and 'reasons' (List[str])
        """
        
        # Prepare context similarly to popup.ts
        # Map fields to what prompt expects: from_address, subject, recipients, attachment_names, urls, body
        
        # Truncate body if too long (approx 8000 chars as in popup.ts)
        body = email_data.get('body', '')
        if len(body) > 8000:
            body = body[:8000] + "\n...[TRUNCATED]..."
            
        # Limit URLs
        urls = email_data.get('urls', [])
        # In our unified dataset, 'urls' is a string (CSV or simple string) or empty
        # convert to list if string
        if isinstance(urls, str) and urls:
            urls_list = [u.strip() for u in urls.split(',')]
        else:
            urls_list = []
            
        if len(urls_list) > 2:
            urls_list = urls_list[:2] + [f"...and {len(urls_list) - 2} more"]

        context_obj = {
            "from_address": email_data.get('sender', ''),
            "subject": email_data.get('subject', ''),
            "recipients": email_data.get('receiver', ''),
            "attachment_names": email_data.get('attachment_names', []), 
            "urls": urls_list,
            "body": body
        }
        
        optimized_context = json.dumps(context_obj, indent=2)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this email:\n{optimized_context}"}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "stream": False
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_retries = 5 if self.retry_on_rate_limit else 1
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                # Auto-retry logic for response_format error (common with some local servers)
                if response.status_code == 400 and "response_format" in response.text:
                    print("  Warning: API rejected response_format, retrying with type='text'...")
                    payload["response_format"] = {"type": "text"}
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=timeout
                    )
                
                # Handle Rate Limits (429) if flag is set
                if response.status_code == 429 and self.retry_on_rate_limit:
                    retry_after = response.headers.get("Retry-After")
                    
                    # If header exists, wait that long. If not, exponential backoff.
                    if retry_after:
                        wait_time = int(float(retry_after))
                        print(f"  Rate Limit Hit. API requested wait: {wait_time}s. Retrying...")
                    else:
                        wait_time = 2 ** retry_count # 1, 2, 4, 8, 16...
                        print(f"  Rate Limit Hit (No Retry-After header). Backing off for {wait_time}s...")
                    
                    time.sleep(wait_time)
                    retry_count += 1
                    continue
                
                if response.status_code != 200:
                    return {"error": f"API Error: {response.status_code} - {response.text}"}
                    
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Remove <think> blocks if present (common in some local models)
                content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
                
                # Clean markdown code blocks if present
                if content.startswith("```"):
                    content = content.strip("`")
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()
                
                # Parse JSON
                try:
                    parsed_result = json.loads(content)
                    
                    # Convert score to int
                    if 'score' in parsed_result:
                        parsed_result['score'] = int(parsed_result['score'])
                        
                    # Ensure structure
                    if 'reasons' not in parsed_result:
                        parsed_result['reasons'] = []
                        
                    return parsed_result
                    
                except json.JSONDecodeError:
                     return {"error": "Failed to parse JSON", "raw_content": content}
                     
            except requests.exceptions.Timeout:
                 if self.retry_on_rate_limit and retry_count < max_retries - 1:
                      wait_time = 2 ** retry_count
                      print(f"  Request Timed Out. Retrying in {wait_time}s...")
                      time.sleep(wait_time)
                      retry_count += 1
                      continue
                 else:
                      return {"error": f"Request timed out after {timeout}s"}
            except Exception as e:
                return {"error": str(e)}
                
        return {"error": "Max retries exceeded"}


