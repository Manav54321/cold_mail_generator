# main.py (FastAPI backend)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from langchain_community.document_loaders import WebBaseLoader
from app.chains import Chain
from app.portfolio import Portfolio
from app.utils import clean_text

app = FastAPI()
chain = Chain()
portfolio = Portfolio()

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cold Mail Generator</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Special+Elite&display=swap');

  /* Root colors */
  :root {
    --primary-color: #000000; /* black */
    --background-white: #fff;
    --text-dark: #222;
    --border-color: #000000; /* black */
    --border-radius: 8px;
    --box-shadow: rgba(0, 0, 0, 0.1);
  }

  /* Reset and base */
  * {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    font-family: 'Special Elite', monospace;
    background-color: var(--background-white);
    color: var(--text-dark);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
    position: relative;
  }

  /* Grain texture overlay */
  body::before {
    content: "";
    pointer-events: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
      url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"><circle fill="%23000" fill-opacity="0.03" cx="1" cy="1" r="1"/></svg>');
    opacity: 0.12;
    z-index: 0;
  }

  /* Container box with black border and subtle shadow */
  header, h1, input, button, #results, #error {
    position: relative;
    z-index: 1;
  }

  /* Header container right aligned */
  header {
    width: 100%;
    max-width: 720px;
    display: flex;
    justify-content: flex-end;
    margin-bottom: 32px;
  }

  /* Removed dark mode toggle, so empty header */
  header > * {
    display: none;
  }

  /* Main title */
  h1 {
    font-weight: 600;
    font-size: 2.5rem;
    margin-bottom: 24px;
    text-align: center;
    max-width: 720px;
  }

  /* Input box styling */
  input[type="text"] {
    width: 100%;
    max-width: 720px;
    padding: 16px 20px;
    font-size: 18px;
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    margin-bottom: 20px;
    outline-offset: 2px;
    background: var(--background-white);
    color: var(--text-dark);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    box-shadow: inset 0 0 3px rgba(0,0,0,0.05);
    font-family: 'Special Elite', monospace;
  }
  input[type="text"]:focus {
    border-color: var(--primary-color);
    box-shadow: 0 0 8px var(--primary-color);
    outline: none;
  }

  /* Submit button */
  button.submit-btn {
    background: var(--primary-color);
    color: white;
    border: none;
    padding: 16px 24px;
    border-radius: var(--border-radius);
    font-size: 18px;
    font-weight: 600;
    cursor: pointer;
    max-width: 720px;
    width: 100%;
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 4px 10px var(--box-shadow);
    font-family: 'Special Elite', monospace;
  }
  button.submit-btn:hover {
    background: #222; /* darker black */
    box-shadow: 0 6px 14px rgba(0,0,0,0.7);
  }

  /* Results container */
  #results {
    margin-top: 32px;
    max-width: 720px;
    width: 100%;
  }

  /* Each email box with black border and slight shadow */
  .email {
    position: relative;
    border: 2px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 20px 60px 20px 20px;
    margin-bottom: 20px;
    background: var(--background-white);
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
    font-family: 'Special Elite', monospace;
  }
  .email:hover {
    border-color: var(--primary-color);
    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
  }

  /* Pre formatting for emails */
  .email pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    overflow-wrap: break-word;
    margin: 0;
    font-family: 'Special Elite', monospace;
    font-size: 15px;
    color: var(--text-dark);
  }

  /* Copy button styling */
  .copy-btn {
    position: absolute;
    top: 20px;
    right: 20px;
    background: var(--primary-color);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    font-family: 'Special Elite', monospace;
  }
  .copy-btn:hover {
    background: #222;
    box-shadow: 0 6px 12px rgba(0,0,0,0.6);
  }

  /* Error message styling */
  .error {
    color: #cc0000;
    margin-top: 28px;
    font-weight: 600;
    max-width: 720px;
    width: 100%;
    font-family: 'Special Elite', monospace;
  }

</style>
</head>
<body>

<header>
  <!-- removed dark mode toggle -->
</header>

<h1>Cold  Mail Generator</h1>

<input id="urlInput" type="text" placeholder="Enter job URL here" value="https://careers.nike.com/business-systems-analyst/job/R-60176" />
<button class="submit-btn" id="submitBtn">Generate Email</button>

<div id="results"></div>
<div id="error" class="error"></div>

<script>
  const submitBtn = document.getElementById('submitBtn');
  const urlInput = document.getElementById('urlInput');
  const resultsDiv = document.getElementById('results');
  const errorDiv = document.getElementById('error');

  function copyEmail(index) {
    const emailText = document.getElementById(`email-${index}`).querySelector('pre').innerText;
    navigator.clipboard.writeText(emailText).then(() => {
      alert('Email copied to clipboard!');
    }).catch(() => {
      alert('Failed to copy email.');
    });
  }

  submitBtn.onclick = async () => {
    errorDiv.textContent = '';
    resultsDiv.innerHTML = 'Loading...';
    try {
      const response = await fetch('/generate_email', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({url: urlInput.value})
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || 'Something went wrong');
      }
      const data = await response.json();
      if (!data.emails || data.emails.length === 0) {
        resultsDiv.innerHTML = '<p>No emails found.</p>';
        return;
      }

      let html = '';
      data.emails.forEach((item, index) => {
        html += `
          <div class="email" id="email-${index}">
            <pre>${item.email}</pre>
            <button onclick="copyEmail(${index})" class="copy-btn">Copy</button>
          </div>
        `;
      });
      resultsDiv.innerHTML = html;

    } catch (e) {
      resultsDiv.innerHTML = '';
      errorDiv.textContent = e.message;
    }
  };
</script>

</body>
</html>
"""

class InputData(BaseModel):
    url: str

@app.post("/generate_email")
def generate_email(data: InputData):
    try:
        loader = WebBaseLoader([data.url])
        page_data = loader.load().pop().page_content
        cleaned = clean_text(page_data)
        portfolio.load_portfolio()
        jobs = chain.extract_jobs(cleaned)
        result = []
        for job in jobs:
            skills = job.get('skills', [])
            links = portfolio.query_links(skills)
            email = chain.write_mail(job, links)
            result.append({"email": email})
        return {"emails": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
