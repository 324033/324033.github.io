"""
Simple Gemini chat backend (Flask blueprint).

The API key stays on the server: it is read from the local .env file (GEMINI_API_KEY=...)
or from the environment, and is only ever sent to Google as the `x-goog-api-key` header.
The browser talks to /api/chat on this local server and never sees the key.

Routes:
  GET  /chat              chat page
  GET  /api/chat/status   {configured, model, models}
  POST /api/chat          {messages:[{role:'user'|'model', text}], model?}
                          -> streamed NDJSON lines {"text": "..."} ... {"done": true} | {"error": "..."}
"""
import json
import os
import time
import urllib.error
import urllib.request

from flask import Blueprint, Response, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, '.env')
API_BASE = 'https://generativelanguage.googleapis.com/v1beta/models'
DEFAULT_MODEL = 'gemini-3.5-flash-lite'
MODELS = {
    'gemini-3.5-flash-lite': 'Gemini 3.5 Flash Lite (default)',
    'gemini-3.8-flash': 'Gemini 3.8 Flash',
}
MAX_HISTORY = 40          # messages sent back to the model as context
MAX_CHARS = 20000         # per message

bp = Blueprint('gemini_chat', __name__)


def read_env_file():
    """Minimal .env reader (KEY=VALUE per line, # comments). Re-read on every request,
    so a key added to .env works without restarting the server."""
    values = {}
    try:
        with open(ENV_FILE, encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return values


def get_config():
    env = read_env_file()
    key = os.environ.get('GEMINI_API_KEY') or env.get('GEMINI_API_KEY') or ''
    if key in ('', 'paste-your-key-here'):
        key = ''
    model = os.environ.get('GEMINI_MODEL') or env.get('GEMINI_MODEL') or DEFAULT_MODEL
    if model not in MODELS:
        model = DEFAULT_MODEL
    return key, model


def explain_http_error(status, body):
    """Turn a Gemini API error into a short, clear message (never contains the key)."""
    detail = ''
    try:
        err = json.loads(body).get('error', {})
        detail = err.get('message', '')
        reasons = json.dumps(err.get('details', []))
    except (ValueError, AttributeError):
        reasons = ''
    if 'API_KEY_INVALID' in reasons or 'API key not valid' in detail:
        return 'The API key is not valid. Check GEMINI_API_KEY in the .env file (or create a new key in Google AI Studio).'
    if status == 403:
        return f'Access denied by Google (403). {detail}'.strip()
    if status == 404:
        return f'Model not found (404). {detail}'.strip()
    if status == 429:
        return 'Rate limit or free quota reached (429). Wait a minute and try again.'
    if status >= 500:
        return f'Google\'s server is busy right now ({status}, tried 3 times). Please try again in a moment.'
    return f'Request failed ({status}). {detail}'.strip()


@bp.route('/chat')
def chat_page():
    return render_template('chat.html')


@bp.route('/api/chat/status')
def chat_status():
    key, model = get_config()
    return jsonify({'configured': bool(key), 'model': model, 'models': MODELS})


@bp.route('/api/chat', methods=['POST'])
def chat():
    key, default_model = get_config()
    if not key:
        return jsonify({'error': 'No API key configured. Add GEMINI_API_KEY=... to the .env file in the project folder.'}), 400

    data = request.get_json(silent=True) or {}
    model = data.get('model') or default_model
    if model not in MODELS:
        return jsonify({'error': f'Unknown model "{model}".'}), 400

    contents = []
    for m in (data.get('messages') or [])[-MAX_HISTORY:]:
        role = 'model' if m.get('role') == 'model' else 'user'
        text = str(m.get('text', ''))[:MAX_CHARS]
        if text.strip():
            contents.append({'role': role, 'parts': [{'text': text}]})
    if not contents or contents[-1]['role'] != 'user':
        return jsonify({'error': 'Please type a message first.'}), 400

    payload = json.dumps({'contents': contents}).encode('utf-8')
    url = f'{API_BASE}/{model}:streamGenerateContent?alt=sse'

    def open_stream():
        """Google sometimes answers 500/503 ("overloaded") - retry twice before giving up."""
        for attempt, wait in enumerate((0, 1, 3)):
            if wait:
                time.sleep(wait)
            req = urllib.request.Request(url, data=payload, method='POST', headers={
                'Content-Type': 'application/json',
                'x-goog-api-key': key,
            })
            try:
                return urllib.request.urlopen(req, timeout=60)
            except urllib.error.HTTPError as e:
                if e.code not in (500, 503) or attempt == 2:
                    raise

    def generate():
        try:
            with open_stream() as resp:
                usage = None
                for raw in resp:
                    line = raw.decode('utf-8', errors='replace').strip()
                    if not line.startswith('data:'):
                        continue
                    try:
                        chunk = json.loads(line[5:].strip())
                    except ValueError:
                        continue
                    usage = chunk.get('usageMetadata', usage)
                    for cand in chunk.get('candidates', []):
                        for part in (cand.get('content') or {}).get('parts', []):
                            if part.get('text') and not part.get('thought'):
                                yield json.dumps({'text': part['text']}) + '\n'
                        if cand.get('finishReason') in ('SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT'):
                            yield json.dumps({'error': 'The answer was blocked by Gemini\'s safety filter.'}) + '\n'
                    if chunk.get('promptFeedback', {}).get('blockReason'):
                        yield json.dumps({'error': 'Your message was blocked by Gemini\'s safety filter.'}) + '\n'
                yield json.dumps({'done': True, 'model': model, 'usage': usage}) + '\n'
        except urllib.error.HTTPError as e:
            yield json.dumps({'error': explain_http_error(e.code, e.read().decode('utf-8', errors='replace'))}) + '\n'
        except urllib.error.URLError as e:
            yield json.dumps({'error': f'Could not reach Google ({e.reason}). Check your internet connection.'}) + '\n'
        except TimeoutError:
            yield json.dumps({'error': 'Google did not answer in time. Please try again.'}) + '\n'

    return Response(generate(), mimetype='application/x-ndjson', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
