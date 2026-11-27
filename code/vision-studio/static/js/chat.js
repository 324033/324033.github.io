// Gemini Chat (front end). Talks only to the local server (/api/chat), which holds the API key.
(() => {
    'use strict';
    const $ = (id) => document.getElementById(id);
    const el = {
        model: $('cModel'), newChat: $('cNew'), setup: $('cSetup'), recheck: $('cRecheck'),
        messages: $('cMessages'), welcome: $('cWelcome'), form: $('cForm'), input: $('cInput'),
        send: $('cSend'), stop: $('cStop'), status: $('cStatus'),
    };

    const history = [];          // [{role: 'user'|'model', text}]
    let configured = false;
    let controller = null;       // AbortController of the running request

    // ---------- tiny, safe markdown (escape first, then format) ----------
    const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    function renderMarkdown(src) {
        const blocks = [];
        let text = esc(src).replace(/```[a-zA-Z0-9]*\n?([\s\S]*?)```/g, (_, code) => { blocks.push(code); return `\u0000${blocks.length - 1}\u0000`; });
        text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
            .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
        const html = text.split(/\n{2,}/).map((para) => {
            const lines = para.split('\n');
            if (lines.every((l) => /^\s*[-*] /.test(l))) return `<ul>${lines.map((l) => `<li>${l.replace(/^\s*[-*] /, '')}</li>`).join('')}</ul>`;
            if (lines.every((l) => /^\s*\d+[.)] /.test(l))) return `<ol>${lines.map((l) => `<li>${l.replace(/^\s*\d+[.)] /, '')}</li>`).join('')}</ol>`;
            if (/^\u0000\d+\u0000$/.test(para.trim())) return para.trim();
            return `<p>${lines.join('<br>')}</p>`;
        }).join('');
        return html.replace(/\u0000(\d+)\u0000/g, (_, i) => `<pre><code>${blocks[Number(i)]}</code></pre>`);
    }

    // ---------- UI helpers ----------
    function addBubble(role, text = '') {
        el.welcome.hidden = true;
        const div = document.createElement('div');
        div.className = `msg ${role}`;
        if (role === 'user') div.textContent = text; else div.innerHTML = text ? renderMarkdown(text) : '';
        el.messages.append(div);
        el.messages.scrollTop = el.messages.scrollHeight;
        return div;
    }

    function setBusy(busy) {
        el.send.disabled = busy || !configured;
        el.stop.hidden = !busy;
        el.send.hidden = busy;
        el.model.disabled = busy;
    }

    function autosize() {
        el.input.style.height = 'auto';
        el.input.style.height = `${Math.min(200, el.input.scrollHeight)}px`;
    }

    // ---------- status ----------
    async function checkStatus() {
        try {
            const s = await (await fetch('/api/chat/status')).json();
            configured = s.configured;
            if (!el.model.options.length) {
                el.model.innerHTML = Object.entries(s.models).map(([id, label]) => `<option value="${esc(id)}">${esc(label)}</option>`).join('');
                el.model.value = s.model;
            }
            el.setup.hidden = configured;
            el.status.textContent = configured ? `API key configured · model: ${el.model.value}` : 'No API key configured yet.';
        } catch (err) {
            el.status.textContent = 'The local server is not reachable. Start it with run.bat.';
        }
        setBusy(!!controller);
    }

    // ---------- sending ----------
    async function send(text) {
        history.push({ role: 'user', text });
        addBubble('user', text);
        const bubble = addBubble('model');
        bubble.classList.add('typing');
        let answer = '';
        controller = new AbortController();
        setBusy(true);
        const t0 = performance.now();
        let errorText = null, usage = null;
        try {
            const res = await fetch('/api/chat', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: history, model: el.model.value }),
                signal: controller.signal,
            });
            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                throw new Error(body.error || `Server error ${res.status}`);
            }
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buf = '';
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                let nl;
                while ((nl = buf.indexOf('\n')) >= 0) {
                    const line = buf.slice(0, nl).trim();
                    buf = buf.slice(nl + 1);
                    if (!line) continue;
                    const msg = JSON.parse(line);
                    if (msg.text) { answer += msg.text; bubble.innerHTML = renderMarkdown(answer); el.messages.scrollTop = el.messages.scrollHeight; }
                    if (msg.error) errorText = msg.error;
                    if (msg.done) usage = msg.usage;
                }
            }
        } catch (err) {
            errorText = err.name === 'AbortError' ? null : (err.message || 'Request failed');
            if (err.name === 'AbortError') answer += answer ? '\n\n*(stopped)*' : '*(stopped)*';
        }
        bubble.classList.remove('typing');
        if (answer) {
            bubble.innerHTML = renderMarkdown(answer);
            history.push({ role: 'model', text: answer });
            const secs = ((performance.now() - t0) / 1000).toFixed(1);
            const meta = document.createElement('span');
            meta.className = 'meta';
            meta.textContent = `${el.model.value} · ${secs} s${usage?.totalTokenCount ? ` · ${usage.totalTokenCount} tokens` : ''}`;
            bubble.append(meta);
        } else {
            bubble.remove();
            history.pop();             // drop the unanswered question so it can be retried
        }
        if (errorText) addBubble('error').textContent = `⚠️ ${errorText}`;
        controller = null;
        setBusy(false);
        el.input.focus();
    }

    // ---------- wiring ----------
    el.form.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = el.input.value.trim();
        if (!text || controller || !configured) return;
        el.input.value = '';
        autosize();
        send(text);
    });
    el.input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); el.form.requestSubmit(); }
    });
    el.input.addEventListener('input', autosize);
    el.stop.addEventListener('click', () => controller?.abort());
    el.newChat.addEventListener('click', () => {
        controller?.abort();
        history.length = 0;
        el.messages.querySelectorAll('.msg').forEach((m) => m.remove());
        el.welcome.hidden = false;
        el.input.focus();
    });
    el.model.addEventListener('change', () => { el.status.textContent = `API key configured · model: ${el.model.value} (answers can take a few seconds)`; });
    el.recheck.addEventListener('click', checkStatus);

    checkStatus().then(() => el.input.focus());
})();
