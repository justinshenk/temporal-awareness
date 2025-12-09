#!/usr/bin/env python3
"""
Interactive Temporal Steering Web App

Slider-based interface to steer GPT-2 generation toward short-term or long-term
outputs using Boltzmann-style logit weighting.

Energy function: E(token) = -probe_score(token)
- Low energy (short-term tokens) → favored when T < 0
- High energy (long-term tokens) → favored when T > 0

The slider controls the "temporal temperature" which biases the distribution.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
temporal_scores = None  # (vocab_size,) tensor of temporal scores
temporal_probe = None  # Trained probe for introspection
PROBE_LAYER = 8  # Layer to extract activations from

# Session state for streaming generation
generation_sessions = {}  # session_id -> {'ids': tensor, 'prompt': str}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Temporal Steering</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }
        h1 {
            text-align: center;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .subtitle a { color: #00d2ff; }
        .control-panel {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .slider-container {
            margin: 20px 0;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 14px;
            color: #aaa;
        }
        .slider-value {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            color: #00d2ff;
        }
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #3498db 0%, #9b59b6 50%, #e74c3c 100%);
            outline: none;
            -webkit-appearance: none;
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        textarea {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: none;
            border-radius: 12px;
            background: rgba(255,255,255,0.1);
            color: white;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
        }
        textarea:focus {
            outline: 2px solid #00d2ff;
        }
        textarea::placeholder { color: #666; }
        button {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 25px;
            cursor: pointer;
            margin-top: 15px;
            width: 100%;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0,210,255,0.3);
        }
        button:disabled {
            background: #444;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .output-section {
            background: rgba(255,255,255,0.05);
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-top: 20px;
        }
        .output-title {
            font-weight: 600;
            margin-bottom: 15px;
            color: #aaa;
        }
        .generated-text {
            font-size: 18px;
            line-height: 1.8;
        }
        .prompt-part { color: #888; }
        .generated-part { color: white; }
        .token {
            padding: 2px 4px;
            border-radius: 4px;
            margin: 1px;
            display: inline;
        }
        .stats {
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 13px;
            color: #888;
        }
        .stats span { margin-right: 20px; }
        .probe-panel {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 15px;
            margin-top: 15px;
        }
        .probe-title {
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .probe-title .icon { font-size: 16px; }
        .probe-bar-container {
            height: 24px;
            background: linear-gradient(90deg, #3498db 0%, #9b59b6 50%, #e74c3c 100%);
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }
        .probe-indicator {
            position: absolute;
            top: 2px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: left 0.3s ease;
        }
        .probe-labels {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }
        .probe-value {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            margin-top: 8px;
        }
        .probe-value.short-term { color: #3498db; }
        .probe-value.long-term { color: #e74c3c; }
        .probe-value.neutral { color: #9b59b6; }
        .horizon-indicator {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            font-size: 12px;
        }
        .horizon-indicator span {
            padding: 4px 12px;
            border-radius: 15px;
        }
        .short-label { background: rgba(52, 152, 219, 0.3); color: #3498db; }
        .long-label { background: rgba(231, 76, 60, 0.3); color: #e74c3c; }
        .loading {
            text-align: center;
            padding: 20px;
        }
        .spinner {
            border: 3px solid rgba(255,255,255,0.1);
            border-top: 3px solid #00d2ff;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 10px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 15px;
        }
        .example-btn {
            background: rgba(255,255,255,0.1);
            border: none;
            padding: 8px 12px;
            border-radius: 8px;
            color: #aaa;
            cursor: pointer;
            margin: 3px;
            font-size: 13px;
            transition: background 0.2s;
        }
        .example-btn:hover {
            background: rgba(255,255,255,0.2);
            color: white;
        }
        .formula {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 14px;
            margin: 15px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Temporal Steering</h1>
    <p class="subtitle">
        Steer GPT-2 generation toward short-term or long-term outputs |
        <a href="https://github.com/justinshenk/temporal-awareness/issues/20">Issue #20</a>
    </p>

    <div class="control-panel">
        <div class="slider-container">
            <div class="slider-label">
                <span class="short-label">Short-term</span>
                <span>Temporal Bias</span>
                <span class="long-label">Long-term</span>
            </div>
            <input type="range" id="temporal-slider" min="-5" max="5" step="0.5" value="0">
            <div class="slider-value" id="slider-value">0.0 (Neutral)</div>
            <div class="horizon-indicator">
                <span class="short-label">now, today, urgent</span>
                <span class="long-label">future, decades, legacy</span>
            </div>
        </div>

        <div class="formula">
            P(token) ∝ exp(original_logit + β × temporal_score(token))
        </div>

        <textarea id="prompt" placeholder="Enter a prompt...">When planning for the future, I should focus on</textarea>

        <div class="examples">
            <button class="example-btn" onclick="setPrompt('When planning for the future, I should focus on')">Future planning</button>
            <button class="example-btn" onclick="setPrompt('The most important thing to do right now is')">Right now</button>
            <button class="example-btn" onclick="setPrompt('My investment strategy is to')">Investment</button>
            <button class="example-btn" onclick="setPrompt('For dinner tonight, I will')">Tonight</button>
            <button class="example-btn" onclick="setPrompt('Over the next decade, technology will')">Decade</button>
        </div>

        <button onclick="startStreaming()" id="generate-btn">Generate (Streaming)</button>
        <button onclick="stopStreaming()" id="stop-btn" style="display:none; background: linear-gradient(90deg, #e74c3c, #c0392b);">Stop</button>
    </div>

    <div id="output-container"></div>

    <script>
        const slider = document.getElementById('temporal-slider');
        const sliderValue = document.getElementById('slider-value');

        let currentSessionId = null;
        let isStreaming = false;
        let tokens = [];
        let scores = [];
        let probeReadings = [];
        const DELAY_MS = 500;  // Delay between tokens
        const MAX_TOKENS = 50;

        slider.addEventListener('input', () => {
            const val = parseFloat(slider.value);
            let label = 'Neutral';
            if (val < -2) label = 'Strong short-term';
            else if (val < 0) label = 'Short-term';
            else if (val > 2) label = 'Strong long-term';
            else if (val > 0) label = 'Long-term';
            sliderValue.textContent = `${val.toFixed(1)} (${label})`;
        });

        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }

        async function startStreaming() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt || isStreaming) return;

            const btn = document.getElementById('generate-btn');
            const stopBtn = document.getElementById('stop-btn');
            const container = document.getElementById('output-container');

            // Reset state
            tokens = [];
            scores = [];
            probeReadings = [];
            isStreaming = true;
            btn.disabled = true;
            btn.textContent = 'Generating...';
            stopBtn.style.display = 'inline-block';

            // Initialize output
            container.innerHTML = `
                <div class="output-section">
                    <div class="output-title">Streaming... (adjust slider to change bias in real-time)</div>
                    <div class="generated-text">
                        <span class="prompt-part">${prompt}</span><span class="generated-part" id="token-output"></span>
                    </div>
                    <div class="probe-panel">
                        <div class="probe-title">
                            <span class="icon">🔬</span>
                            <span>Real-time Introspection (Layer 8 Probe)</span>
                        </div>
                        <div class="probe-bar-container">
                            <div class="probe-indicator" id="probe-indicator" style="left: calc(50% - 10px);"></div>
                        </div>
                        <div class="probe-labels">
                            <span>Immediate</span>
                            <span>Long-term</span>
                        </div>
                        <div class="probe-value neutral" id="probe-value">50% long-term</div>
                    </div>
                    <div class="stats" id="live-stats">
                        <span><strong>Tokens:</strong> 0</span>
                        <span><strong>Current β:</strong> ${slider.value}</span>
                    </div>
                </div>
            `;

            try {
                // Start session
                const startRes = await fetch('/start_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt })
                });
                const startData = await startRes.json();
                if (startData.error) throw new Error(startData.error);

                currentSessionId = startData.session_id;

                // Generate tokens one by one
                await generateNextToken();
            } catch (err) {
                container.innerHTML = `<div class="output-section"><p style="color: #e74c3c;">Error: ${err.message}</p></div>`;
                finishStreaming();
            }
        }

        async function generateNextToken() {
            if (!isStreaming || !currentSessionId || tokens.length >= MAX_TOKENS) {
                finishStreaming();
                return;
            }

            const beta = parseFloat(slider.value);  // Read current slider value

            try {
                const response = await fetch('/generate_token', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessionId, beta })
                });
                const data = await response.json();

                if (data.done || data.error) {
                    finishStreaming();
                    return;
                }

                // Add token to display
                tokens.push(data.token);
                scores.push(data.score);
                probeReadings.push(data.probe_reading);
                updateDisplay(data.token, data.score, data.probe_reading, beta);

                // Schedule next token
                setTimeout(generateNextToken, DELAY_MS);
            } catch (err) {
                console.error('Token generation error:', err);
                finishStreaming();
            }
        }

        function updateDisplay(token, score, probeReading, beta) {
            const output = document.getElementById('token-output');
            const stats = document.getElementById('live-stats');
            const probeIndicator = document.getElementById('probe-indicator');
            const probeValue = document.getElementById('probe-value');

            const color = getColor(score);
            const safeToken = token.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            output.innerHTML += `<span class="token" style="background-color: ${color};" title="Score: ${score.toFixed(3)}, Probe: ${probeReading.toFixed(3)}, β: ${beta}">${safeToken}</span>`;

            // Update probe visualization
            const meanProbe = probeReadings.reduce((a, b) => a + b, 0) / probeReadings.length;
            const probePercent = Math.max(0, Math.min(100, meanProbe * 100));
            probeIndicator.style.left = `calc(${probePercent}% - 10px)`;

            // Update probe value text
            let probeClass = 'neutral';
            let probeText = `${Math.round(probePercent)}% long-term`;
            if (probePercent < 35) {
                probeClass = 'short-term';
                probeText = `${Math.round(100 - probePercent)}% immediate`;
            } else if (probePercent > 65) {
                probeClass = 'long-term';
            }
            probeValue.className = `probe-value ${probeClass}`;
            probeValue.textContent = probeText;

            const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
            stats.innerHTML = `
                <span><strong>Mean score:</strong> ${meanScore.toFixed(3)}</span>
                <span><strong>Probe:</strong> ${meanProbe.toFixed(3)}</span>
                <span><strong>Tokens:</strong> ${tokens.length}</span>
                <span><strong>Current β:</strong> ${beta.toFixed(1)}</span>
            `;
        }

        function stopStreaming() {
            if (currentSessionId) {
                fetch('/stop_stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ session_id: currentSessionId })
                });
            }
            finishStreaming();
        }

        function finishStreaming() {
            isStreaming = false;
            currentSessionId = null;

            const btn = document.getElementById('generate-btn');
            const stopBtn = document.getElementById('stop-btn');
            btn.disabled = false;
            btn.textContent = 'Generate (Streaming)';
            stopBtn.style.display = 'none';

            // Update title to show complete
            const title = document.querySelector('.output-title');
            if (title && tokens.length > 0) {
                const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
                title.textContent = `Generated ${tokens.length} tokens (mean score: ${meanScore.toFixed(3)})`;
            }
        }

        function getColor(score) {
            let r, g, b;
            if (score < 0.5) {
                r = Math.round(52 + (155 - 52) * (score * 2));
                g = Math.round(152 + (89 - 152) * (score * 2));
                b = Math.round(219 + (182 - 219) * (score * 2));
            } else {
                r = Math.round(155 + (231 - 155) * ((score - 0.5) * 2));
                g = Math.round(89 + (76 - 89) * ((score - 0.5) * 2));
                b = Math.round(182 + (60 - 182) * ((score - 0.5) * 2));
            }
            return `rgba(${r}, ${g}, ${b}, 0.5)`;
        }

        document.getElementById('prompt').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                startStreaming();
            }
        });
    </script>
</body>
</html>
"""


def load_models():
    """Load GPT-2, probe, and compute vocabulary temporal scores."""
    global model, tokenizer, temporal_scores, temporal_probe

    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model.eval()
    print("✓ Model loaded")

    # Load the temporal probe for introspection
    probe_path = Path('results/checkpoints/temporal_caa_layer_8_probe.pkl')
    if probe_path.exists():
        print(f"Loading temporal probe from {probe_path}...")
        with open(probe_path, 'rb') as f:
            probe_data = pickle.load(f)
        # Handle both dict format and direct LogisticRegression object
        if isinstance(probe_data, dict):
            temporal_probe = probe_data['probe']
        else:
            temporal_probe = probe_data
        print("✓ Temporal probe loaded")
    else:
        print("⚠ Temporal probe not found, introspection disabled")

    # Check for cached scores
    cache_path = Path('results/temporal_steering/vocab_temporal_scores.pt')
    if cache_path.exists():
        print(f"Loading cached temporal scores from {cache_path}...")
        temporal_scores = torch.load(cache_path)
    else:
        # Compute scores using embedding approximation
        print("Computing vocabulary temporal scores...")
        probe_path = 'results/probes/token_temporal_layer_8_probe.pkl'

        with open(probe_path, 'rb') as f:
            data = pickle.load(f)
        probe = data['probe']

        direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
        embeddings = model.transformer.wte.weight.detach()

        scores = F.cosine_similarity(
            embeddings,
            direction.unsqueeze(0).expand(embeddings.shape[0], -1),
            dim=1
        )
        temporal_scores = (scores - scores.min()) / (scores.max() - scores.min())

        # Cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(temporal_scores, cache_path)

    print(f"✓ Temporal scores loaded ({len(temporal_scores)} tokens)")


def generate_with_steering(prompt, beta, max_new_tokens=40):
    """Generate text with temporal logit steering."""

    inputs = tokenizer(prompt, return_tensors='pt')
    generated_ids = inputs['input_ids'].clone()

    generated_tokens = []
    generated_scores = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated_ids)
            logits = outputs.logits[:, -1, :]

            # Apply temporal steering: logits += beta * (score - 0.5) * 2
            # This centers the scores around 0 and scales by beta
            bias = (temporal_scores - 0.5) * 2 * beta
            modified_logits = logits + bias.unsqueeze(0)

            # Sample with temperature
            probs = F.softmax(modified_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            generated_tokens.append(tokenizer.decode([next_token.item()]))
            generated_scores.append(temporal_scores[next_token.item()].item())

    return generated_tokens, generated_scores


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        beta = float(data.get('beta', 0.0))

        if not prompt:
            return jsonify({'error': 'No prompt provided'})

        tokens, scores = generate_with_steering(prompt, beta)

        return jsonify({
            'tokens': tokens,
            'scores': scores,
            'mean_score': float(np.mean(scores)) if scores else 0.5
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Initialize a new streaming generation session."""
    import uuid
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'No prompt provided'})

        session_id = str(uuid.uuid4())
        inputs = tokenizer(prompt, return_tensors='pt')
        generation_sessions[session_id] = {
            'ids': inputs['input_ids'].clone(),
            'prompt': prompt
        }
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/generate_token', methods=['POST'])
def generate_token():
    """Generate a single token with the current beta value and probe introspection."""
    try:
        data = request.json
        session_id = data.get('session_id')
        beta = float(data.get('beta', 0.0))

        if session_id not in generation_sessions:
            return jsonify({'error': 'Invalid session', 'done': True})

        session = generation_sessions[session_id]
        generated_ids = session['ids']

        with torch.no_grad():
            # Run forward pass with hidden states for probe
            outputs = model(generated_ids, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]

            # Apply temporal steering
            bias = (temporal_scores - 0.5) * 2 * beta
            modified_logits = logits + bias.unsqueeze(0)

            # Sample with temperature
            probs = F.softmax(modified_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                del generation_sessions[session_id]
                return jsonify({'done': True})

            # Extract probe reading from layer 8 hidden state
            probe_reading = 0.5  # Default if probe not available
            if temporal_probe is not None and outputs.hidden_states is not None:
                # Get hidden state at probe layer (layer 8 = index 9 with embedding layer)
                hidden_state = outputs.hidden_states[PROBE_LAYER + 1]  # +1 for embedding layer
                # Use the last token's hidden state
                last_hidden = hidden_state[0, -1, :].cpu().numpy().reshape(1, -1)
                # Get probability from probe
                probe_proba = temporal_probe.predict_proba(last_hidden)[0]
                # Assuming class 1 is "long-term"
                probe_reading = float(probe_proba[1]) if len(probe_proba) > 1 else float(probe_proba[0])

            # Update session state
            session['ids'] = torch.cat([generated_ids, next_token], dim=1)
            token_text = tokenizer.decode([next_token.item()])
            score = temporal_scores[next_token.item()].item()

            return jsonify({
                'token': token_text,
                'score': score,
                'probe_reading': probe_reading,
                'done': False
            })
    except Exception as e:
        return jsonify({'error': str(e), 'done': True})


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop and clean up a streaming session."""
    try:
        data = request.json
        session_id = data.get('session_id')
        if session_id in generation_sessions:
            del generation_sessions[session_id]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    load_models()
    print()
    print("=" * 60)
    print("TEMPORAL STEERING WEB APP")
    print("=" * 60)
    print("\nOpen http://127.0.0.1:8081 in your browser")
    print("Use the slider to bias generation toward short-term or long-term")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host='127.0.0.1', port=8081)
