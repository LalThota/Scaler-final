/**
 * CSOS++ Debug Console - Logic Layer
 * Precision engineering tool for AI agent observation and debugging.
 */

const elements = {
    taskId: document.getElementById('task-id-select'),
    resetBtn: document.getElementById('reset-btn'),
    refreshBtn: document.getElementById('refresh-btn'),
    submitBtn: document.getElementById('submit-btn'),
    form: document.getElementById('action-form'),
    obsView: document.getElementById('observation-json'),
    debugView: document.getElementById('debug-json'),
    debugErrors: document.getElementById('debug-errors'),
    scoreView: document.getElementById('score-val'),
    feedbackView: document.getElementById('feedback-val'),
    breakdownGrid: document.getElementById('breakdown-val'),
    historyList: document.getElementById('history-list'),
    stepCount: document.getElementById('step-count'),
    statusBadge: document.getElementById('current-status'),
    toast: document.getElementById('toast'),
    // Inputs
    inputIntents: document.getElementById('input-intents'),
    inputPriority: document.getElementById('input-priority'),
    inputDepts: document.getElementById('input-departments'),
    inputMsg: document.getElementById('input-message'),
    inputResolved: document.getElementById('input-resolved'),
    inputClarify: document.getElementById('input-clarification')
};

let actionHistory = [];

/** API Utilities **/
async function apiCall(endpoint, method = 'GET', body = null) {
    const taskId = elements.taskId.value;
    const url = `${endpoint}?task_id=${taskId}`;
    
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    if (body) options.body = JSON.stringify(body);

    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        return await response.json();
    } catch (err) {
        showToast(err.message);
        console.error(`[API ERROR] ${endpoint}:`, err);
        return null;
    }
}

/** UI Updates **/
function showToast(msg) {
    elements.toast.innerText = msg;
    elements.toast.classList.add('show');
    setTimeout(() => elements.toast.classList.remove('show'), 3000);
}

function updateObservation(data) {
    if (!data) return;
    elements.obsView.textContent = JSON.stringify(data, null, 2);
    elements.stepCount.textContent = data.step_count || 0;
    elements.statusBadge.textContent = `STATUS: ${(data.status || 'UNKNOWN').toUpperCase()}`;
    
    // Status color coding
    const statusMap = {
        'open': 'var(--accent)',
        'in_progress': 'var(--warning)',
        'resolved': 'var(--success)',
        'escalated': 'var(--danger)'
    };
    elements.statusBadge.style.background = statusMap[data.status] || 'var(--text-muted)';
}

function updateReward(reward) {
    if (!reward) return;
    elements.scoreView.textContent = (reward.score || 0).toFixed(2);
    elements.scoreView.style.color = reward.score > 0.7 ? 'var(--success)' : reward.score > 0.4 ? 'var(--warning)' : 'var(--danger)';
    elements.feedbackView.textContent = reward.feedback || "No feedback provided.";
    
    elements.breakdownGrid.innerHTML = '';
    if (reward.breakdown) {
        Object.entries(reward.breakdown).forEach(([key, val]) => {
            const item = document.createElement('div');
            item.className = 'breakdown-item';
            item.innerHTML = `<span>${key}</span> <b>${val.toFixed(2)}</b>`;
            elements.breakdownGrid.appendChild(item);
        });
    }
}

async function updateDebug() {
    const debug = await apiCall('/debug');
    if (!debug) return;
    elements.debugView.textContent = JSON.stringify(debug, null, 2);
    
    if (debug.validation_errors && debug.validation_errors.length > 0) {
        elements.debugErrors.innerHTML = `<strong>Validation Errors:</strong><br>${debug.validation_errors.join('<br>')}`;
        elements.debugErrors.classList.remove('hidden');
    } else {
        elements.debugErrors.classList.add('hidden');
    }
}

function addHistory(action) {
    const summary = action.mark_resolved ? "RESOLVE" : action.response_message ? "RESPOND" : "CLASSIFY";
    const timestamp = new Date().toLocaleTimeString([], { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    
    actionHistory.unshift(`[${timestamp}] ${summary} - Intents: ${action.intents.slice(0,2)}...`);
    if (actionHistory.length > 5) actionHistory.pop();
    
    elements.historyList.innerHTML = actionHistory.map(item => `<li>${item}</li>`).join('');
}

/** Controller Actions **/
async function handleReset() {
    elements.resetBtn.disabled = true;
    const obs = await apiCall('/reset', 'POST');
    if (obs) {
        updateObservation(obs);
        actionHistory = [];
        elements.historyList.innerHTML = '';
        elements.scoreView.textContent = '0.00';
        elements.feedbackView.textContent = 'Environment reset.';
        elements.breakdownGrid.innerHTML = '';
        await updateDebug();
    }
    elements.resetBtn.disabled = false;
}

async function handleRefresh() {
    const obs = await apiCall('/state');
    if (obs) updateObservation(obs);
    await updateDebug();
}

async function handleSubmit(e) {
    e.preventDefault();
    const action = {
        intents: elements.inputIntents.value.split(',').map(s => s.trim()).filter(s => s),
        priority: elements.inputPriority.value,
        departments: elements.inputDepts.value.split(',').map(s => s.trim()).filter(s => s),
        response_message: elements.inputMsg.value.trim(),
        mark_resolved: elements.inputResolved.checked,
        ask_clarification: elements.inputClarify.checked
    };

    // Validation (Basic)
    if (action.intents.length === 0 && !action.mark_resolved) {
        showToast("Enter at least one intent or mark as resolved.");
        return;
    }

    elements.submitBtn.disabled = true;
    const result = await apiCall('/step', 'POST', action);
    
    if (result) {
        updateObservation(result.observation);
        updateReward(result.reward);
        addHistory(action);
        await updateDebug();
        
        if (result.done) {
            showToast("Terminal state reached.");
        }
    }
    elements.submitBtn.disabled = false;
}

/** Initialization **/
elements.resetBtn.addEventListener('click', handleReset);
elements.refreshBtn.addEventListener('click', handleRefresh);
elements.form.addEventListener('submit', handleSubmit);
elements.taskId.addEventListener('change', handleRefresh);

// Initial State Sync
window.addEventListener('DOMContentLoaded', handleRefresh);
