const taskSelect = document.getElementById('task-id');
const resetBtn = document.getElementById('reset-btn');
const stepBtn = document.getElementById('step-btn');

const intentsInput = document.getElementById('intents-input');
const priorityInput = document.getElementById('priority-input');
const departmentsInput = document.getElementById('departments-input');
const responseInput = document.getElementById('response-input');
const resolvedInput = document.getElementById('resolved-input');

const obsContent = document.getElementById('observation-content');
const rewardContent = document.getElementById('reward-content');
const scoreText = document.getElementById('score-text');
const feedbackText = document.getElementById('feedback-text');
const breakdownList = document.getElementById('breakdown-list');
const debugContent = document.getElementById('debug-content');

let currentTaskId = 'EASY-001';

async function updateState() {
    const resp = await fetch(`/state?task_id=${currentTaskId}`);
    const state = await resp.json();
    obsContent.innerHTML = `<pre>${JSON.stringify(state, null, 2)}</pre>`;
    
    const debugResp = await fetch(`/debug?task_id=${currentTaskId}`);
    const debug = await debugResp.json();
    debugContent.innerHTML = `<pre>${JSON.stringify(debug, null, 2)}</pre>`;
}

async function handleReset() {
    currentTaskId = taskSelect.value;
    const resp = await fetch(`/reset?task_id=${currentTaskId}`, { method: 'POST' });
    const obs = await resp.json();
    obsContent.innerHTML = `<pre>${JSON.stringify(obs, null, 2)}</pre>`;
    
    scoreText.innerText = '0.00';
    feedbackText.innerText = 'Environment reset.';
    breakdownList.innerHTML = '';
    
    await updateState();
}

async function handleStep() {
    currentTaskId = taskSelect.value;
    const action = {
        intents: intentsInput.value.split(',').map(s => s.trim()).filter(s => s),
        priority: priorityInput.value,
        departments: departmentsInput.value.split(',').map(s => s.trim()).filter(s => s),
        response_message: responseInput.value,
        mark_resolved: resolvedInput.checked
    };

    const resp = await fetch(`/step?task_id=${currentTaskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action)
    });
    
    const result = await resp.json();
    
    scoreText.innerText = result.reward.score.toFixed(2);
    feedbackText.innerText = result.reward.feedback;
    
    breakdownList.innerHTML = '';
    Object.entries(result.reward.breakdown).forEach(([key, val]) => {
        const item = document.createElement('div');
        item.className = 'breakdown-item';
        item.innerText = `${key}: ${val.toFixed(2)}`;
        breakdownList.appendChild(item);
    });

    await updateState();
    
    if (result.done) {
        feedbackText.innerText += " (Task Terminated)";
    }
}

resetBtn.addEventListener('click', handleReset);
stepBtn.addEventListener('click', handleStep);

// Initial load
window.onload = handleReset;
