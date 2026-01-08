// Advanced Cyber Dashboard JavaScript - FIXED

const CONFIG = {
    API_ENDPOINT: '/api/status',
    POLL_INTERVAL: 1000,
    TIMEOUT: 5000,
};

let appState = {
    isOnline: false,
    previousCount: 0,
    startTime: Date.now(),
    chartData: [],
    maxChartPoints: 50,
};

// DOM Elements
const DOM = {
    peopleCount: document.getElementById('peopleCount'),
    changeIndicator: document.getElementById('changeIndicator'),
    riskLevel: document.getElementById('riskLevel'),
    riskHexagon: document.getElementById('riskHexagon'),
    riskGlow: document.getElementById('riskGlow'),
    riskDescription: document.getElementById('riskDescription'),
    densityValue: document.getElementById('densityValue'),
    surgeValue: document.getElementById('surgeValue'),
    systemStatus: document.getElementById('systemStatus'),
    systemDot: document.getElementById('systemDot'),
    currentTime: document.getElementById('currentTime'),
    lastUpdate: document.getElementById('lastUpdate'),
    uptime: document.getElementById('uptime'),
    videoProgress: document.getElementById('videoProgress'),
    sysStatus: document.getElementById('sysStatus'),
    predCurrent: document.getElementById('predCurrent'),
    pred5s: document.getElementById('pred5s'),
    pred10s: document.getElementById('pred10s'),
    trendValue: document.getElementById('trendValue'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidencePercent: document.getElementById('confidencePercent'),
    densityCanvas: document.getElementById('densityCanvas'),
    timelineChart: document.getElementById('timelineChart'),
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Cyber Dashboard Initialized');
    initParticles();
    initCharts();
    startPolling();
    setInterval(updateTime, 1000);
    setInterval(updateUptime, 1000);
    updateTime();
});

// Particle Background
function initParticles() {
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    const particles = [];
    for (let i = 0; i < 50; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            size: Math.random() * 2 + 1,
        });
    }
    
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0, 255, 157, 0.5)';
        
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            
            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;
            
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

// Initialize Charts
function initCharts() {
    const densityCtx = DOM.densityCanvas.getContext('2d');
    drawDensityMeter(densityCtx, 0);
    
    const timelineCtx = DOM.timelineChart.getContext('2d');
    drawTimelineChart(timelineCtx, []);
}

function drawDensityMeter(ctx, percentage) {
    const centerX = 150;
    const centerY = 150;
    const radius = 120;
    const lineWidth = 20;
    
    ctx.clearRect(0, 0, 300, 300);
    
    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = 'rgba(0, 255, 157, 0.1)';
    ctx.stroke();
    
    // Progress arc
    const angle = (percentage / 100) * Math.PI * 2 - Math.PI / 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, -Math.PI / 2, angle);
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = percentage > 70 ? '#ff0055' : percentage > 40 ? '#ffaa00' : '#00ff9d';
    ctx.stroke();
    
    // Glow effect
    ctx.shadowBlur = 20;
    ctx.shadowColor = ctx.strokeStyle;
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function drawTimelineChart(ctx, data) {
    const width = ctx.canvas.width;
    const height = ctx.canvas.height;
    const padding = 20;
    
    ctx.clearRect(0, 0, width, height);
    
    if (data.length < 2) return;
    
    const max = Math.max(...data, 10);
    const step = (width - padding * 2) / (data.length - 1);
    
    // Draw grid
    ctx.strokeStyle = 'rgba(0, 255, 157, 0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = padding + (height - padding * 2) * (i / 4);
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }
    
    // Draw line
    ctx.beginPath();
    ctx.strokeStyle = '#00ff9d';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#00ff9d';
    
    data.forEach((value, i) => {
        const x = padding + i * step;
        const y = height - padding - (value / max) * (height - padding * 2);
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    
    ctx.stroke();
    ctx.shadowBlur = 0;
    
    // Draw points
    ctx.fillStyle = '#00d4ff';
    data.forEach((value, i) => {
        const x = padding + i * step;
        const y = height - padding - (value / max) * (height - padding * 2);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
    });
}

// Polling
function startPolling() {
    fetchStatus();
    setInterval(fetchStatus, CONFIG.POLL_INTERVAL);
}

async function fetchStatus() {
    try {
        const response = await fetch(CONFIG.API_ENDPOINT);
        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        updateUI(data);
        setOnlineStatus(true);
        
    } catch (error) {
        console.error('❌ Error:', error);
        setOnlineStatus(false);
    }
}

// UI Updates
function updateUI(data) {
    updateCounter(data.people_count);
    updateDensity(data.density_percent); // Use density_percent for the meter
    updateRiskLevel(data.risk_level, data.risk_description);
    updateSurge(data.surge_score);
    updatePredictions(data.predictions); // FIXED: Now properly receives predictions
    updateChart(data.people_count);
    updateSystemInfo(data);
}

function updateCounter(count) {
    const formatted = String(count).padStart(4, '0');
    if (DOM.peopleCount.textContent !== formatted) {
        DOM.peopleCount.textContent = formatted;
        DOM.peopleCount.style.animation = 'none';
        setTimeout(() => {
            DOM.peopleCount.style.animation = 'counterGlow 2s ease-in-out infinite';
        }, 10);
        
        // Show change
        const change = count - appState.previousCount;
        if (change !== 0) {
            const arrow = change > 0 ? '▲' : '▼';
            const color = change > 0 ? '#00ff9d' : '#ff0055';
            DOM.changeIndicator.textContent = `${arrow} ${Math.abs(change)}`;
            DOM.changeIndicator.style.color = color;
            setTimeout(() => {
                DOM.changeIndicator.textContent = '';
            }, 3000);
        }
        
        appState.previousCount = count;
    }
}

function updateDensity(densityPercent) {
    // FIXED: Use actual density percentage from API
    const rounded = Math.round(densityPercent);
    DOM.densityValue.textContent = rounded;
    drawDensityMeter(DOM.densityCanvas.getContext('2d'), rounded);
}

function updateRiskLevel(level, description) {
    level = level.toUpperCase();
    
    // Remove all classes
    DOM.riskHexagon.className = 'risk-hexagon';
    
    // Set color and description
    if (level === 'LOW') {
        DOM.riskHexagon.classList.add('low');
        DOM.riskLevel.style.color = '#00ff9d';
        DOM.riskGlow.style.background = '#00ff9d';
    } else if (level === 'MEDIUM') {
        DOM.riskHexagon.classList.add('medium');
        DOM.riskLevel.style.color = '#ffaa00';
        DOM.riskGlow.style.background = '#ffaa00';
    } else if (level === 'HIGH') {
        DOM.riskHexagon.classList.add('high');
        DOM.riskLevel.style.color = '#ff4400';
        DOM.riskGlow.style.background = '#ff4400';
    } else if (level === 'CRITICAL' || level === 'EMERGENCY') {
        DOM.riskHexagon.classList.add('high');
        DOM.riskLevel.style.color = '#ff0055';
        DOM.riskGlow.style.background = '#ff0055';
    }
    
    DOM.riskLevel.textContent = level;
    
    // FIXED: Use description from API
    if (description) {
        DOM.riskDescription.textContent = description;
    }
}

function updateSurge(score) {
    // FIXED: Score is already 0-100 from API
    const surgePct = Math.round(score);
    DOM.surgeValue.textContent = `${surgePct}%`;
    
    // Update surge bars
    const bars = [
        document.getElementById('surgeBar1'),
        document.getElementById('surgeBar2'),
        document.getElementById('surgeBar3'),
        document.getElementById('surgeBar4'),
        document.getElementById('surgeBar5'),
    ];
    
    bars.forEach((bar, i) => {
        const threshold = (i + 1) * 20; // 20%, 40%, 60%, 80%, 100%
        const height = surgePct >= threshold ? 100 : (surgePct / threshold) * 100;
        bar.style.height = `${Math.max(height, 0)}%`;
    });
}

function updatePredictions(predictions) {
    // FIXED: Properly handle predictions object
    if (!predictions) {
        console.log('No predictions data received');
        return;
    }
    
    console.log('Predictions:', predictions); // Debug log
    
    // Update prediction values
    DOM.predCurrent.textContent = predictions.current || 0;
    DOM.pred5s.textContent = predictions.predicted_5s || predictions.current || 0;
    DOM.pred10s.textContent = predictions.predicted_10s || predictions.current || 0;
    
    // Update trend
    const trend = predictions.trend || 'stable';
    DOM.trendValue.textContent = trend.toUpperCase();
    
    // Color code trend
    if (trend === 'increasing') {
        DOM.trendValue.style.color = '#ff0055';
    } else if (trend === 'decreasing') {
        DOM.trendValue.style.color = '#00ff9d';
    } else {
        DOM.trendValue.style.color = '#00d4ff';
    }
    
    // Update confidence
    const confidence = predictions.confidence || 0;
    DOM.confidenceFill.style.width = `${confidence}%`;
    DOM.confidencePercent.textContent = `${Math.round(confidence)}%`;
}

function updateChart(count) {
    appState.chartData.push(count);
    if (appState.chartData.length > appState.maxChartPoints) {
        appState.chartData.shift();
    }
    drawTimelineChart(DOM.timelineChart.getContext('2d'), appState.chartData);
}

function updateSystemInfo(data) {
    DOM.sysStatus.textContent = 'ONLINE';
    DOM.sysStatus.style.color = '#00ff9d';
    
    if (data.video_progress !== undefined) {
        DOM.videoProgress.textContent = `${Math.round(data.video_progress)}%`;
    }
    
    if (data.timestamp) {
        const time = new Date(data.timestamp).toLocaleTimeString();
        DOM.lastUpdate.textContent = time;
    }
}

function setOnlineStatus(online) {
    appState.isOnline = online;
    DOM.systemStatus.textContent = online ? 'ONLINE' : 'OFFLINE';
    DOM.systemStatus.style.color = online ? '#00ff9d' : '#ff0055';
    DOM.systemDot.style.background = online ? '#00ff9d' : '#ff0055';
    DOM.systemDot.style.boxShadow = online ? '0 0 15px #00ff9d' : '0 0 15px #ff0055';
}

function updateTime() {
    const now = new Date();
    const time = now.toLocaleTimeString('en-US', { hour12: false });
    DOM.currentTime.textContent = time;
}

function updateUptime() {
    const elapsed = Math.floor((Date.now() - appState.startTime) / 1000);
    const hours = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const minutes = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const seconds = String(elapsed % 60).padStart(2, '0');
    DOM.uptime.textContent = `${hours}:${minutes}:${seconds}`;
}

// Console
console.log('%c🎮 CYBER INTELLIGENCE SYSTEM', 'color: #00ff9d; font-size: 20px; font-weight: bold;');
console.log('%cSystem Online ✓', 'color: #00d4ff; font-size: 14px;');
console.log('%cPrediction Module Active', 'color: #ffaa00; font-size: 14px;');