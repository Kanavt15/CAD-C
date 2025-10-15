// API Configuration
const API_URL = 'http://localhost:5000';

// DOM Elements - with null checks
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsCard = document.getElementById('resultsCard');
const resultsContainer = document.getElementById('resultsContainer');

// Verify critical elements exist
if (!imageInput || !uploadArea || !analyzeBtn || !loadingIndicator || !resultsCard) {
    console.error('Critical DOM elements missing! Check HTML IDs.');
    console.log('Elements found:', {
        imageInput: !!imageInput,
        uploadArea: !!uploadArea,
        analyzeBtn: !!analyzeBtn,
        loadingIndicator: !!loadingIndicator,
        resultsCard: !!resultsCard
    });
}

// State
let selectedImage = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIHealth();
});

function setupEventListeners() {
    if (!imageInput || !uploadArea || !analyzeBtn) {
        console.error('Cannot setup event listeners - elements missing');
        return;
    }
    
    // File input change
    imageInput.addEventListener('change', handleImageSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Analyze button
    analyzeBtn.addEventListener('click', analyzeImage);
    
    // Model checkboxes
    const modelCheckboxes = document.querySelectorAll('input[name="model"]');
    modelCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateAnalyzeButton);
    });
}

// API Health Check
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/api/health`);
        const data = await response.json();
        console.log('API Health:', data);
        
        if (data.status === 'healthy') {
            showNotification('System ready! Models loaded successfully.', 'success');
        }
    } catch (error) {
        console.error('API Health Check Failed:', error);
        showNotification('Warning: Could not connect to backend server. Please ensure the Flask app is running.', 'warning');
    }
}

// Image Selection
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        loadImage(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        imageInput.files = event.dataTransfer.files;
        loadImage(file);
    } else {
        showNotification('Please drop a valid image file.', 'error');
    }
}

function loadImage(file) {
    selectedImage = file;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadArea.querySelector('.upload-placeholder').style.display = 'none';
        previewContainer.style.display = 'block';
        updateAnalyzeButton();
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    selectedImage = null;
    imageInput.value = '';
    imagePreview.src = '';
    uploadArea.querySelector('.upload-placeholder').style.display = 'block';
    previewContainer.style.display = 'none';
    resultsCard.style.display = 'none';
    updateAnalyzeButton();
}

function updateAnalyzeButton() {
    const hasImage = selectedImage !== null;
    const hasSelectedModel = Array.from(document.querySelectorAll('input[name="model"]:checked')).length > 0;
    analyzeBtn.disabled = !(hasImage && hasSelectedModel);
}

// Analysis
async function analyzeImage() {
    if (!selectedImage) {
        showNotification('Please select an image first.', 'error');
        return;
    }
    
    const selectedModels = Array.from(document.querySelectorAll('input[name="model"]:checked'))
        .map(cb => cb.value);
    
    if (selectedModels.length === 0) {
        showNotification('Please select at least one model.', 'error');
        return;
    }
    
    // Show loading - with safety checks
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (analyzeBtn) analyzeBtn.disabled = true;
    if (resultsCard) resultsCard.style.display = 'none';
    
    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('image', selectedImage);
        selectedModels.forEach(model => {
            formData.append('models[]', model);
        });
        
        // Make API request
        const response = await fetch(`${API_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        console.log('API Response:', data);
        
        if (data.success) {
            displayResults(data.results);
            showNotification('Analysis completed successfully!', 'success');
            
            // Scroll to results
            setTimeout(() => {
                if (resultsCard) {
                    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 300);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis Error:', error);
        showNotification(`Analysis failed: ${error.message}`, 'error');
    } finally {
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        if (analyzeBtn) analyzeBtn.disabled = false;
    }
}

// Display Results
function displayResults(results) {
    console.log('Displaying results:', results);
    
    if (!resultsContainer) {
        console.error('Results container not found!');
        return;
    }
    
    resultsContainer.innerHTML = '';
    
    if (!results || results.length === 0) {
        resultsContainer.innerHTML = '<p style="color: var(--text-light);">No results to display.</p>';
        if (resultsCard) resultsCard.style.display = 'block';
        return;
    }
    
    // Sort results: individual models first, then ensemble
    const sortedResults = results.sort((a, b) => {
        if (a.model === 'ensemble') return 1;
        if (b.model === 'ensemble') return -1;
        return 0;
    });
    
    sortedResults.forEach(result => {
        if (result.error) {
            resultsContainer.innerHTML += createErrorResult(result);
        } else {
            resultsContainer.innerHTML += createResultCard(result);
        }
    });
    
    if (resultsCard) resultsCard.style.display = 'block';
    console.log('Results card displayed');
}

function createResultCard(result) {
    const isCancerous = result.prediction === 'Cancerous';
    const isEnsemble = result.model === 'ensemble';
    const confidence = result.confidence;
    
    let confidenceLevel = 'high';
    if (confidence < 70) confidenceLevel = 'medium';
    if (confidence < 50) confidenceLevel = 'low';
    
    const modelNames = {
        'densenet': 'DenseNet169',
        'efficientnet': 'EfficientNet-B0',
        'resnet101': 'ResNet101',
        'ensemble': 'Ensemble Prediction'
    };
    
    const modelIcons = {
        'densenet': 'fa-network-wired',
        'efficientnet': 'fa-bolt',
        'resnet101': 'fa-layer-group',
        'ensemble': 'fa-brain'
    };
    
    return `
        <div class="result-item ${isEnsemble ? 'ensemble' : ''}">
            <div class="result-header">
                <h3>
                    <i class="fas ${modelIcons[result.model] || 'fa-robot'}"></i>
                    ${modelNames[result.model] || result.model}
                </h3>
                ${isEnsemble ? '<span class="badge badge-primary">Ensemble</span>' : ''}
            </div>
            
            <div class="result-prediction ${isCancerous ? 'cancerous' : 'non-cancerous'}">
                <i class="fas ${isCancerous ? 'fa-exclamation-circle' : 'fa-check-circle'}"></i>
                ${result.prediction}
            </div>
            
            <div class="result-confidence">
                Confidence: <strong>${confidence.toFixed(2)}%</strong>
            </div>
            
            ${result.threshold ? `
            <div class="threshold-info" style="font-size: 0.85em; color: var(--text-light); margin: 8px 0; padding: 6px 12px; background: rgba(255,255,255,0.05); border-radius: 6px;">
                <i class="fas fa-ruler-horizontal"></i> Classification Threshold: <strong>${result.threshold}%</strong>
                <br><span style="font-size: 0.9em;">(≥ ${result.threshold}% cancerous probability → Cancerous)</span>
            </div>
            ` : ''}
            
            <div class="confidence-bar">
                <div class="confidence-fill ${confidenceLevel}" style="width: ${confidence}%"></div>
            </div>
            
            <div class="probabilities">
                <div class="probability-item">
                    <div class="probability-label">
                        <i class="fas fa-check-circle" style="color: var(--success-color);"></i>
                        Non-Cancerous
                    </div>
                    <div class="probability-value" style="color: var(--success-color);">
                        ${result.probabilities.non_cancerous.toFixed(2)}%
                    </div>
                </div>
                <div class="probability-item">
                    <div class="probability-label">
                        <i class="fas fa-exclamation-circle" style="color: var(--danger-color);"></i>
                        Cancerous
                    </div>
                    <div class="probability-value" style="color: var(--danger-color);">
                        ${result.probabilities.cancerous.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>
    `;
}

function createErrorResult(result) {
    return `
        <div class="result-item" style="border-color: var(--danger-color);">
            <div class="result-header">
                <h3>
                    <i class="fas fa-exclamation-triangle" style="color: var(--danger-color);"></i>
                    ${result.model}
                </h3>
                <span class="badge badge-danger">Error</span>
            </div>
            <p style="color: var(--text-light);">${result.error}</p>
        </div>
    `;
}

// Notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-color)' : 
                     type === 'error' ? 'var(--danger-color)' : 
                     type === 'warning' ? 'var(--warning-color)' : 
                     'var(--primary-color)'};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        max-width: 400px;
        animation: slideInRight 0.3s ease-out;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    `;
    
    const icon = type === 'success' ? 'fa-check-circle' :
                 type === 'error' ? 'fa-exclamation-circle' :
                 type === 'warning' ? 'fa-exclamation-triangle' :
                 'fa-info-circle';
    
    notification.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});
