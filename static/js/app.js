/**
 * Frontend Application Logic for Chest X-Ray Disease Detection
 * Handles image upload, API communication, and UI updates
 */

// API Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// DOM Elements
let uploadZone, fileInput, imagePreview, previewImage, removeImageBtn;
let resultsSection, predictedClass, confidenceScore, confidenceBars;
let gradcamSection, gradcamImage;
let historyItems, loadingOverlay;

// State
let currentFile = null;

/**
 * Initialize the application
 */
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    setupEventListeners();
    loadPredictionHistory();

    console.log('Application initialized');
});

/**
 * Initialize DOM element references
 */
function initializeElements() {
    uploadZone = document.getElementById('uploadZone');
    fileInput = document.getElementById('fileInput');
    imagePreview = document.getElementById('imagePreview');
    previewImage = document.getElementById('previewImage');
    removeImageBtn = document.getElementById('removeImage');

    resultsSection = document.getElementById('resultsSection');
    predictedClass = document.getElementById('predictedClass');
    confidenceScore = document.getElementById('confidenceScore');
    confidenceBars = document.getElementById('confidenceBars');

    gradcamSection = document.getElementById('gradcamSection');
    gradcamImage = document.getElementById('gradcamImage');

    historyItems = document.getElementById('historyItems');
    loadingOverlay = document.getElementById('loadingOverlay');
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Upload zone click
    uploadZone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Remove image button
    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    // Analyze button
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeImage);
    }
}

/**
 * Handle file selection from input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Handle drag over event
 */
function handleDragOver(event) {
    event.preventDefault();
    uploadZone.classList.add('drag-over');
}

/**
 * Handle drag leave event
 */
function handleDragLeave(event) {
    event.preventDefault();
    uploadZone.classList.remove('drag-over');
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    uploadZone.classList.remove('drag-over');

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Process selected file
 */
function processFile(file) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPEG or PNG)');
        return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size must be less than 16MB');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        imagePreview.classList.add('active');
        uploadZone.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Auto-analyze after a short delay
    setTimeout(() => analyzeImage(), 500);
}

/**
 * Reset upload state
 */
function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.classList.remove('active');
    uploadZone.style.display = 'block';
    resultsSection.classList.remove('active');
    gradcamSection.classList.add('hidden');
}

/**
 * Analyze the uploaded image
 */
async function analyzeImage() {
    console.log('analyzeImage called');

    if (!currentFile) {
        showError('Please upload an image first');
        return;
    }

    console.log('Starting analysis for file:', currentFile.name);

    // Show loading overlay
    showLoading('Analyzing X-ray image...', 'This may take a few seconds');

    // Hide previous results
    resultsSection.classList.remove('active');
    gradcamSection.classList.add('hidden');

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);

        console.log('Sending request to API...');

        // Send request to API
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        console.log('Response received:', response.status, response.statusText);

        const data = await response.json();
        console.log('Response data:', data);

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        // Display results
        console.log('Displaying results...');
        displayResults(data);

        // Reload history
        loadPredictionHistory();

    } catch (error) {
        console.error('Error during analysis:', error);
        showError(error.message || 'An error occurred during analysis');
    } finally {
        hideLoading();
    }
}

/**
 * Display prediction results
 */
function displayResults(data) {
    const { prediction, gradcam } = data;

    // Display predicted class and confidence
    predictedClass.textContent = prediction.predicted_class;
    confidenceScore.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;

    // Apply color based on prediction
    const resultCard = document.querySelector('.prediction-result');
    if (prediction.predicted_class === 'Normal') {
        resultCard.style.background = 'linear-gradient(135deg, #06d6a0 0%, #00a8e8 100%)';
    } else {
        resultCard.style.background = 'linear-gradient(135deg, #ff9f1c 0%, #ef476f 100%)';
    }

    // Display confidence bars for all classes
    displayConfidenceBars(prediction.all_predictions);

    // Display Grad-CAM if available
    if (gradcam) {
        gradcamImage.src = gradcam;
        gradcamSection.classList.remove('hidden');
    }

    // Show results section with animation
    resultsSection.classList.add('active');

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 300);
}

/**
 * Display confidence bars for all disease classes
 */
function displayConfidenceBars(predictions) {
    confidenceBars.innerHTML = '';

    // Sort predictions by confidence (highest first)
    const sortedPredictions = Object.entries(predictions)
        .sort((a, b) => b[1] - a[1]);

    sortedPredictions.forEach(([className, confidence]) => {
        const item = document.createElement('div');
        item.className = 'confidence-item';

        const percentage = (confidence * 100).toFixed(1);

        item.innerHTML = `
            <div class="confidence-label">
                <span class="class-name">${className}</span>
                <span class="confidence-value">${percentage}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: 0%"></div>
            </div>
        `;

        confidenceBars.appendChild(item);

        // Animate bar fill
        setTimeout(() => {
            const barFill = item.querySelector('.confidence-bar-fill');
            barFill.style.width = `${percentage}%`;
        }, 100);
    });
}

/**
 * Load prediction history
 */
async function loadPredictionHistory() {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=20`);
        const data = await response.json();

        if (data.success && data.history.length > 0) {
            displayHistory(data.history);
        } else {
            displayEmptyHistory();
        }
    } catch (error) {
        console.error('Error loading history:', error);
        displayEmptyHistory();
    }
}

/**
 * Display prediction history
 */
function displayHistory(history) {
    historyItems.innerHTML = '';

    history.forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.style.animationDelay = `${index * 0.05}s`;

        const timestamp = new Date(item.timestamp);
        const timeAgo = getTimeAgo(timestamp);

        historyItem.innerHTML = `
            <div class="history-item-content">
                <img src="${item.thumbnail}" alt="X-ray" class="history-thumbnail">
                <div class="history-info">
                    <div class="history-class">${item.predicted_class}</div>
                    <div class="history-confidence">${(item.confidence * 100).toFixed(1)}% confidence</div>
                    <div class="history-time">${timeAgo}</div>
                </div>
            </div>
        `;

        // Click to view details
        historyItem.addEventListener('click', () => {
            showHistoryDetails(item);
        });

        historyItems.appendChild(historyItem);
    });
}

/**
 * Display empty history message
 */
function displayEmptyHistory() {
    historyItems.innerHTML = `
        <div class="empty-history">
            <div class="empty-history-icon">📋</div>
            <p>No predictions yet</p>
            <p style="font-size: 0.85rem;">Upload an X-ray image to get started</p>
        </div>
    `;
}

/**
 * Show history item details
 */
function showHistoryDetails(item) {
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Display the prediction
    displayResults({
        prediction: {
            predicted_class: item.predicted_class,
            confidence: item.confidence,
            all_predictions: item.all_predictions
        },
        gradcam: item.gradcam
    });
}

/**
 * Get time ago string
 */
function getTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);

    const intervals = {
        year: 31536000,
        month: 2592000,
        week: 604800,
        day: 86400,
        hour: 3600,
        minute: 60
    };

    for (const [unit, secondsInUnit] of Object.entries(intervals)) {
        const interval = Math.floor(seconds / secondsInUnit);
        if (interval >= 1) {
            return `${interval} ${unit}${interval > 1 ? 's' : ''} ago`;
        }
    }

    return 'Just now';
}

/**
 * Show loading overlay
 */
function showLoading(text, subtext) {
    const loadingText = document.getElementById('loadingText');
    const loadingSubtext = document.getElementById('loadingSubtext');

    if (loadingText) loadingText.textContent = text;
    if (loadingSubtext) loadingSubtext.textContent = subtext;

    loadingOverlay.classList.add('active');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.classList.remove('active');
}

/**
 * Show error message
 */
function showError(message) {
    alert(`Error: ${message}`);
    console.error(message);
}
