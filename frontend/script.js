// API Configuration
// If frontend is served from Flask, use relative URLs
// Otherwise, use full URL to Flask API
const API_BASE_URL = window.location.origin === 'http://localhost:5000' 
    ? ''  // Same origin, use relative URLs
    : 'http://localhost:5000';  // Different origin, use full URL

// DOM Elements
const form = document.getElementById('predictionForm');
const submitBtn = document.getElementById('submitBtn');
const clearBtn = document.getElementById('clearBtn');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
// Population impact elements
const populationCard = document.getElementById('populationCard');
const affectedPopulationValue = document.getElementById('affectedPopulationValue');
const affectedPopulationRange = document.getElementById('affectedPopulationRange');
const affectedPopulationBar = document.getElementById('affectedPopulationBar');
const affectedPopulationContext = document.getElementById('affectedPopulationContext');

// Initialize date input to today
document.getElementById('date').valueAsDate = new Date();

// Form Submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form values
    const latitude = parseFloat(document.getElementById('latitude').value);
    const longitude = parseFloat(document.getElementById('longitude').value);
    const date = document.getElementById('date').value;
    
    // Validate inputs
    if (!validateInputs(latitude, longitude, date)) {
        return;
    }
    
    // Show loading, hide results and errors
    showLoading();
    hideResults();
    hideError();
    
    // Disable submit button
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    try {
        // Make API request
        const predictUrl = API_BASE_URL ? `${API_BASE_URL}/predict` : '/predict';
        const response = await fetch(predictUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                latitude: latitude,
                longitude: longitude,
                date: date
            })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get prediction');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to API. Make sure the Flask server is running.');
    } finally {
        // Re-enable submit button
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-search"></i> Predict Flood & Population Impact';
        hideLoading();
    }
});

// Clear button
clearBtn.addEventListener('click', () => {
    form.reset();
    document.getElementById('date').valueAsDate = new Date();
    hideResults();
    hideError();
});

// Validation
function validateInputs(lat, lon, date) {
    // Check latitude range (Assam: 24.0 to 28.0)
    if (lat < 24.0 || lat > 28.0) {
        showError('Latitude must be between 24.0° and 28.0° (Assam region)');
        return false;
    }
    
    // Check longitude range (Assam: 89.0 to 96.0)
    if (lon < 89.0 || lon > 96.0) {
        showError('Longitude must be between 89.0° and 96.0° (Assam region)');
        return false;
    }
    
    // Check date
    if (!date) {
        showError('Please select a date');
        return false;
    }
    
    // Check if date is reasonable (not too far in the past or future)
    const selectedDate = new Date(date);
    const today = new Date();
    today.setHours(23, 59, 59, 999);
    const maxPastDate = new Date();
    maxPastDate.setFullYear(maxPastDate.getFullYear() - 5); // Allow up to 5 years in past
    
    // Allow future dates but show warning
    if (selectedDate > today) {
        const daysInFuture = Math.ceil((selectedDate - today) / (1000 * 60 * 60 * 24));
        if (daysInFuture > 30) {
            showError('Date cannot be more than 30 days in the future. Historical data is not available for future dates.');
            return false;
        }
        // Show warning but allow prediction
        console.warn(`Warning: Predicting for future date (${daysInFuture} days ahead). Using current/latest available data.`);
    }
    
    // Check if date is too far in the past
    if (selectedDate < maxPastDate) {
        showError('Date cannot be more than 5 years in the past. Historical data may not be available.');
        return false;
    }
    
    return true;
}

// Display Results
function displayResults(data) {
    if (!data.success) {
        showError(data.error || 'Prediction failed');
        return;
    }
    
    // Show warning if future date was used
    if (data.features && data.features.is_future_prediction) {
        const daysAhead = data.features.days_ahead || 0;
        const effectiveDate = data.features.effective_date || data.input.date;
        showWarning(`⚠️ Future date prediction: Using latest available data (${effectiveDate}). Historical data is not available for future dates.`);
    }
    
    // Update location preview
    const locationText = document.getElementById('locationText');
    locationText.textContent = `Lat: ${data.input.latitude.toFixed(4)}°, Lon: ${data.input.longitude.toFixed(4)}°`;
    
    // Update risk indicator
    const riskLevel = document.getElementById('riskLevel');
    const riskProbability = document.getElementById('riskProbability');
    const riskInterpretation = document.getElementById('riskInterpretation');
    
    const risk = data.prediction.flood_risk;
    const probability = data.prediction.flood_probability;
    
    riskLevel.textContent = risk;
    riskLevel.className = `risk-level ${risk.toLowerCase()}`;
    
    riskProbability.textContent = `${probability}%`;
    riskInterpretation.textContent = data.prediction.interpretation;
    
    // Display features
    displayFeatures(data.features);
    
    // Attempt to fetch affected population (optional, if inputs are provided)
    fetchAffectedPopulationIfPossible(data);
    
    // Show results section
    showResults();
}

// Display Features
function displayFeatures(features) {
    const featuresGrid = document.getElementById('featuresGrid');
    featuresGrid.innerHTML = '';
    
    const featureConfig = [
        { key: 'rainfall_mm', label: 'Daily Rainfall', unit: 'mm', icon: 'fa-cloud-rain' },
        { key: 'cumulative_rainfall_30d_mm', label: '30-Day Cumulative Rainfall', unit: 'mm', icon: 'fa-cloud-showers-heavy' },
        { key: 'avg_daily_rainfall_mm', label: 'Avg Daily Rainfall', unit: 'mm', icon: 'fa-cloud-sun-rain' },
        { key: 'soil_moisture_mm', label: 'Soil Moisture', unit: 'mm', icon: 'fa-seedling' },
        { key: 'elevation_m', label: 'Elevation', unit: 'm', icon: 'fa-mountain' },
        { key: 'slope_degree', label: 'Slope', unit: '°', icon: 'fa-chart-line' },
        { key: 'flow_accumulation', label: 'Flow Accumulation', unit: '', icon: 'fa-water' },
        { key: 'ndwi', label: 'NDWI', unit: '', icon: 'fa-swimming-pool' }
    ];
    
    featureConfig.forEach(config => {
        const value = features[config.key];
        if (value !== undefined && value !== null) {
            const featureItem = createFeatureItem(config.label, value, config.unit, config.icon);
            featuresGrid.appendChild(featureItem);
        }
    });
}

// Create Feature Item
function createFeatureItem(label, value, unit, icon) {
    const item = document.createElement('div');
    item.className = 'feature-item';
    
    const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
    
    item.innerHTML = `
        <div class="feature-label">
            <i class="fas ${icon}"></i> ${label}
        </div>
        <div class="feature-value">
            ${formattedValue}<span class="feature-unit">${unit}</span>
        </div>
    `;
    
    return item;
}

// UI Helper Functions
function showLoading() {
    loadingIndicator.classList.remove('hidden');
}

function hideLoading() {
    loadingIndicator.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    resultsSection.classList.add('hidden');
    // Hide population card as well
    if (populationCard) {
        populationCard.style.display = 'none';
    }
}

function showWarning(message) {
    // Show warning in a non-blocking way (yellow/orange style)
    const warningDiv = document.createElement('div');
    warningDiv.className = 'warning-message';
    warningDiv.style.cssText = `
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    `;
    warningDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>${message}</span>
    `;
    
    // Insert at top of results section or form
    const form = document.getElementById('predictionForm');
    form.parentNode.insertBefore(warningDiv, form.nextSibling);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
        if (warningDiv.parentNode) {
            warningDiv.parentNode.removeChild(warningDiv);
        }
    }, 10000);
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.classList.add('hidden');
}

// --------------------------------------------------------------------
// Affected Population Prediction (Optional Extension)
// --------------------------------------------------------------------

async function fetchAffectedPopulationIfPossible(floodData) {
    // Check if population model is available via health endpoint (optional optimization)
    // And ensure required additional inputs are provided
    const district = document.getElementById('district')?.value || '';
    const riverLevelInput = document.getElementById('riverLevel')?.value;
    const popDensityInput = document.getElementById('populationDensity')?.value;
    
    if (!district || !riverLevelInput || !popDensityInput) {
        // Missing optional inputs -> skip population prediction gracefully
        if (populationCard) {
            populationCard.style.display = 'none';
        }
        return;
    }
    
    try {
        const riverLevel = parseFloat(riverLevelInput);
        const populationDensity = parseFloat(popDensityInput);
        if (isNaN(riverLevel) || isNaN(populationDensity)) {
            return;
        }
        
        const dateStr = floodData.input.date;
        const dateObj = new Date(dateStr);
        const year = dateObj.getFullYear();
        const month = dateObj.getMonth() + 1; // JS months are 0-based
        
        const features = floodData.features || {};
        const floodPred = floodData.prediction || {};
        
        // Build payload using existing features + extra inputs
        const payload = {
            district: district,
            latitude: floodData.input.latitude,
            longitude: floodData.input.longitude,
            year: year,
            month: month,
            daily_rainfall_mm: features.rainfall_mm ?? features.daily_rainfall_mm ?? 0,
            cumulative_rainfall_30d_mm: features.cumulative_rainfall_30d_mm ?? 0,
            soil_moisture_mm: features.soil_moisture_mm ?? 0,
            river_water_level_m: riverLevel,
            elevation_m: features.elevation_m ?? 0,
            flood_risk_level: floodPred.flood_risk_level ?? 0,
            population_density_per_sqkm: populationDensity
        };
        
        const url = API_BASE_URL ? `${API_BASE_URL}/predict-affected-population` : '/predict-affected-population';
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const popData = await resp.json();
        if (!resp.ok || !popData.success) {
            console.warn('Population prediction not available:', popData.error || resp.statusText);
            if (populationCard) {
                populationCard.style.display = 'none';
            }
            return;
        }
        
        displayPopulationResult(popData);
    } catch (err) {
        console.warn('Error fetching affected population:', err);
        if (populationCard) {
            populationCard.style.display = 'none';
        }
    }
}

function displayPopulationResult(popData) {
    if (!populationCard || !affectedPopulationValue || !affectedPopulationRange || !affectedPopulationBar || !affectedPopulationContext) {
        return;
    }
    
    const pred = popData.prediction || {};
    const value = pred.affected_population ?? null;
    const range = pred.confidence_range ?? null;
    
    if (value === null) {
        populationCard.style.display = 'none';
        return;
    }
    
    populationCard.style.display = 'block';
    
    // Set numeric value and range
    affectedPopulationValue.textContent = Number(value).toLocaleString('en-IN');
    affectedPopulationRange.textContent = range ? `Estimated range: ${range} people` : '';
    
    // Enhanced visualization: scale to a max (250k for better visualization)
    const maxRef = 250000; // Reference max for bar scaling
    const frac = Math.max(0, Math.min(1, value / maxRef));
    const percentage = (frac * 100).toFixed(1);
    
    // Update bar visualization
    affectedPopulationBar.style.width = `${percentage}%`;
    
    // Update bar label
    const barLabel = document.getElementById('populationBarLabel');
    if (barLabel) {
        barLabel.textContent = `${percentage}%`;
    }
    
    // Update gauge visualization
    const gaugeFill = document.getElementById('gaugeFill');
    const gaugeText = document.getElementById('gaugeText');
    if (gaugeFill && gaugeText) {
        const circumference = 251.2; // 2 * PI * 40 (radius)
        const offset = circumference - (frac * circumference);
        gaugeFill.style.strokeDashoffset = offset;
        
        // Color based on severity
        if (frac < 0.33) {
            gaugeFill.setAttribute('stroke', '#10b981'); // Green
        } else if (frac < 0.66) {
            gaugeFill.setAttribute('stroke', '#f59e0b'); // Orange
        } else {
            gaugeFill.setAttribute('stroke', '#ef4444'); // Red
        }
        
        gaugeText.textContent = `${percentage}%`;
    }
    
    // Context text
    const riskLevel = (popData.flood_risk_level !== undefined)
        ? popData.flood_risk_level
        : (popData.input?.flood_risk_level ?? null);
    
    let context = 'Based on predicted flood severity and local population density.';
    if (riskLevel === 0) {
        context = 'Low flood risk – expected impact on population is minimal.';
    } else if (riskLevel === 1) {
        context = 'Medium flood risk – moderate number of people may be affected.';
    } else if (riskLevel === 2) {
        context = 'High flood risk – large population impact is likely. Consider evacuation and emergency planning.';
    }
    affectedPopulationContext.textContent = context;

    // Inline district risk map (show single district marker)
    try {
        const district = popData?.input?.district || document.getElementById('district')?.value || '';
        const mapWrapper = document.getElementById('districtMapWrapper');
        const mapFrame = document.getElementById('districtRiskMapFrame');
        if (district && mapWrapper && mapFrame) {
            const base = API_BASE_URL ? `${API_BASE_URL}/risk-map` : '/risk-map';
            mapFrame.src = `${base}?district=${encodeURIComponent(district)}`;
            mapWrapper.style.display = 'block';
        }
    } catch (e) {
        // If iframe fails, don't break population UI
    }
}

// Check API health on load
window.addEventListener('load', async () => {
    try {
        const healthUrl = API_BASE_URL ? `${API_BASE_URL}/api/health` : '/api/health';
        const response = await fetch(healthUrl);
        const data = await response.json();
        
        if (!data.model_loaded) {
            console.warn('Model not loaded. Train the model first.');
        }
    } catch (error) {
        console.warn('Could not connect to API. Make sure Flask server is running.');
    }
});

