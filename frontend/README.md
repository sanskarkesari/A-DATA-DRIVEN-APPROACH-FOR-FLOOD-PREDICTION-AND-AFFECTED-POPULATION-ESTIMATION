# Frontend for Flood Prediction System

A modern, responsive web interface for the Flood Prediction System.

## Features

- 🎨 **Modern UI** - Clean, professional design
- 📱 **Responsive** - Works on desktop, tablet, and mobile
- ⚡ **Real-time** - Live API integration with Flask backend
- 📊 **Visual Results** - Beautiful display of predictions and features
- ✅ **Input Validation** - Ensures data quality
- 🔄 **Loading States** - User-friendly feedback

## Usage

### Option 1: Open Directly in Browser

Simply open `index.html` in your web browser:

```bash
# From the frontend directory
cd frontend
# Open index.html in your browser
```

Or use Python's simple HTTP server:

```bash
cd frontend
python3 -m http.server 8000
# Then open http://localhost:8000 in your browser
```

### Option 2: Serve with Flask (Recommended)

You can also serve the frontend through Flask by adding a route in `app.py`:

```python
@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('frontend', path)
```

## Configuration

The frontend is configured to connect to `http://localhost:5000` by default.

To change the API URL, edit `script.js`:

```javascript
const API_BASE_URL = 'http://your-api-url:5000';
```

## API Endpoints Used

- `GET /` - Health check
- `POST /predict` - Get flood prediction

## Browser Compatibility

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## Features Displayed

The frontend displays:

1. **Flood Risk Prediction**
   - Risk Level (Low/Medium/High)
   - Flood Probability (%)
   - Interpretation message

2. **Extracted Features**
   - Daily Rainfall
   - 30-Day Cumulative Rainfall
   - Average Daily Rainfall
   - Soil Moisture
   - Elevation
   - Slope
   - Flow Accumulation
   - NDWI (Normalized Difference Water Index)

3. **Location Preview**
   - Coordinates display

## Customization

### Colors

Edit `style.css` to change the color scheme:

```css
:root {
    --primary-color: #2563eb;
    --low-risk: #10b981;
    --medium-risk: #f59e0b;
    --high-risk: #ef4444;
}
```

### Styling

All styles are in `style.css`. The design uses:
- CSS Grid for layouts
- Flexbox for alignment
- CSS Variables for theming
- Smooth animations and transitions

## Troubleshooting

### API Connection Error

If you see "Failed to connect to API":
1. Make sure Flask server is running (`python app.py`)
2. Check that API_BASE_URL in `script.js` matches your Flask server URL
3. Check browser console for CORS errors

### CORS Issues

If you get CORS errors, make sure Flask-CORS is installed and configured in `app.py`:

```python
from flask_cors import CORS
CORS(app)
```

### Model Not Loaded Warning

If you see "Model not loaded":
1. Train the model first: `python train_model.py`
2. Make sure model files exist in `models/` directory







