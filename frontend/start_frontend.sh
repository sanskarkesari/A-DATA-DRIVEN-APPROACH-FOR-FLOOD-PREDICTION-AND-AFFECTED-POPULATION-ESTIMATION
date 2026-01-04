#!/bin/bash
# Simple script to start a local server for the frontend
# This allows you to run the frontend separately from Flask

echo "Starting frontend server..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
python3 -m http.server 8000







