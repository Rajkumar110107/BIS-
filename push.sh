#!/bin/bash

echo "🚀 Pushing project to GitHub..."

# Check status
git status

# Add all files
git add .

# Commit (auto message with date)
git commit -m "Update: $(date '+%Y-%m-%d %H:%M:%S')" || echo "⚠️ Nothing to commit"

# Pull latest changes (avoid conflicts)
git pull origin main --rebase

# Push to GitHub
git push origin main

echo "✅ Push complete!"