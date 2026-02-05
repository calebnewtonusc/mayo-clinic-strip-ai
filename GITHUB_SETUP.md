# GitHub Setup Instructions

The local git repository has been initialized and the initial commit has been created. Follow these steps to push to GitHub:

## Option 1: Using GitHub CLI (gh)

```bash
cd /Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai

# Authenticate with GitHub (if not already done)
gh auth login

# Create repository and push
gh repo create mayo-clinic-strip-ai --public --source=. --remote=origin --push
```

## Option 2: Manual Setup

1. Go to GitHub and create a new repository named `mayo-clinic-strip-ai`
2. Do NOT initialize with README, .gitignore, or license (we already have these)
3. Copy the repository URL
4. Run these commands:

```bash
cd /Users/joelnewton/Desktop/2026-Code/projects/mayo-clinic-strip-ai

# Add the remote repository
git remote add origin https://github.com/YOUR_USERNAME/mayo-clinic-strip-ai.git

# Push the code
git push -u origin main
```

## Repository Settings Recommendations

### Collaborators
- Add your friends as collaborators in Settings > Collaborators
- Set appropriate permissions (Write or Admin)

### Branch Protection (Optional but Recommended)
- Protect the `main` branch
- Require pull request reviews before merging
- Require status checks to pass

### Repository Description
```
Deep learning classification of stroke blood clot origin (CE vs LAA) using medical imaging. Mayo Clinic STRIP AI project for healthcare ML research.
```

### Topics/Tags
- `machine-learning`
- `deep-learning`
- `medical-imaging`
- `healthcare`
- `stroke-classification`
- `pytorch`
- `computer-vision`

## Cloning for Your Friends

Once pushed, your friends can clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mayo-clinic-strip-ai.git
cd mayo-clinic-strip-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Next Steps After Pushing

1. Add your friends as collaborators
2. Create issues for major tasks from the implementation plan
3. Set up a project board to track progress
4. Review the implementation plan in docs/IMPLEMENTATION_PLAN.md
5. Start with Phase 1: Environment Setup & Data Infrastructure
