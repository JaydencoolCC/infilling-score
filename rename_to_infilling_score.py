#!/usr/bin/env python3
"""
Script to rename the infilling_score project to infilling_score
"""

import os
import shutil
import re
from pathlib import Path

def replace_in_file(file_path, old_text, new_text):
    """Replace text in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes needed: {file_path}")
    except Exception as e:
        print(f"Error updating {file_path}: {e}")

def rename_project():
    """Rename the project from infilling_score to infilling_score."""
    
    print("Starting project rename from 'infilling_score' to 'infilling_score'...")
    
    # Step 1: Uninstall old package
    print("\n1. Uninstalling old package...")
    os.system("pip uninstall infilling-score -y")
    
    # Step 2: Remove old egg-info directory
    if os.path.exists("infilling_score.egg-info"):
        shutil.rmtree("infilling_score.egg-info")
        print("Removed old egg-info directory")
    
    # Step 3: Rename the main package directory
    if os.path.exists("infilling_score"):
        if os.path.exists("infilling_score"):
            shutil.rmtree("infilling_score")
        shutil.move("infilling_score", "infilling_score")
        print("Renamed package directory: infilling_score -> infilling_score")
    
    # Step 4: Update imports in all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    
    print(f"\n2. Updating imports in {len(python_files)} Python files...")
    
    replacements = [
        ("infilling_score", "infilling_score"),
        ("infilling-score", "infilling-score"),
        ("Infilling Score", "Infilling Score"),
        ("infilling score", "infilling score"),
        ("InfillingScoreDetector", "InfillingScoreDetector"),
    ]
    
    for file_path in python_files:
        for old_text, new_text in replacements:
            replace_in_file(file_path, old_text, new_text)
    
    # Step 5: Update setup.py
    print("\n3. Updating setup.py...")
    setup_replacements = [
        ("infilling_score", "infilling_score"),
        ("infilling-score", "infilling-score"),
        ("Optimized infilling score detection for language models", 
         "Optimized infilling score detection for language models"),
        ("Membership Inference Team", "Infilling Score Team"),
    ]
    
    for old_text, new_text in setup_replacements:
        replace_in_file("setup.py", old_text, new_text)
    
    # Step 6: Update README.md
    print("\n4. Updating README.md...")
    readme_replacements = [
        ("# Infilling Score Detection", "# Infilling Score Detection"),
        ("infilling_score", "infilling_score"),
        ("infilling-score", "infilling-score"),
        ("Infilling Score", "Infilling Score"),
        ("infilling score", "infilling score"),
        ("InfillingScoreDetector", "InfillingScoreDetector"),
    ]
    
    for old_text, new_text in readme_replacements:
        replace_in_file("README.md", old_text, new_text)
    
    # Step 7: Create/update .gitignore
    print("\n5. Creating .gitignore...")
    gitignore_content = """# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Project specific
results/
*.csv
*.log
models/
checkpoints/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Step 8: Install new package
    print("\n6. Installing renamed package...")
    os.system("pip install -e .")
    
    print("\n✅ Project successfully renamed to 'infilling_score'!")
    print("\nNext steps:")
    print("1. Test the renamed project: python main.py --help")
    print("2. Initialize git repository: git init")
    print("3. Add files: git add .")
    print("4. Make initial commit: git commit -m 'Initial commit'")
    print("5. Create GitHub repository and push")

if __name__ == "__main__":
    rename_project()

