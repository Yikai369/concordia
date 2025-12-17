# Quick Start: Where to Place Your Project

## Recommended: Sibling Directory (Best Practice)

**Current location:** `C:\Users\22455\OneDrive\Documents\concordia\` (Concordia framework)

**Create your project here:** `C:\Users\22455\OneDrive\Documents\my_project\`

### Steps:

1. **Navigate to parent directory:**
   ```cmd
   cd C:\Users\22455\OneDrive\Documents
   ```

2. **Create your project folder:**
   ```cmd
   mkdir my_project
   cd my_project
   ```

3. **Copy the template (optional):**
   ```cmd
   xcopy ..\concordia\example_project_template\* . /E /I
   ```

4. **Install Concordia** (choose one):

   **Option A: Install from PyPI (recommended for production):**
   ```cmd
   pip install gdm-concordia
   ```

   **Option B: Use local framework (for development):**
   ```cmd
   pip install -e ..\concordia
   ```

   **Option C: Add to PYTHONPATH (temporary):**
   ```cmd
   set PYTHONPATH=C:\Users\22455\OneDrive\Documents\concordia;%PYTHONPATH%
   ```

5. **Start building your project!**

## Alternative: Projects Folder (If you prefer everything together)

If you want to keep your project inside the Concordia directory:

1. **Create projects folder:**
   ```cmd
   cd C:\Users\22455\OneDrive\Documents\concordia
   mkdir projects
   cd projects
   mkdir my_project
   ```

2. **Your project will be at:**
   `C:\Users\22455\OneDrive\Documents\concordia\projects\my_project\`

3. **Import Concordia** (it's already in the parent directory):
   ```python
   # In your scripts, Concordia is already importable:
   from concordia.prefabs.simulation import generic as simulation
   ```

## Which Should You Choose?

- **Sibling Directory**: Better for production projects, cleaner separation
- **Projects Folder**: Better for experimentation, easier to reference framework code
