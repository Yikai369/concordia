# API Key Setup Guide

## Quick Start

The script reads the OpenAI API key from the `OPENAI_API_KEY` environment variable.

## Option 1: Environment Variable (Recommended)

### Windows PowerShell (Current Session)
```powershell
$Env:OPENAI_API_KEY = "sk-your-api-key-here"
python main.py --turns 2
```

### Windows Command Prompt (Current Session)
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
python main.py --turns 2
```

### Windows (Permanent)
1. Open System Properties → Environment Variables
2. Add new User variable: `OPENAI_API_KEY` = `sk-your-api-key-here`
3. Restart your terminal/IDE

### Linux/Mac (Current Session)
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
python main.py --turns 2
```

### Linux/Mac (Permanent - Add to ~/.bashrc or ~/.zshrc)
```bash
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Option 2: .env File (Convenient)

1. **Install python-dotenv** (optional, but recommended):
   ```bash
   pip install python-dotenv
   ```

2. **Create `.env` file** in `projects/impression_management/`:
   ```bash
   cd projects/impression_management
   cp .env.example .env
   ```

3. **Edit `.env`** and add your API key:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

4. **Run the script** - it will automatically load the `.env` file:
   ```bash
   python main.py --turns 2
   ```

**Note**: The `.env` file is already in `.gitignore` and will NOT be committed to version control.

## Option 3: Local Model (No API Key Needed)

If you want to avoid using an API key, you can use a local Ollama model:

```bash
python main.py --turns 2 --llm_type local --local_model llama3.1:8b
```

This requires Ollama to be installed and running locally.

## Security Best Practices

1. ✅ **DO**: Use environment variables or `.env` files
2. ✅ **DO**: Add `.env` to `.gitignore` (already done)
3. ❌ **DON'T**: Hardcode API keys in source code
4. ❌ **DON'T**: Commit `.env` files to version control
5. ❌ **DON'T**: Share API keys in chat/email

## Troubleshooting

**Error**: `ERROR: OPENAI_API_KEY environment variable required for OpenAI.`

**Solution**: Make sure you've set the environment variable or created a `.env` file with your API key.

**Check if variable is set**:
- PowerShell: `$Env:OPENAI_API_KEY`
- Command Prompt: `echo %OPENAI_API_KEY%`
- Linux/Mac: `echo $OPENAI_API_KEY`
