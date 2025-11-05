# StructGPT with Ollama Integration

A modified version of [StructGPT](https://github.com/HKUNLP/StructGPT) that uses **Ollama with local LLMs** (like Gemma3) instead of OpenAI's API.

<p align="center">
  <img src="./asset/model.png" width="750" title="Overview of StructGPT" alt="">
</p>

## 🚀 Key Features

- ✅ **No API Keys Required** - Runs completely locally with Ollama
- ✅ **No OpenAI Dependencies** - Uses direct HTTP calls to Ollama
- ✅ **Free to Use** - No API costs, no rate limits
- ✅ **Privacy-First** - Your data never leaves your machine
- ✅ **GPU Accelerated** - Full CUDA/Metal support via Ollama
- ✅ **Multiple Models** - Works with Gemma3, Llama, Phi3, and more

## 📊 Performance

Tested on Spider dataset (1034 examples) with Gemma3:
- **Exact Match Accuracy:** 27.56%
- **Processing Time:** ~58 minutes (~3.4s per example)
- **GPU Utilization:** 85-95% (RTX 3060 Ti)
- **VRAM Usage:** ~5-6 GB

For comparison, the original paper reports GPT-3.5-turbo achieves ~70-75% accuracy.

## 📦 Installation

### 1. Clone this repository

```bash
git clone https://github.com/YOUR_USERNAME/StructGPT-Ollama.git
cd StructGPT-Ollama
```

### 2. Setup Python environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 3. Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download from [ollama.com](https://ollama.com/download)

### 4. Setup Ollama

```bash
# Start Ollama server (in a separate terminal, keep running)
ollama serve

# Pull the Gemma3 model
ollama pull gemma3

# Verify installation
ollama list
```

## 🎯 Quick Start

### Run Spider Dataset (Text-to-SQL)

```bash
source venv/bin/activate
bash ./scripts/run_spider_wo_icl_v1_ollama.sh
```

### Evaluate Results

```bash
python simple_evaluate.py ./outputs/spider/output_wo_icl_v1_ollama.jsonl
```

## 📚 Available Datasets

| Dataset | Task | Script |
|---------|------|--------|
| Spider | Text-to-SQL | `run_spider_wo_icl_v1_ollama.sh` |
| TabFact | Table Verification | `run_tabfact_wo_icl_v1_ollama.sh` |
| WikiTableQuestions | Table QA | `run_wtq_wo_icl_v1_ollama.sh` |
| WikiSQL | Text-to-SQL | `run_wikisql_wo_icl_v1_ollama.sh` |
| WebQSP | Knowledge Graph QA | `run_webqsp_wo_icl_v1_ollama.sh` |

## 🔧 Configuration

### Change the Model

Edit the model name at the top of any `*_ollama.py` file:

```python
OLLAMA_MODEL = "gemma3"  # Change to: llama3.2, phi3, etc.
```

### Change Ollama Endpoint

If running Ollama on a different host:

```python
OLLAMA_API_BASE = "http://localhost:11434/api"  # Change as needed
```

## 📖 Detailed Documentation

See [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) for:
- Complete setup instructions
- Troubleshooting guide
- Performance tuning tips
- GPU usage monitoring
- Model recommendations

## 🗂️ Project Structure

```
StructGPT-Ollama/
├── structgpt_for_text_to_sql_ollama.py    # Text-to-SQL implementation
├── structgpt_for_tableqa_ollama.py         # Table QA implementation
├── structgpt_for_webqsp_ollama.py          # Knowledge graph QA
├── simple_evaluate.py                       # Evaluation script (no DB needed)
├── requirements.txt                         # Python dependencies
├── scripts/                                 # Bash scripts for each dataset
├── prompts/                                 # Prompt templates
├── KnowledgeBase/                           # KG utilities
└── outputs/                                 # Results directory

```

## 🆚 Comparison with Original StructGPT

| Feature | Original | Ollama Version |
|---------|----------|----------------|
| API Provider | OpenAI | Ollama (Local) |
| Cost | Pay per token | Free |
| Privacy | Data sent to OpenAI | Data stays local |
| Dependencies | `openai` library | `requests` only |
| Models | GPT-3.5/4 | Gemma, Llama, Phi3, etc. |
| Setup | API key required | Local installation |
| Accuracy | ~70-75% (Spider) | ~27% (Gemma3) |

## 🎓 Citation

Original StructGPT paper:

```bibtex
@InProceedings{Jiang-StructGPT-2022,
  author = {Jinhao Jiang and Kun Zhou and Zican Dong and Keming Ye and Wayne Xin Zhao and Ji-Rong Wen},
  title = {StructGPT: A general framework for Large Language Model to Reason on Structured Data},
  year = {2023},
  journal = {arXiv preprint arXiv:2305.09645},
  url = {https://arxiv.org/pdf/2305.09645}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for more Ollama models
- Improved prompt engineering
- Few-shot learning examples
- Better evaluation metrics
- Performance optimizations

## 📝 License

This project maintains the same license as the original StructGPT repository.
See [LICENSE](./LICENSE) for details.

## 🙏 Acknowledgments

- Original [StructGPT](https://github.com/HKUNLP/StructGPT) by HKUNLP
- [Ollama](https://ollama.com) for local LLM inference
- Spider, TabFact, WikiSQL, and WebQSP dataset creators

## 💡 Tips for Better Results

1. **Try different models**: `phi3:mini` or `llama3.2:8b` may perform better
2. **Adjust temperature**: Currently set to 0 for deterministic outputs
3. **Add few-shot examples**: Modify prompts to include example SQL queries
4. **Fine-tune models**: For production use, consider fine-tuning on your data
5. **Use quantized models**: Faster inference with minimal quality loss

## 🐛 Troubleshooting

**"Connection Error - Is Ollama running?"**
```bash
# Make sure Ollama server is running
ollama serve
```

**"Model not found"**
```bash
# Pull the model first
ollama pull gemma3
```

**Out of Memory**
```bash
# Try a smaller model
ollama pull gemma2:2b
```

## 📧 Contact

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note:** This is an unofficial modification of StructGPT for use with Ollama. For the original implementation, see the [official StructGPT repository](https://github.com/HKUNLP/StructGPT).
