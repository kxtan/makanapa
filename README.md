# makanapa
Age old question of where to eat

## Enhanced Semantic Search with LLM Integration

This project now includes LLM-powered enhancement of semantic search results using OpenRouter API, providing intelligent analysis and recommendations based on restaurant review data.

### Features

- **Semantic Search**: Vector-based search using ChromaDB and sentence-transformers
- **LLM Enhancement**: AI-powered analysis of search results using OpenRouter models
- **REST API**: FastAPI backend with enhanced search endpoints
- **Fallback System**: Graceful degradation when LLM is unavailable

### Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure OpenRouter API**:
   - Get your API key from [OpenRouter](https://openrouter.ai/)
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your OpenRouter API key:
     ```
     OPENROUTER_API_KEY=your_actual_api_key_here
     OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
     OPENROUTER_MODEL=openai/gpt-3.5-turbo
     ```

3. **Available Models** (modify in `.env`):
   - `openai/gpt-3.5-turbo` (default)
   - `anthropic/claude-3-sonnet`
   - `google/gemini-pro`
   - `mistralai/mistral-7b-instruct`

### API Usage

#### Enhanced Search Endpoint

**POST** `/search`

**Request Body**:
```json
{
  "query": "cozy Italian restaurant",
  "n_results": 5,
  "filters": {"location": "Kuching"}
}
```

**Response**:
```json
{
  "original_results": [...],
  "enhanced_summary": "Based on your search for 'cozy Italian restaurant', I found several excellent options...",
  "llm_model": "openai/gpt-3.5-turbo"
}
```

#### Database Statistics

**GET** `/stats` - Returns database statistics including total chunks, unique locations, and average ratings.

### Running the Application

1. **Start the FastAPI server**:
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

2. **Test the LLM enhancement**:
   ```bash
   python test_llm_enhancement.py
   ```

### Architecture

- **Backend**: FastAPI with ChromaDB for vector storage
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM Integration**: OpenRouter API with direct HTTP calls
- **Fallback**: Simple enhancement when LLM is unavailable

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | Your OpenRouter API key | Required |
| `OPENROUTER_BASE_URL` | OpenRouter API base URL | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | Model to use for enhancement | `openai/gpt-3.5-turbo` |

### Development

- The LLM enhancer is located in `backend/llm_enhancer.py`
- Main API endpoints are in `backend/app.py`
- Test script: `backend/test_llm_enhancement.py`

### Notes

- The system will automatically fall back to simple enhancement if LLM services are unavailable
- Ensure your OpenRouter API key has sufficient credits for the chosen model
- Response times may vary based on the selected LLM model
