# src/generation/llm.py
import logging
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)

class LLM:
    """Interface for language models."""
    
    def __init__(self, config: dict):
        self.config = config
        self.provider = config['llm']['provider']
        self.model = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens = config['llm']['max_tokens']
        
        self.client = None
        self.has_api_key = False
        self.has_model = False
        
        # Initialize
        self._initialize()
        
    def _initialize(self):
        """Initialize the appropriate provider."""
        if self.provider == "groq":
            self._init_groq()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "local":
            self._init_local()
        else:
            logger.warning(f"Unsupported provider: {self.provider}")
    
    def _init_groq(self):
        """Initialize Groq client."""
        try:
            from groq import Groq
            api_key = os.getenv('GROQ_API_KEY')
            
            if not api_key:
                logger.warning("GROQ_API_KEY not found in .env file")
                logger.info("Please add: GROQ_API_KEY=your-key-here")
                return
            
            self.client = Groq(api_key=api_key)
            self.has_api_key = True
            logger.info(f"✅ Groq client initialized with model: {self.model}")
            
            # Test the connection
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                logger.info("✅ Groq API connection successful!")
            except Exception as e:
                logger.warning(f"Groq API test failed: {e}")
                
        except ImportError:
            logger.error("groq not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return
            self.client = OpenAI(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return
            self.client = Anthropic(api_key=api_key)
            self.has_api_key = True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
    
    def _init_huggingface(self):
        """Initialize Hugging Face client."""
        try:
            from huggingface_hub import InferenceClient
            token = os.getenv('HUGGINGFACE_TOKEN')
            if not token:
                return
            self.client = InferenceClient(model=self.model, token=token)
            self.has_api_key = True
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face: {e}")
    
    def _init_local(self):
        """Initialize local model."""
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                'text-generation',
                model=self.model,
                max_length=self.max_tokens,
                device=-1
            )
            self.has_model = True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
    
    def generate(self, 
                 prompt: str, 
                 system_prompt: Optional[str] = None,
                 **kwargs) -> str:
        """Generate response."""
        if self.provider == "groq" and self.has_api_key:
            return self._generate_groq(prompt, system_prompt, **kwargs)
        elif self.provider == "openai" and self.has_api_key:
            return self._generate_openai(prompt, system_prompt, **kwargs)
        elif self.provider == "anthropic" and self.has_api_key:
            return self._generate_anthropic(prompt, system_prompt, **kwargs)
        elif self.provider == "huggingface" and self.has_api_key:
            return self._generate_huggingface(prompt, system_prompt, **kwargs)
        elif self.provider == "local" and self.has_model:
            return self._generate_local(prompt, **kwargs)
        else:
            return self._get_fallback_message()
    
    def _generate_groq(self, prompt, system_prompt=None, **kwargs):
        """Generate using Groq."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            if "rate_limit" in str(e).lower():
                return "⚠️ **Rate Limit Exceeded**\n\nPlease wait a moment and try again. Groq has rate limits for free tier."
            return self._get_fallback_message()
    
    def _generate_openai(self, prompt, system_prompt=None, **kwargs):
        """Generate using OpenAI."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._get_fallback_message()
    
    def _generate_anthropic(self, prompt, system_prompt=None, **kwargs):
        """Generate using Anthropic."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return self._get_fallback_message()
    
    def _generate_huggingface(self, prompt, system_prompt=None, **kwargs):
        """Generate using Hugging Face."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            response = self.client.text_generation(
                full_prompt,
                max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature)
            )
            return response
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            return self._get_fallback_message()
    
    def _generate_local(self, prompt, **kwargs):
        """Generate using local model."""
        try:
            response = self.pipeline(
                prompt[:500],
                max_length=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                do_sample=True,
                pad_token_id=50256
            )
            return response[0]['generated_text']
        except Exception as e:
            logger.error(f"Local model error: {e}")
            return self._get_fallback_message()
    
    def _get_fallback_message(self) -> str:
        """Get fallback message."""
        return """
⚠️ **LLM Service Unavailable**

Please check your configuration:

**For Groq (Recommended - Free & Fast):**
1. Get API key: https://console.groq.com/
2. Add to .env: GROQ_API_KEY=your-key
3. Update config: provider: "groq"

**Available Groq Models:**
- mixtral-8x7b-32768 (Mixtral)
- llama3-70b-8192 (Llama 3 70B)
- llama3-8b-8192 (Llama 3 8B)
- gemma2-9b-it (Gemma 2 9B)

The document retrieval system is still working perfectly!"""
    
    def build_rag_prompt(self, query: str, context: str, citations: List[Dict[str, Any]]) -> str:
        """Build RAG prompt."""
        if not context:
            return f"Question: {query}\n\nNo relevant context found."
        
        citation_text = "\n".join([
            f"[{i+1}] {c.get('document_title', 'Unknown')} (Page: {c.get('page_start', 'N/A')})"
            for i, c in enumerate(citations) if c
        ])
        
        return f"""You are an AI research assistant. Answer the question based ONLY on the provided context.

Question: {query}

Context:
{context}

Available Sources:
{citation_text}

Instructions:
1. Answer based only on the provided context
2. Include citations using the format [1], [2], etc.
3. If the context doesn't contain the answer, say "I cannot find this information in the provided documents"
4. Be clear and concise

Answer:"""