import os
import json
import logging
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from typing import Optional, Dict, Any
from pydantic import BaseModel

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @staticmethod
    def format(text, color):
        return f"{color}{text}{Colors.END}"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic models for structured responses
class BeliefAttributionAnswer(BaseModel):
    answer: str

class BeliefUpdateAnswer(BaseModel):
    answer: int


class QwenLLM:
    """Minimal Qwen LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="qwen-max"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required. Please set DASHSCOPE_API_KEY or QWEN_API_KEY in your environment or .env file")

        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using function calling only"""
        try:
            if temperature is None:
                temperature = self.default_temperature
            
            # Define function schema based on task type
            if task_type == "belief_attribution":
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_attribution",
                        "description": "Answer a belief attribution question with a single letter choice",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                    "description": "The letter choice for the answer"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            elif task_type == "belief_update":
                scale_min, scale_max = scale if scale else [1, 10]
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_update",
                        "description": f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "integer",
                                    "minimum": scale_min,
                                    "maximum": scale_max,
                                    "description": f"The numeric answer from {scale_min} to {scale_max}"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] QwenLLM using function calling mode")
            
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                tools=[function_schema],
                tool_choice={"type": "function", "function": {"name": (
                    "answer_belief_attribution" if task_type=="belief_attribution" else "answer_belief_update"
                )}},
                temperature=temperature,
                seed=self.random_seed,
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                args = json.loads(tool_call.function.arguments or "{}")
                if "answer" in args:
                    return {"raw_response": tool_call.function.arguments,
                            "extracted_answer": args["answer"],
                            "method_used": "function_call"}
            
            return {"raw_response": None, "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}


class GeminiLLM:
    """Minimal Gemini LLM client using function calling only"""

    def __init__(self, api_key=None, model="gemini-2.5-flash"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY in your environment or .env file")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using function calling only"""
        try:
            if temperature is None:
                temperature = self.default_temperature
            
            # Configure tools and config for Gemini
            from google.genai import types
            
            # Define function schema based on task type using strong types
            if task_type == "belief_attribution":
                function_declaration = types.FunctionDeclaration(
                    name="answer_belief_attribution",
                    description="Answer a belief attribution question with a single letter choice",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "answer": types.Schema(
                                type=types.Type.STRING,
                                enum=list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                description="The letter choice for the answer"
                            )
                        },
                        required=["answer"]
                    )
                )
            elif task_type == "belief_update":
                scale_min, scale_max = scale if scale else [1, 10]
                function_declaration = types.FunctionDeclaration(
                    name="answer_belief_update",
                    description=f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "answer": types.Schema(
                                type=types.Type.INTEGER,
                                description=f"The numeric answer from {scale_min} to {scale_max}"
                            )
                        },
                        required=["answer"]
                    )
                )
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] GeminiLLM using function calling mode")
            
            tools = types.Tool(function_declarations=[function_declaration])
            
            # Configure function calling mode
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY"
                )
            )
            
            # Create the generation config
            config = types.GenerateContentConfig(
                tools=[tools],
                tool_config=tool_config
            )
            
            # Combine system message and user prompt
            combined_prompt = f"{system_message}\n\n{prompt}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=combined_prompt,
                config=config
            )
            
            # Check for function call in response
            if (response.candidates and 
                response.candidates[0].content.parts and 
                response.candidates[0].content.parts[0].function_call):
                
                function_call = response.candidates[0].content.parts[0].function_call
                if hasattr(function_call, 'args') and 'answer' in function_call.args:
                    return {
                        "raw_response": str(function_call.args),
                        "extracted_answer": function_call.args['answer'],
                        "method_used": "function_call"
                    }
            
            return {"raw_response": None, "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}

        
class LlamaLLM:
    """Minimal Llama LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="meta-llama/llama-3.3-70b-instruct"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not api_key:
            raise ValueError("Llama API key is required. Please set LLAMA_API_KEY in your environment or .env file")
        
        self.client = OpenAI(
            base_url="https://api.novita.ai/openai",
            api_key=api_key,
            max_retries=0,
            timeout=60.0,
        )
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using function calling only"""
        try:
            if temperature is None:
                temperature = self.default_temperature
            
            # Define function schema based on task type
            if task_type == "belief_attribution":
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_attribution",
                        "description": "Answer a belief attribution question with a single letter choice",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                    "description": "The letter choice for the answer"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            elif task_type == "belief_update":
                scale_min, scale_max = scale if scale else [1, 10]
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_update",
                        "description": f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "integer",
                                    "minimum": scale_min,
                                    "maximum": scale_max,
                                    "description": f"The numeric answer from {scale_min} to {scale_max}"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] LlamaLLM using function calling mode")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                tools=[function_schema],
                tool_choice="required",
                temperature=temperature,
                seed=self.random_seed
            )
            
            if response and response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function and tool_call.function.arguments:
                    function_args = json.loads(tool_call.function.arguments)
                    if "answer" in function_args:
                        return {
                            "raw_response": tool_call.function.arguments,
                            "extracted_answer": function_args["answer"],
                            "method_used": "function_call"
                        }
            
            return {"raw_response": None, "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}


class ChatGPTLLM:
    """Minimal ChatGPT LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="gpt-4o"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment or .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using function calling only"""
        try:
            if temperature is None:
                temperature = self.default_temperature
            
            # Define function schema based on task type
            if task_type == "belief_attribution":
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_attribution",
                        "description": "Answer a belief attribution question with a single letter choice",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                    "description": "The letter choice for the answer"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            elif task_type == "belief_update":
                scale_min, scale_max = scale if scale else [1, 10]
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_update",
                        "description": f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "integer",
                                    "minimum": scale_min,
                                    "maximum": scale_max,
                                    "description": f"The numeric answer from {scale_min} to {scale_max}"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] ChatGPTLLM using function calling mode")
            
            # Build request parameters
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                "tools": [function_schema],
                "tool_choice": "required"
            }
            
            # Handle model-specific parameter support
            if self.model == "gpt-4o":
                # gpt-4o supports all parameters
                params["temperature"] = temperature
                params["seed"] = self.random_seed
            elif self.model == "gpt-5-mini":
                # gpt-5-mini: no temperature
                params["seed"] = self.random_seed
            elif self.model == "o3-mini":
                # o3-mini: no temperature
                params["seed"] = self.random_seed
            elif not self.model.startswith(("o1", "o3", "gpt-5")):
                # Legacy models (gpt-3.5, gpt-4, etc.)
                params["temperature"] = temperature
                params["seed"] = self.random_seed
            
            response = self.client.chat.completions.create(**params)
            
            if response and response.choices and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                if tool_call.function and tool_call.function.arguments:
                    function_args = json.loads(tool_call.function.arguments)
                    if "answer" in function_args:
                        return {
                            "raw_response": tool_call.function.arguments,
                            "extracted_answer": function_args["answer"],
                            "method_used": "function_call"
                        }
            
            return {"raw_response": None, "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}


class GPT51LLM:
    """Minimal GPT-5.1 LLM client using responses API with reasoning effort"""

    def __init__(self, api_key=None, model="gpt-5.1"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your environment or .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using Responses API with reasoning effort high"""
        try:
            # Define tool schema for Responses API (flat structure, no nested function)
            if task_type == "belief_attribution":
                tool_name = "answer_belief_attribution"
                tools = [{
                    "type": "function",
                    "name": tool_name,
                    "description": "Answer a belief attribution question with a single letter choice",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "enum": list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                "description": "The letter choice for the answer"
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }]
            elif task_type == "belief_update":
                tool_name = "answer_belief_update"
                scale_min, scale_max = scale if scale else [1, 10]
                tools = [{
                    "type": "function",
                    "name": tool_name,
                    "description": f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "integer",
                                "minimum": scale_min,
                                "maximum": scale_max,
                                "description": f"The numeric answer from {scale_min} to {scale_max}"
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False
                    },
                    "strict": True
                }]
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] GPT51LLM using Responses API with reasoning effort=high")
            
            # Combine system message and prompt
            combined_input = f"{system_message}\n\n{prompt}"
            
            # Use responses.create() API with reasoning effort
            response = self.client.responses.create(
                model=self.model,
                input=combined_input,
                tools=tools,
                tool_choice={"type": "allowed_tools", "mode": "required", "tools": [
                    {"type": "function", "name": tool_name}
                ]},
                reasoning={"effort": "high"}
            )
            
            # Extract function call from response.output (Responses API style)
            if response.output and isinstance(response.output, list):
                for item in response.output:
                    if getattr(item, "type", None) == "function_call" and getattr(item, "name", None) == tool_name:
                        args = json.loads(item.arguments or "{}")
                        if "answer" in args:
                            return {
                                "raw_response": item.arguments,
                                "extracted_answer": args["answer"],
                                "method_used": "function_call"
                            }
            
            return {"raw_response": getattr(response, "output_text", None), "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}


class DeepSeekLLM:
    """Minimal DeepSeek LLM client for generating benchmark questions"""

    def __init__(self, api_key=None, model="deepseek-r1-0528"):
        # Try to load API key from environment files
        parent_env_path = Path(__file__).parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent / ".env.local"
        local_env_path = Path(__file__).parent / ".env"

        if local_env_path.exists():
            load_dotenv(dotenv_path=local_env_path)
        elif parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        # Use QWEN_API_KEY as specified
        api_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DeepSeek API key is required. Please set QWEN_API_KEY or DASHSCOPE_API_KEY in your environment or .env file")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_retries=0,
            timeout=60.0,
        )
        self.model = model
        
        # Load LLM configuration
        self.random_seed = int(os.getenv("LLM_RANDOM_SEED", "42"))
        self.default_temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    def generate_with_fallback(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: Optional[float] = None,
        debug: bool = False,
        max_retries: int = 3,
        task_type: str = "belief_attribution",
        answer_options: Optional[Dict[str, str]] = None,
        scale: Optional[list] = None
    ) -> Dict[str, Any]:
        """Generate structured response using function calling only"""
        try:
            if temperature is None:
                temperature = self.default_temperature
            
            # Define function schema based on task type
            if task_type == "belief_attribution":
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_attribution",
                        "description": "Answer a belief attribution question with a single letter choice",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "enum": list(answer_options.keys()) if answer_options else ["A", "B", "C", "D"],
                                    "description": "The letter choice for the answer"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            elif task_type == "belief_update":
                scale_min, scale_max = scale if scale else [1, 10]
                function_schema = {
                    "type": "function",
                    "function": {
                        "name": "answer_belief_update",
                        "description": f"Answer a belief update question with a number from {scale_min} to {scale_max}",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "integer",
                                    "minimum": scale_min,
                                    "maximum": scale_max,
                                    "description": f"The numeric answer from {scale_min} to {scale_max}"
                                }
                            },
                            "required": ["answer"]
                        }
                    }
                }
            else:
                return {"raw_response": None, "extracted_answer": None, "method_used": "unsupported_task_type"}
            
            if debug:
                print(f"[DEBUG] DeepSeekLLM using function calling mode")
            
            client = OpenAI(
                api_key=self.client.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            response = client.chat.completions.create(
                model="deepseek-r1-0528",
                messages=[{"role":"system","content":system_message},
                          {"role":"user","content":prompt}],
                tools=[function_schema],
                tool_choice={"type":"function","function":{"name":(
                    "answer_belief_attribution" if task_type=="belief_attribution" else "answer_belief_update"
                )}},
                temperature=temperature,
                seed=self.random_seed,
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                args = json.loads(tool_call.function.arguments or "{}")
                if "answer" in args:
                    return {"raw_response": tool_call.function.arguments,
                            "extracted_answer": args["answer"],
                            "method_used": "function_call"}
            
            return {"raw_response": None, "extracted_answer": None, "method_used": "no_function_call"}
            
        except Exception as e:
            if debug:
                print(f"Function call failed: {e}")
            return {"raw_response": None, "extracted_answer": None, "method_used": "error"}

def main():
    """Test the LLM classes with function calling"""
    # Test data
    belief_attribution_test = {
        "prompt": "Based on the conversation, what does the person believe about urban development?",
        "system_message": "You are analyzing beliefs from conversation data.",
        "answer_options": {"A": "Supports development", "B": "Opposes development", "C": "Neutral", "D": "Uncertain"}
    }
    
    belief_update_test = {
        "prompt": "How strongly does this person agree with the statement? Rate from 1-5.",
        "system_message": "You are predicting survey responses.",
        "scale": [1, 5]
    }
    
    # Test function calling models
    function_calling_models = [
        # ("QwenLLM", QwenLLM),
        # ("LlamaLLM", LlamaLLM),
        # ("ChatGPTLLM", ChatGPTLLM),
        ("DeepSeekLLM", DeepSeekLLM),
        # ("GeminiLLM", GeminiLLM),
        # ("GPT51LLM", GPT51LLM)
    ]
    
    for model_name, model_class in function_calling_models:
        print(f"\n🧪 Testing {model_name}...")
        try:
            llm = model_class()
            
            # Test belief attribution
            print(f"📝 Testing belief attribution...")
            result = llm.generate_with_fallback(
                belief_attribution_test["prompt"],
                system_message=belief_attribution_test["system_message"],
                task_type="belief_attribution",
                answer_options=belief_attribution_test["answer_options"],
                debug=True
            )
            print(f"  Method: {result['method_used']}")
            print(f"  Answer: {result['extracted_answer']}")
            print(f"  Raw: {result['raw_response']}")
            
            # Test belief update
            print(f"📊 Testing belief update...")
            result = llm.generate_with_fallback(
                belief_update_test["prompt"],
                system_message=belief_update_test["system_message"],
                task_type="belief_update",
                scale=belief_update_test["scale"],
                debug=True
            )
            print(f"  Method: {result['method_used']}")
            print(f"  Answer: {result['extracted_answer']}")
            print(f"  Raw: {result['raw_response']}")
            
        except Exception as e:
            print(f"❌ {model_name} Error: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    main()