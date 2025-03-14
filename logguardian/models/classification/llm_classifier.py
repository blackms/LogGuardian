"""
LLM-based classifier for log sequence anomaly detection.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig
)
from loguru import logger

from logguardian.models.classification.base_classifier import BaseLogClassifier


class LlamaLogClassifier(BaseLogClassifier):
    """
    Log sequence classifier using Llama for anomaly detection.
    
    This classifier uses a transformer decoder model (Llama) to classify
    sequences of log messages as normal or anomalous. It can work with
    either raw log messages or with pre-computed embeddings.
    """
    
    # Default text templates for prompts
    DEFAULT_SYSTEM_PROMPT = """You are a log analysis expert responsible for detecting anomalies in system logs.
Your task is to analyze the provided log sequence and determine if it contains anomalous patterns.
Respond with only "normal" or "anomalous" based on your analysis."""

    DEFAULT_PROMPT_TEMPLATE = """System: {system_prompt}

Below is a sequence of system log messages:
{log_sequence}

Is this sequence normal or anomalous?"""

    DEFAULT_LABELS = {
        "normal": 0,
        "anomalous": 1
    }
    
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3-8b",
        tokenizer=None,
        model=None,
        system_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        labels: Optional[Dict[str, int]] = None,
        max_length: int = 2048,
        generation_config: Optional[Dict[str, Any]] = None,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        device: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = "auto",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the LLM classifier.
        
        Args:
            model_name_or_path: Name or path of the pre-trained LLM
            tokenizer: Optional pre-loaded tokenizer
            model: Optional pre-loaded model
            system_prompt: System prompt for the LLM
            prompt_template: Template for formatting log sequences
            labels: Dictionary mapping text labels to numeric labels
            max_length: Maximum sequence length for tokenization
            generation_config: Configuration for text generation
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            device: Device to run the model on. If None, use CUDA if available
            device_map: Device map for model parallelism
            config: Optional configuration dictionary
        """
        super().__init__(config)
        
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.labels = labels or self.DEFAULT_LABELS
        
        # Reverse mapping from numeric to text labels
        self.id2label = {v: k for k, v in self.labels.items()}
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer if not provided
        if tokenizer is None or model is None:
            self._load_model_and_tokenizer(
                model_name_or_path=model_name_or_path,
                tokenizer=tokenizer,
                model=model,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                device_map=device_map
            )
        else:
            self.tokenizer = tokenizer
            self.model = model
        
        # Set up generation config
        self._setup_generation_config(generation_config)
        
        logger.info(f"Initialized LlamaLogClassifier with model {model_name_or_path}")
    
    def _load_model_and_tokenizer(
        self,
        model_name_or_path: str,
        tokenizer=None,
        model=None,
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = "auto"
    ):
        """
        Load the pre-trained LLM and tokenizer.
        
        Args:
            model_name_or_path: Name or path of the pre-trained LLM
            tokenizer: Optional pre-loaded tokenizer
            model: Optional pre-loaded model
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            device_map: Device map for model parallelism
        """
        logger.info(f"Loading LLM from {model_name_or_path}")
        
        # Load tokenizer
        if tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    use_fast=True
                )
                logger.debug(f"Loaded tokenizer: {model_name_or_path}")
            except Exception as e:
                logger.error(f"Error loading tokenizer: {e}")
                raise
        else:
            self.tokenizer = tokenizer
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model is None:
            try:
                # Set up quantization config if needed
                quantization_config = None
                if load_in_8bit or load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=load_in_8bit,
                        load_in_4bit=load_in_4bit,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                
                # Load the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    trust_remote_code=True
                )
                logger.debug(f"Loaded model: {model_name_or_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise
        else:
            self.model = model
    
    def _setup_generation_config(self, generation_config: Optional[Dict[str, Any]] = None):
        """
        Set up configuration for text generation.
        
        Args:
            generation_config: Configuration for text generation
        """
        # Default generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            num_return_sequences=1,
            do_sample=False,  # Deterministic for classification
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Update with provided config
        if generation_config:
            self.generation_config = GenerationConfig(**generation_config)
            
        logger.debug(f"Generation config: {self.generation_config}")
    
    def _prepare_prompt(self, log_sequence: Union[List[str], str]) -> str:
        """
        Prepare the prompt for the LLM.
        
        Args:
            log_sequence: Sequence of log messages or a string
            
        Returns:
            Formatted prompt string
        """
        # Convert list of logs to a string if needed
        if isinstance(log_sequence, list):
            log_sequence = "\n".join(log_sequence)
        
        # Format the prompt
        prompt = self.prompt_template.format(
            system_prompt=self.system_prompt,
            log_sequence=log_sequence
        )
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM's response to extract the classification.
        
        Args:
            response_text: Text response from the LLM
            
        Returns:
            Dictionary with parsed results including:
                - label: Text label (normal or anomalous)
                - label_id: Numeric label (0 or 1)
                - confidence: Confidence score (estimated)
                - raw_text: Raw response text
        """
        # Normalize and clean the response
        response_text = response_text.strip().lower()
        
        # Initialize results
        results = {
            "label": None,
            "label_id": None,
            "confidence": 0.0,
            "raw_text": response_text
        }
        
        # Check for each possible label
        for label, label_id in self.labels.items():
            if label.lower() in response_text:
                results["label"] = label
                results["label_id"] = label_id
                
                # Simple heuristic for confidence
                # Higher confidence if the response contains only the label word
                if response_text == label.lower():
                    results["confidence"] = 1.0
                else:
                    results["confidence"] = 0.8
                
                break
        
        # If no label was found, assign the most likely one
        if results["label"] is None:
            # Simple heuristic: check which label has more characters in common
            normal_label = list(self.labels.keys())[0]
            anomaly_label = list(self.labels.keys())[1]
            
            normal_count = sum(c in response_text for c in normal_label)
            anomaly_count = sum(c in response_text for c in anomaly_label)
            
            if normal_count >= anomaly_count:
                results["label"] = normal_label
                results["label_id"] = self.labels[normal_label]
                results["confidence"] = 0.5
            else:
                results["label"] = anomaly_label
                results["label_id"] = self.labels[anomaly_label]
                results["confidence"] = 0.5
        
        return results
    
    def classify(
        self, 
        log_sequence: Union[List[str], str, List[torch.Tensor], torch.Tensor],
        raw_output: bool = False
    ) -> Union[int, Dict[str, Any]]:
        """
        Classify a single log sequence.
        
        Args:
            log_sequence: Sequence of log messages or embeddings
            raw_output: Whether to return raw model outputs
            
        Returns:
            Classification result (1 for anomaly, 0 for normal)
            If raw_output is True, returns a dictionary with detailed outputs
        """
        # For embeddings, we need to integrate with the tokenizer
        # This will need custom implementation based on how embeddings are used
        if isinstance(log_sequence, torch.Tensor) or (
            isinstance(log_sequence, list) and isinstance(log_sequence[0], torch.Tensor)
        ):
            logger.warning("Embedding input not fully implemented yet. Converting to placeholder.")
            log_sequence = ["<EMBEDDING_PLACEHOLDER>" for _ in range(10)]
        
        # Prepare the prompt
        prompt = self._prepare_prompt(log_sequence)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=self.generation_config
            )
        
        # Decode the response
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse the response
        results = self._parse_response(response_text)
        
        # Return raw results if requested
        if raw_output:
            return results
        
        # Otherwise, return just the label ID
        return results["label_id"]
    
    def classify_batch(
        self, 
        log_sequences: Union[List[List[str]], List[str], List[List[torch.Tensor]], List[torch.Tensor]],
        raw_output: bool = False,
        batch_size: int = 8
    ) -> Union[List[int], List[Dict[str, Any]]]:
        """
        Classify a batch of log sequences.
        
        Args:
            log_sequences: Batch of log sequences
            raw_output: Whether to return raw model outputs
            batch_size: Size of batches for processing
            
        Returns:
            List of classification results
        """
        # Process in smaller batches to avoid OOM
        results = []
        
        for i in range(0, len(log_sequences), batch_size):
            batch = log_sequences[i:i+batch_size]
            batch_results = [self.classify(seq, raw_output=raw_output) for seq in batch]
            results.extend(batch_results)
        
        return results
    
    def add_lora_adapters(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None
    ) -> None:
        """
        Add LoRA adapters to the model for fine-tuning.
        
        Args:
            lora_rank: Rank of the LoRA adaptation matrices
            lora_alpha: Alpha parameter for LoRA
            lora_dropout: Dropout probability for LoRA
            target_modules: List of module names to apply LoRA to
        """
        # Default target modules for Llama models
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        # Prepare model for k-bit training if using quantization
        if hasattr(self.model, "is_quantized") and self.model.is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info(f"Added LoRA adapters to the model with rank {lora_rank}")
    
    def save(self, path: str) -> None:
        """
        Save the classifier to a directory.
        
        Args:
            path: Path to save the classifier to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(os.path.join(path, "model"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        
        # Save configuration
        config_dict = {
            "model_name_or_path": self.model_name_or_path,
            "max_length": self.max_length,
            "system_prompt": self.system_prompt,
            "prompt_template": self.prompt_template,
            "labels": self.labels,
            "generation_config": self.generation_config.to_dict()
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved LlamaLogClassifier to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str, 
        device: Optional[str] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = "auto",
        **kwargs
    ) -> 'LlamaLogClassifier':
        """
        Load a classifier from a directory.
        
        Args:
            path: Path to load the classifier from
            device: Device to load the model onto
            device_map: Device map for model parallelism
            
        Returns:
            Loaded classifier
        """
        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(path, "tokenizer"),
            use_fast=True
        )
        
        # Load model
        try:
            is_peft_model = os.path.exists(os.path.join(path, "model", "adapter_config.json"))
            
            if is_peft_model:
                # Load PEFT model
                peft_config = PeftConfig.from_pretrained(os.path.join(path, "model"))
                
                # First load the base model
                model = AutoModelForCausalLM.from_pretrained(
                    peft_config.base_model_name_or_path,
                    device_map=device_map,
                    trust_remote_code=True,
                    **kwargs
                )
                
                # Then load the adapters
                model = PeftModel.from_pretrained(
                    model,
                    os.path.join(path, "model"),
                    device_map=device_map
                )
            else:
                # Load regular model
                model = AutoModelForCausalLM.from_pretrained(
                    os.path.join(path, "model"),
                    device_map=device_map,
                    trust_remote_code=True,
                    **kwargs
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Generation config
        generation_config = GenerationConfig(**config_dict.get("generation_config", {}))
        
        # Create classifier
        classifier = cls(
            model_name_or_path=config_dict.get("model_name_or_path", ""),
            tokenizer=tokenizer,
            model=model,
            system_prompt=config_dict.get("system_prompt"),
            prompt_template=config_dict.get("prompt_template"),
            labels=config_dict.get("labels"),
            max_length=config_dict.get("max_length", 2048),
            generation_config=generation_config.to_dict(),
            device=device,
            device_map=None  # Already set during model loading
        )
        
        logger.info(f"Loaded LlamaLogClassifier from {path}")
        
        return classifier