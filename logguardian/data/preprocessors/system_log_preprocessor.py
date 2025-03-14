"""
Preprocessor for system logs.
"""
import re
from typing import Dict, Any, Optional, List, Pattern

from loguru import logger

from logguardian.data.preprocessors.base_preprocessor import BaseLogPreprocessor


class SystemLogPreprocessor(BaseLogPreprocessor):
    """
    Preprocessor for system and server logs.
    
    This preprocessor handles common system log formats and masks variable
    parts like timestamps, IP addresses, file paths, etc. with constant tokens.
    """
    
    # Common regex patterns for variable parts in system logs
    DEFAULT_PATTERNS = {
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "timestamp": r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}[T ]\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b",
        "file_path": r"\b(?:/[\w.-]+)+\b",
        "windows_path": r"\b(?:[A-Za-z]:\\[\w\\.-]+)\b",
        "number": r"\b\d+\b",
        "hex": r"\b0x[0-9a-fA-F]+\b",
        "uuid": r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        "url": r"\bhttps?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?::\d+)?(?:/[\w/.-]*)*(?:\?[\w=&.-]*)?(?:#[\w-]*)?\b",
    }
    
    # Replacement tokens for each pattern
    DEFAULT_TOKENS = {
        "ip_address": "<IP>",
        "timestamp": "<TIMESTAMP>",
        "file_path": "<PATH>",
        "windows_path": "<PATH>",
        "number": "<NUM>",
        "hex": "<HEX>",
        "uuid": "<UUID>",
        "url": "<URL>",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the system log preprocessor.
        
        Args:
            config: Optional configuration dictionary with the following keys:
                - patterns: Dict mapping pattern names to regex patterns
                - tokens: Dict mapping pattern names to replacement tokens
                - case_sensitive: Whether to preserve case in log messages
                - remove_punctuation: Whether to remove punctuation
        """
        super().__init__(config)
        
        # Get patterns and tokens from config or use defaults
        self.patterns = self.config.get("patterns", self.DEFAULT_PATTERNS)
        self.tokens = self.config.get("tokens", self.DEFAULT_TOKENS)
        self.case_sensitive = self.config.get("case_sensitive", False)
        self.remove_punctuation = self.config.get("remove_punctuation", False)
        
        # Compile regex patterns for better performance
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.patterns.items()
        }
        
        logger.debug(f"Initialized SystemLogPreprocessor with {len(self.patterns)} patterns")
    
    def preprocess(self, log_message: str) -> str:
        """
        Preprocess a single log message.
        
        Args:
            log_message: The raw log message to preprocess
            
        Returns:
            The preprocessed log message with variables masked
        """
        # Convert case if needed
        if not self.case_sensitive:
            log_message = log_message.lower()
        
        # Apply all regex replacements
        for name, pattern in self.compiled_patterns.items():
            token = self.tokens.get(name)
            if token:
                log_message = pattern.sub(token, log_message)
        
        # Remove punctuation if configured
        if self.remove_punctuation:
            log_message = re.sub(r'[^\w\s]', ' ', log_message)
        
        # Remove extra whitespace
        log_message = re.sub(r'\s+', ' ', log_message).strip()
        
        return log_message
    
    def fit(self, log_messages: List[str]) -> 'SystemLogPreprocessor':
        """
        Fit the preprocessor on a dataset of log messages.
        
        This implementation currently doesn't learn anything from the dataset,
        but could be extended to automatically discover common patterns.
        
        Args:
            log_messages: List of log messages to fit on
            
        Returns:
            self
        """
        logger.info(f"Fitting SystemLogPreprocessor on {len(log_messages)} messages")
        return self
    
    def add_pattern(self, name: str, pattern: str, token: str) -> None:
        """
        Add a new pattern for variable masking.
        
        Args:
            name: Unique name for the pattern
            pattern: Regex pattern string
            token: Replacement token
        """
        self.patterns[name] = pattern
        self.tokens[name] = token
        self.compiled_patterns[name] = re.compile(pattern)
        logger.debug(f"Added new pattern: {name}")
    
    def remove_pattern(self, name: str) -> bool:
        """
        Remove a pattern by name.
        
        Args:
            name: Name of the pattern to remove
            
        Returns:
            True if pattern was removed, False if not found
        """
        if name in self.patterns:
            del self.patterns[name]
            del self.tokens[name]
            del self.compiled_patterns[name]
            logger.debug(f"Removed pattern: {name}")
            return True
        return False