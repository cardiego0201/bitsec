 #!/usr/bin/env python3
"""
Simple web server for file finding functionality.
Provides REST API endpoint for find_similar_files function.
"""

import os
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify
import logging

# Configure logging to use stdout for PM2 compatibility
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Use stdout instead of stderr
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SEARCH_DIR = os.path.join(os.path.dirname(__file__), "samples", "clean-codebases")
FILE_EXT = ".sol"
MIN_SIMILARITY = 0.1
MAX_RESULTS = 1

@dataclass
class FileMatch:
    """Represents a match between a code snippet and a file."""
    file_path: str
    similarity_score: float
    file_size: int
    file_content: str

def normalize_code(code: str) -> str:
    """Normalize code by removing whitespace, tabs, carriage returns, and comment lines."""
    lines = code.splitlines()
    filtered_lines = []
    
    for line in lines:
        stripped_line = line.lstrip()
        if not stripped_line.startswith("//"):
            filtered_lines.append(line)
    
    filtered_code = '\n'.join(filtered_lines)
    return filtered_code.replace("\r", "").replace("\t", "").replace(" ", "")

def jaccard_similarity(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two code strings."""
    a_lines = set(normalize_code(a).splitlines())
    b_lines = set(normalize_code(b).splitlines())
    intersection = a_lines & b_lines
    union = a_lines | b_lines
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _find_similar_files(snippet_content: str) -> List[FileMatch]:
    """Find files similar to a given snippet using Jaccard similarity."""
    search_directory = SEARCH_DIR
    file_extension = FILE_EXT
    min_similarity = MIN_SIMILARITY
    max_results = MAX_RESULTS

    snippet_lines = len(snippet_content.splitlines())
    snippet_words = len(snippet_content.split())
    
    logger.info(f"Searching for snippet with {snippet_words} words, {snippet_lines} lines")
    logger.info(f"Search directory: {search_directory}")
    logger.info(f"File extension: {file_extension}")
    logger.info(f"Minimum similarity: {min_similarity}")
    
    # Pre-normalize snippet content for performance
    normalized_snippet = normalize_code(snippet_content)
    snippet_lines_set = set(normalized_snippet.splitlines())
    
    # Find all files with the specified extension
    search_path = Path(search_directory)
    files = list(search_path.rglob(f"*{file_extension}"))
    
    logger.info(f"Found {len(files)} files to compare")
    
    matches = []
    processed = 0
    start_time = time.time()
    
    printed_file_count = 0
    for file_path in files:
        processed += 1
        
        # Progress update every 3000 files
        if processed % 3000 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(files) - processed) / rate if rate > 0 else 0
            logger.info(f"Processed {processed}/{len(files)} files ({processed/len(files)*100:.1f}%) - "
                  f"Rate: {rate:.1f} files/sec - ETA: {eta:.1f}s")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Skip empty files
            if not file_content.strip():
                continue
            
            # Calculate similarity using pre-normalized snippet
            file_lines_set = set(normalize_code(file_content).splitlines())
            intersection = snippet_lines_set & file_lines_set
            union = snippet_lines_set | file_lines_set
            if not union:
                similarity = 0.0
            else:
                similarity = len(intersection) / len(union)
            
            # Add to matches if above threshold
            if similarity >= min_similarity:
                file_size = len(file_content)
                match = FileMatch(
                    file_path=str(file_path),
                    similarity_score=similarity,
                    file_size=file_size,
                    file_content=file_content,
                )
                matches.append(match)
                
                # Log high similarity matches
                if similarity > 0.9 and printed_file_count < 3:
                    logger.info(f"  High similarity: {similarity:.3f} - {file_path}")
                    printed_file_count += 1
                elif similarity > 0.9 and printed_file_count == 3:
                    logger.info(f"  ... and more")
                    printed_file_count += 1
                    
        except Exception as e:
            logger.info(f"Error reading {file_path}: {e}")
            continue
    
    # Sort by similarity score (highest first) and return top results
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    return matches[:max_results]

def find_similar_files(changed_code: str) -> Optional[FileMatch]:
    try:
        logger.info("=" * 80)
        logger.info("=" * 80)
        logger.info("Received snippet_content")
        logger.info(f"Snippet preview: {changed_code[:100]}...")
        logger.info("Starting to find similar files")
        
        if not changed_code:
            logger.warning("changed_code is required")
            return None
        
        # Record start time
        start_time = time.time()
        
        matches = _find_similar_files(snippet_content=changed_code)
        
        # Calculate processing time
        processing_time = time.time() - start_time

        logger.info(f"Found {len(matches)} matches in {processing_time:.2f} seconds")
        
        
        # Convert matches to dictionaries for JSON serialization
        if not matches:
            logger.info("No similar files found.")
            return None

        best_match = matches[0]
        logger.info(
            "Returning found file: %s (similarity %.3f)",
            best_match.file_path,
            best_match.similarity_score,
        )
        logger.info("=" * 80)
        logger.info("=" * 80)

        return best_match
        
    except Exception as e:
        logger.error(f"Error in find_similar_files: {e}")
        return None