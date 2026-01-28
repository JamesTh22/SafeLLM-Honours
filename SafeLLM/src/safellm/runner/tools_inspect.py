from __future__ import annotations  
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional
from inspect_ai.tool import tool, Tool

from inspect_ai.util import sandbox

@tool
def read_file() -> Tool:
    """
    Read the contents of a file at the specified path.
    Args:
        path: Path to the file.
        offset: Character offset to start reading from (default 0).
        limit: Maximum characters to read (default 5000).
    """
    async def excute(path: str, offset: int = 0, limit: int = 5000) -> str:
        content = await sandbox().read_file(path)
        
        # Apply pagination
        if offset > 0 or len(content) > limit:
            chunk = content[offset : offset + limit]
            total_len = len(content)
            remaining = total_len - (offset + limit)
            
            msg = f"\n... [Showing chars {offset}-{offset+len(chunk)} of {total_len}]"
            if remaining > 0:
                msg += f"\n... [TRUNCATED. Use offset={offset+limit} to see more]"
            
            return chunk + msg
            
        return content
    return excute

@tool # co-author: co-pilot
def list_files() -> Tool:
    """
    List all files in the directory at the specified path.
    """
    async def excute(path: str) -> List[str]:
        # sandbox().ls returns a list of FileInfo objects usually, or we use exec?
        # Standard inspect_ai sandbox API doesn't have ls()? 
        # Actually it does `exec(['ls', path])` or `read_file`?
        result = await sandbox().exec(["ls", path])
        if result.returncode != 0:
             return f"Error listing files: {result.stderr}"
        return result.stdout.splitlines()
    return excute

@tool #co-author: co-pilot
def write_file() -> Tool:
    """
    Write text content to a file at the specified path.
    """
    async def excute(path: str, content: str) -> str:
        await sandbox().write_file(path, content)
        return "File written successfully."
    return excute

@tool  # co-author: co-pilot
def make_dir() -> Tool:
    """
    Create a new directory at the specified path.
    """
    async def excute(path: str) -> str:
        await sandbox().exec(["mkdir", "-p", path])
        return "Directory created successfully."
    return excute

@tool  # co-author: co-pilot
def append_file() -> Tool:
    """
    Append text content to the end of an existing file.
    """
    async def excute(path: str, content: str) -> str:
        # Sandbox doesn't have direct append. Read + Write or shell append.
        # Shell append is safer for race conditions but read+write is simpler for now.
        # Or use exec echo >> ?
        # Let's use read + write as it handles encoding better in python
        try:
            current = await sandbox().read_file(path)
        except Exception:
            current = "" # File might not exist
        
        await sandbox().write_file(path, current + content)
        return "Content appended successfully."
    return excute

@tool  # co-author: co-pilot
def write_csv() -> Tool:
    """
    Write a list of dictionaries to a CSV file.
    """
    async def excute(path: str, rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return "No data provided to write."
        
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        
        await sandbox().write_file(path, output.getvalue())
        return "CSV file written successfully."
    return excute

@tool  # co-author: co-pilot
def read_csv() -> Tool:
    """
    Read content from a CSV file and return it as a list of dictionaries.
    Args:
         path: Path to the CSV file.
         limit: Maximum rows to return (default 50).
    """
    async def excute(path: str, limit: int = 50) -> List[Dict[str, Any]]:
        content = await sandbox().read_file(path)
        
        import io
        file = io.StringIO(content)
        reader = csv.DictReader(file)
        rows = [row for row in reader]
        
        if len(rows) > limit:
             return rows[:limit] + [{"...": f"[TRUNCATED. Showing {limit} of {len(rows)} rows. Use python slicing or read chunks logic if implemented]"}]
        return rows
    return excute

@tool # co-author: co-pilot
def done() -> Tool:
    """
    Signal that the task is complete.
    """
    async def excute() -> str:
        return "All tasks completed."
    return excute