import pytest
import os
from pathlib import Path
from sms_agent.inventory import InventoryCollector

def test_inventory_empty_directory(tmp_path):
    collector = InventoryCollector(str(tmp_path))
    result = collector.collect()
    assert len(result) == 0

def test_inventory_one_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    
    collector = InventoryCollector(str(tmp_path))
    result = collector.collect()
    
    assert len(result) == 1
    assert result[0].key == "hello.txt"
    assert result[0].filename == "hello.txt"
    assert result[0].extension == ".txt"
    assert result[0].size_bytes == 11
    assert result[0].content_hash is not None

def test_inventory_deterministic_hash(tmp_path):
    f1 = tmp_path / "f1.bin"
    f1.write_bytes(b"data1")
    f2 = tmp_path / "f2.bin"
    f2.write_bytes(b"data1")
    f3 = tmp_path / "f3.bin"
    f3.write_bytes(b"data2")
    
    collector = InventoryCollector(str(tmp_path))
    result = collector.collect()
    
    # Sort by key to be deterministic in test
    result.sort(key=lambda x: x.key)
    
    assert len(result) == 3
    # Two identical files produce same hash
    assert result[0].content_hash == result[1].content_hash
    # Two different files produce different hashes
    assert result[0].content_hash != result[2].content_hash
    
    # Mark duplicates properly
    assert result[0].is_duplicate is True
    assert result[1].is_duplicate is True
    assert result[2].is_duplicate is False

def test_inventory_nested_directories(tmp_path):
    sub = tmp_path / "folder" / "subfolder"
    sub.mkdir(parents=True)
    f = sub / "nested.log"
    f.write_text("nested")
    
    collector = InventoryCollector(str(tmp_path))
    result = collector.collect()
    
    assert len(result) == 1
    assert result[0].key == "folder/subfolder/nested.log"
    assert result[0].filename == "nested.log"
    assert result[0].extension == ".log"

def test_inventory_missing_non_file_does_not_crash(tmp_path):
    # Create a directory
    sub = tmp_path / "empty_dir"
    sub.mkdir()
    
    collector = InventoryCollector(str(tmp_path))
    result = collector.collect()
    
    # The directory itself shouldn't be yielded as a file, and shouldn't crash
    assert len(result) == 0

def test_inventory_invalid_root():
    collector = InventoryCollector("/path/does/not/exist/ever")
    result = collector.collect()
    assert result == []
