"""Attempt to extract readable text from the C2PA manifest embedded in the MP3."""
from mutagen.id3 import ID3
import re

tags = ID3('narration.mp3')
geob = tags.get('GEOB:c2pa manifest store')
if not geob:
    print("No C2PA tag found.")
else:
    raw = geob.data
    print(f"C2PA blob size: {len(raw)} bytes")

    # Try to find readable JSON or text strings in the binary blob
    # C2PA manifests often contain JSON assertions
    text = raw.decode('utf-8', errors='replace')
    
    # Look for JSON-like content
    json_matches = re.findall(r'\{[^{}]{20,}\}', text)
    print(f"\nJSON-like fragments found: {len(json_matches)}")
    for i, m in enumerate(json_matches[:10]):
        clean = ''.join(c if ord(c) >= 32 else ' ' for c in m)
        print(f"  [{i}] {clean[:200]}")

    # Look for any long readable strings (potential text content)
    readable = re.findall(r'[A-Za-z][A-Za-z0-9 ,.\'\-!?:;]{40,}', text)
    print(f"\nLong readable strings: {len(readable)}")
    for i, r in enumerate(readable[:20]):
        print(f"  [{i}] {r[:150]}")
