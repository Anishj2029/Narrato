import json
from pathlib import Path

data = json.loads(Path('transcript_segments.json').read_text(encoding='utf-8'))
segs = data['segments']
print(f'Total segments: {len(segs)}')
print(f'Total words:    {data["total_words"]}')
print(f'Duration:       {data["audio_duration_seconds"]}s')
print()
print('All segments (pause >= 0.7s flagged as potential boundary):')
for i, seg in enumerate(segs):
    gap_str = ''
    if i > 0:
        pause = round(seg['start'] - segs[i-1]['end'], 3)
        if pause >= 0.7:
            gap_str = f'  <<<<< PAUSE {pause:.2f}s'
        elif pause >= 0.4:
            gap_str = f'  < gap {pause:.2f}s'
    wc = len(seg['text'].split())
    print(f'  [{seg["start"]:7.2f}-{seg["end"]:7.2f}] wc={wc:3d}  {seg["text"][:72]}{gap_str}')
