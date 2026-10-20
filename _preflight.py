import json, os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv('.env')

keys = ['GEMINI_API_KEY','IMAGE_BACKEND','STABILITY_API_KEY','REPLICATE_API_TOKEN']
for k in keys:
    v = os.getenv(k, '')
    placeholders = {'your_stability_api_key_here','your_replicate_token_here',
                    'your_gemini_api_key_here',''}
    if v and v not in placeholders:
        print(f'{k}: SET ({len(v)} chars)')
    else:
        print(f'{k}: NOT SET or placeholder')

plan = json.loads(Path('scene_plan.json').read_text(encoding='utf-8'))
print()
print(f'scene_plan.json: {len(plan)} scenes')
print(f'Fields: {list(plan[0].keys())}')
prompt_snippet = plan[0]["image_prompt"][:80]
path_val = repr(plan[0]["image_path"])
fname = plan[0]["filename_suggestion"]
print(f'image_prompt sample: {prompt_snippet}')
print(f'image_path value:    {path_val}')
print(f'filename_suggestion: {fname}')
print()
# Check how many scenes have real (non-fallback) prompts
fallback = sum(1 for s in plan if s["image_prompt"].startswith("[FALLBACK]"))
empty = sum(1 for s in plan if not s["image_prompt"])
print(f'Scenes with [FALLBACK] prompt: {fallback}')
print(f'Scenes with empty prompt:      {empty}')
print(f'Scenes with real prompt:       {len(plan) - fallback - empty}')
