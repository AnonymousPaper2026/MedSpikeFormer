from setuptools import setup
import os

def parse_requirements(file_path):
    requirements = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path}")
    
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line or line.startswith(('#', 'conda')):
                continue
            if line.startswith('pip install '):
                pkg_line = line[len('pip install '):].strip()
                if '--extra-index-url' in pkg_line:
                    pkg_part = [p for p in pkg_line.split() if not p.startswith('--')]
                    requirements.extend(pkg_part)
                else:
                    requirements.append(pkg_line)
    return requirements

req_c = parse_requirements('utils/requirement_c')
req_e = parse_requirements('utils/requirement_e')

all_requirements = []
seen = set()
for req in req_c + req_e:
    if req not in seen:
        seen.add(req)
        all_requirements.append(req)

setup(
    name='Net',
    version='0.1.0',
    description='Project using requirements from utils/requirement_c and utils/requirement_e',
    install_requires=all_requirements,
    python_requires='>=3.8',
    packages=[], 
)
