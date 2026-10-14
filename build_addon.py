#!/usr/bin/env python3
import os
import sys
import zipfile

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, 'blender_addon', 'vid_anim')
    out_dir = os.path.join(here, 'dist')
    os.makedirs(out_dir, exist_ok=True)
    zp = os.path.join(out_dir, 'Vid_Anim.zip')
    if not os.path.isdir(src):
        print(f'ERROR: {src} not found')
        sys.exit(1)
    if os.path.exists(zp):
        os.remove(zp)
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.join('vid_anim', os.path.relpath(full, src))
                z.write(full, arc)
    # verify
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        if not all(n.startswith('vid_anim/') for n in names):
            print('ERROR: bad structure')
            sys.exit(1)
        if 'vid_anim/__init__.py' not in names:
            print('ERROR: missing __init__.py')
            sys.exit(1)
    print(f'Built: {zp}')

if __name__ == '__main__':
    main()
