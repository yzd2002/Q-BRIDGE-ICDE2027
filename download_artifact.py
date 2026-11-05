#!/usr/bin/env python3
"""Download, hash-check, and optionally extract the complete Q-BRIDGE artifact."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import urllib.parse
import urllib.request
import zipfile

BASE = 'https://github.com/yzd2002/Q-BRIDGE-ICDE2027/releases/download/artifact-v5.2/'
MANIFEST_SHA256 = 'c0339fad7a81c29540aa38a5ec1fd990630c989f0481322595ff924250131d9d'
ROOT_NAME = 'Q-BRIDGE-ICDE2027'
CHUNK = 4 * 1024 * 1024

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()

def check_name(name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in {'.', '..'}:
        raise ValueError('Unsafe asset name')
    return name

def publish_file(partial, target):
    # Atomic no-overwrite publication on the same filesystem.
    os.link(partial, target)
    partial.unlink()

def read_manifest():
    p = Path(__file__).resolve().with_name('artifact-manifest.json')
    raw = p.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('Artifact manifest differs from this downloader version')
    manifest = json.loads(raw)
    if manifest['format'] != 'qbridge-public-release-v1' or not manifest['parts']:
        raise ValueError('Unrecognized or empty artifact manifest')
    names = [check_name(x['name']) for x in manifest['parts']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate part names')
    check_name(manifest['archive']['name'])
    if sum(x['bytes'] for x in manifest['parts']) != manifest['archive']['bytes']:
        raise ValueError('Part sizes do not sum to the complete archive size')
    return manifest

def matches(path, row):
    if path.is_symlink():
        raise ValueError('Symlink output refused')
    return path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256']

def fetch_part(row, output):
    target = output / check_name(row['name'])
    if target.exists() or target.is_symlink():
        if matches(target, row):
            print('Already verified:', target.name, flush=True)
            return target
        raise ValueError('Existing part does not match: ' + target.name)
    partial = target.with_name(target.name + '.partial')
    if partial.exists() or partial.is_symlink():
        raise ValueError('Previous incomplete download retained: ' + partial.name)
    url = BASE + urllib.parse.quote(row['name'])
    request = urllib.request.Request(url, headers={'User-Agent': 'Q-BRIDGE-public-artifact-downloader'})
    h = hashlib.sha256()
    count = 0
    print('Downloading:', target.name, flush=True)
    with partial.open('xb') as dst, urllib.request.urlopen(request, timeout=90) as src:
        for data in iter(lambda: src.read(CHUNK), b''):
            count += len(data)
            if count > row['bytes']:
                raise ValueError('Download exceeds the declared part size')
            dst.write(data)
            h.update(data)
    if count != row['bytes'] or h.hexdigest() != row['sha256']:
        raise ValueError('Part integrity check failed; partial file retained')
    publish_file(partial, target)
    print('SHA-256 verified:', target.name, flush=True)
    return target

def assemble(manifest, parts, output):
    row = manifest['archive']
    target = output / check_name(row['name'])
    if target.exists() or target.is_symlink():
        if matches(target, row):
            return target
        raise ValueError('Existing complete archive does not match')
    partial = target.with_name(target.name + '.assembling')
    h = hashlib.sha256()
    count = 0
    with partial.open('xb') as dst:
        for path in parts:
            with path.open('rb') as src:
                for data in iter(lambda: src.read(CHUNK), b''):
                    dst.write(data)
                    h.update(data)
                    count += len(data)
    if count != row['bytes'] or h.hexdigest() != row['sha256']:
        raise ValueError('Complete archive integrity check failed; assembly retained')
    publish_file(partial, target)
    return target

def extract(archive, output):
    root = output / ROOT_NAME
    if root.exists() or root.is_symlink():
        raise ValueError('Extraction directory already exists; it will not be overwritten')
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        seen = set()
        for entry in entries:
            path = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            canonical = path.as_posix() + ('/' if entry.is_dir() else '')
            if (path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename
                    or any(ord(c) < 32 or ord(c) == 127 for c in entry.filename)
                    or canonical != entry.filename
                    or not path.parts or path.parts[0] != ROOT_NAME
                    or entry.filename in seen or stat.S_ISLNK(mode)
                    or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)))):
                raise ValueError('Unsafe or repeated ZIP entry')
            seen.add(entry.filename)
        root.mkdir()
        for entry in entries:
            target = output.joinpath(*PurePosixPath(entry.filename).parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(entry) as src, target.open('xb') as dst:
                for data in iter(lambda: src.read(CHUNK), b''):
                    dst.write(data)
            target.chmod((entry.external_attr >> 16) & 0o777 or 0o644)
    return root

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('downloaded-artifact'))
    parser.add_argument('--extract', action='store_true', help='Extract only after all hashes pass')
    args = parser.parse_args()
    if args.output.is_symlink() or any(p.is_symlink() for p in args.output.absolute().parents):
        raise ValueError('Symlink destination refused')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    parts = [fetch_part(row, args.output) for row in manifest['parts']]
    archive = assemble(manifest, parts, args.output)
    print('Complete archive SHA-256 verified:', archive, flush=True)
    if args.extract:
        root = extract(archive, args.output)
        print('Extracted:', root, flush=True)
        print('Next: python3 -B -S "' + str(root / 'verify_artifact.py') + '"', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Download stopped (' + type(error).__name__ + '). Existing files were retained.', file=sys.stderr)
        if isinstance(error, (ValueError, FileNotFoundError)):
            print(str(error), file=sys.stderr)
        raise SystemExit(1)
