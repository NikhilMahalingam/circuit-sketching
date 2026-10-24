"""Restore only archived historical line endings; never change source logic."""
import argparse
import hashlib
import json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore',action='store_true',help='restore the archived bytes only when current source differs solely by CRLF/LF')
    args=parser.parse_args();root=Path(__file__).resolve().parents[2]
    folder=root/'results/mattr_data_scaling/gate5f'
    manifest=json.loads((folder/'source_byte_manifest.json').read_text())
    protocol=json.loads((folder/'gate5f_protocol.json').read_text())
    for name,record in manifest.items():
        target=root/name;saved=folder/record['archive'];expected=saved.read_bytes();actual=target.read_bytes()
        assert hashlib.sha256(expected).hexdigest()==record['sha256']
        assert record['sha256']==protocol['source_sha256'][name]
        if actual==expected:print('Already compatible:',name);continue
        if actual.replace(b'\r\n',b'\n')!=expected.replace(b'\r\n',b'\n'):
            raise ValueError('Source logic differs; refusing to overwrite: '+name)
        if not args.restore:raise ValueError('Line endings differ. Inspect the archive, then explicitly use --restore: '+name)
        target.write_bytes(expected);print('Restored archived line endings:',name)

if __name__=='__main__':main()
