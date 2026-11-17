"""Publish report/Canvas ZIP as review assets using existing GitHub sign-in.

Credentials are read in memory from Git's configured credential helper. They
are never written to the repository, included in URLs or printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'denishatank12/Lab-2'
TAG = 'lab2-submission-review'
API = 'https://api.github.com/repos/' + REPOSITORY


def credential():
    result = subprocess.run(['git', 'credential', 'fill'], cwd=ROOT,
        input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True,
        env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'}, timeout=20)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode or not fields.get('password'):
        raise RuntimeError('Existing GitHub sign-in was unavailable. No credential was printed or saved.')
    return fields['password']


def request(url, method='GET', token=None, payload=None, raw=None, content_type=None):
    if urllib.parse.urlsplit(url).netloc not in {'api.github.com', 'uploads.github.com'}:
        raise ValueError('Unexpected authenticated destination')
    headers = {'User-Agent': 'Lab2-submission-publisher', 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    data = raw
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    elif content_type:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)


def publish():
    files = [ROOT / 'output/pdf/Lab2_Team09_Report.pdf', ROOT / 'output/Pair_Programming_Team_09.zip']
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError('Build the report and Canvas ZIP before publishing: ' + path.name)
    with zipfile.ZipFile(files[1]) as bundle:
        if bundle.testzip() or set(bundle.namelist()) != {
                'Lab2_Team09_Report.pdf', 'CONTRIBUTIONS.md', 'AI_USAGE.md', 'links.txt'}:
            raise ValueError('Unexpected Canvas ZIP contents')
        if bundle.read(files[0].name) != files[0].read_bytes():
            raise ValueError('Report and ZIP disagree; rebuild them together')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if dirty:
        raise RuntimeError('Commit the reviewed source first so the download tag has a definite source revision.')
    token = credential()
    try:
        release = request(API + '/releases/tags/' + TAG, token=token)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise RuntimeError('GitHub release lookup failed with HTTP ' + str(error.code)) from None
        release = request(API + '/releases', 'POST', token, payload={
            'tag_name': TAG, 'target_commitish': commit, 'name': 'Team 09 submission review files',
            'draft': False, 'prerelease': True, 'make_latest': 'false',
            'body': 'Report PDF and Canvas review ZIP for Denisha Ketan Tank and Shibin Biji Thomas Lnu.\n\n'
                    'This is a review package. Real human ratings, original selected Part 1 execution files, '
                    'partner final raw artifacts and other audit items remain pending. Fictional rating '
                    'examples are format guides only and are not included in the Canvas ZIP or any human score.\n\n'
                    'Source commit: `' + commit + '`.\n\n'
                    '[Read the requirement audit](https://github.com/' + REPOSITORY + '/blob/' + commit + '/SUBMISSION_STATUS.md). '
                    'The required final part tags are separate and are not certified by this review release.'})
    if not release.get('prerelease') or release.get('target_commitish') != commit:
        raise RuntimeError('An existing review release points to a different revision. Nothing was overwritten.')
    uploaded = []
    existing = {a['name']: a for a in release.get('assets', [])}
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name in existing:
            asset = existing[path.name]
            if asset.get('digest') != 'sha256:' + digest:
                raise RuntimeError('A different asset already exists; preserve it and choose a new review version.')
        else:
            upload_url = release['upload_url'].split('{', 1)[0] + '?' + urllib.parse.urlencode({'name': path.name})
            if not upload_url.startswith('https://uploads.github.com/repos/' + REPOSITORY + '/releases/'):
                raise RuntimeError('Upload destination does not match the requested repository')
            asset = request(upload_url, 'POST', token, raw=path.read_bytes(),
                content_type='application/pdf' if path.suffix == '.pdf' else 'application/zip')
        if asset.get('state') != 'uploaded' or asset.get('size') != path.stat().st_size:
            raise RuntimeError('GitHub did not confirm the full asset upload')
        if asset.get('digest') and asset['digest'] != 'sha256:' + digest:
            raise RuntimeError('Uploaded digest mismatch')
        uploaded.append(dict(name=path.name, url=asset['browser_download_url'], bytes=asset['size'],
                             sha256=digest, github_asset_id=asset['id']))
    record = dict(review_release=release['html_url'], review_tag=TAG, source_commit=commit,
                  final_submission_ready=False, assets=uploaded)
    (ROOT / 'submission/github_downloads.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Published review downloads:', record['review_release'])
    for asset in uploaded:
        print(asset['name'], asset['url'], 'SHA256', asset['sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true', required=True,
                        help='Upload the current report and Canvas review package to the requested GitHub repository.')
    parser.parse_args()
    publish()
