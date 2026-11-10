"""Honest cross-part readiness: execution, human evidence and common deliverables differ."""
import json
from pathlib import Path
import subprocess
import sys
import re
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'part2_diffusion/src'))
sys.path.insert(0,str(ROOT/'tools'))
from submission import audit,read_json
from review_measurements import groundedness


def collect():
    part2=audit()
    p1=ROOT/'part1_rag/outputs/v2/evaluation'
    p1_evaluation=read_json(p1/'evaluation.json',{})
    golden=read_json(ROOT/'part1_rag/eval/golden_candidate.json',[])
    golden_ids=p1_evaluation.get('golden_ids',[r['id'] for r in golden])
    human=groundedness(p1/'human_ratings_hybrid_multimodal.json',golden_ids)
    human_baseline=groundedness(p1/'human_ratings_dense_baseline.json',golden_ids)
    latest=read_json(ROOT/'part3_agents/outputs/v2/latest_run.json',{})
    p3=Path(latest['folder']) if latest.get('folder') else None
    p3_evaluation=read_json(p3/'evaluation.json',{}) if p3 else {}
    p3_links=read_json(p3/'hub_links.json',{}) if p3 else {}
    uploads_match=bool(p3 and p3_links.get('commit') and p3_links.get('images') and all(
        p3_links.get('trace_sha256',{}).get(system)==hashlib.sha256((p3/name).read_bytes()).hexdigest()
        for system,name in [('baseline','scenario_traces_no_critique.json'),('critique','scenario_traces.json')]
        if (p3/name).is_file()) and all((p3/name).is_file() for name in ['scenario_traces_no_critique.json','scenario_traces.json']))
    details=read_json(ROOT/'part2_diffusion/submission_details.json',{})
    team=read_json(ROOT/'team_review.json',{})
    try:
        remote=subprocess.run(['git','remote','get-url','origin'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
        repository=re.sub(r'^(?:https://github.com/|git@github.com:)','',remote).removesuffix('.git')
        result=subprocess.run(['gh','repo','view',repository,'--json','name,isPrivate'],
                              capture_output=True,text=True,timeout=10,check=True)
        github=json.loads(result.stdout)
    except (OSError,subprocess.SubprocessError,ValueError):
        github={}
    # A standalone Colab bundle is intentionally not a git checkout. Lack of
    # submission tags remains pending, without a misleading fatal git message.
    tags=(subprocess.run(['git','tag','--list'],cwd=ROOT,capture_output=True,text=True).stdout.splitlines()
          if (ROOT/'.git').exists() else [])
    repo_name=f"data266-lab2-team{details.get('team_number')}" if details.get('team_number') else None
    checks={
        'part1_model_evaluation_available': bool(p1_evaluation.get('results')),
        'part1_team_reviewed_golden_set': p1_evaluation.get('golden_all_team_verified') is True,
        'part1_two_real_groundedness_ratings': human['complete'] and human_baseline['complete'],
        'part2_evidence_complete': part2['ready'],
        'part3_model_evaluation_and_real_ratings_complete': p3_evaluation.get('ready') is True,
        'part3_generated_images_uploaded': uploads_match,
        'part3_image_dataset_grader_access': uploads_match and p3_links.get('grader_access_verified') is True,
        'part3_demo_rehearsal_confirmed': team.get('demo_rehearsal_confirmed') is True,
        'team_details_supplied': bool(details.get('partner_name') and details.get('team_number')),
        'private_team_repository_and_grader_access': bool(github.get('isPrivate') is True and repo_name
              and github.get('name')==repo_name and team.get('grader_access_confirmed') is True),
        'meeting_recordings_and_contributions_verified': team.get('contributions_and_recordings_verified') is True,
        'all_three_submission_tags': all(t in tags for t in ['p1-v1.0','p2-v1.0','p3-v1.0']),
        'final_kaggle_submission_selected': team.get('kaggle_final_submission_selected') is True,
        'single_16gb_execution_evidence_reviewed': team.get('single_16gb_execution_evidence_reviewed') is True,
        'final_combined_report_reviewed_by_both_members': team.get('combined_report_reviewed') is True}
    result=dict(ready=all(checks.values()),checks=checks,pending=[k for k,v in checks.items() if not v],
                part1=p1_evaluation,part1_human=human,part1_baseline_human=human_baseline,part2=part2,part3=p3_evaluation,
                github=github,scope='No missing human rating, GPU result, team contribution or external submission is imputed.')
    output=ROOT/'runtime_checks/final_submission_audit.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print('\nCross-part readiness:',result['ready'])
    for name in result['pending']:
        print('PENDING:',name)
    return result


if __name__=='__main__':
    collect()
