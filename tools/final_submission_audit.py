"""Check the selected evidence. Missing real-world judgments stay pending."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(name, default=None):
    path = ROOT / name
    return json.loads(path.read_text()) if path.is_file() else default


def notebook(name):
    path = ROOT / name
    if not path.is_file():
        return dict(available=False, complete=False)
    cells = [c for c in json.loads(path.read_text())['cells'] if c['cell_type'] == 'code']
    executed = sum(c.get('execution_count') is not None for c in cells)
    errors = sum(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', []))
    return dict(available=True, code_cells=len(cells), executed_code_cells=executed,
                saved_error_outputs=errors, complete=bool(cells) and executed == len(cells) and errors == 0,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def collect(write=True):
    p1 = read('part1_rag/selected_run.json', {})
    p2 = read('part2_diffusion/evidence/current_result_audit.json', {})
    prefix = 'part3_agents/evidence/denisha_baseline/'
    p3 = read(prefix + 'baseline_outputs/baseline_metrics.json', {})
    identity = read(prefix + 'baseline_outputs/runtime_identity.json', {})
    status = read(prefix + 'baseline_outputs/execution_status.json', {})
    imported = read(prefix + 'IMPORT.json', {})
    partner = read('part3_agents/partner_snapshot/reported_results/p3_results.json', {})
    flags = read('team_review.json', {})
    details = read('part2_diffusion/submission_details.json', {})
    notebooks = {k: notebook(v) for k, v in {
        'part1': 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb',
        'part2': 'part2_diffusion/notebooks/Part2_FineTune_Run.ipynb',
        'part3_baseline': 'part3_agents/notebooks/Part3_Baseline_Executed.ipynb',
        'part3_partner': 'part3_agents/partner_snapshot/Partner_Part3_Executed.ipynb'}.items()}
    match = bool(imported.get('files')) and all((ROOT / n).is_file() and
        hashlib.sha256((ROOT / n).read_bytes()).hexdigest() == h for n, h in imported.get('files', {}).items())
    tags = subprocess.run(['git', 'tag', '--list'], cwd=ROOT, text=True, capture_output=True).stdout.splitlines()
    gates = {
        'part1_original_selected_execution': p1.get('original_executed_v6_outputs_available_in_this_checkout') is True and notebooks['part1']['complete'],
        'part1_team_reference_review': flags.get('part1_references_reviewed_by_team') is True,
        'part1_real_groundedness_ratings': flags.get('part1_real_ratings_verified') is True,
        'part2_computation': notebooks['part2']['complete'] and all(p2.get('checks', {}).get(k) is True for k in
            ['run1_cuda_training', 'run2_cuda_training', '120_images_same_protocol', 'kid_clip_lpips_for_all_models']),
        'part2_three_real_raters': p2.get('checks', {}).get('25_blind_pairs_3_real_raters') is True,
        'part2_rights_and_grader_access': all(p2.get('checks', {}).get(k) is True for k in
            ['artifact_access_for_graders_verified', 'license_and_redistribution_reviewed']),
        'part3_real_t4_baseline': match and notebooks['part3_baseline']['complete'] and status.get('completed') is True
            and status.get('mock_only') is False and p3.get('real_model_execution') is True and identity.get('gpu') == 'Tesla T4',
        'part3_partner_paired_run': notebooks['part3_partner']['complete'] and len(partner.get('table', [])) == 2,
        'part3_real_human_ratings': flags.get('part3_real_ratings_verified') is True,
        'part3_hub_artifacts_and_access': flags.get('part3_hub_artifacts_and_access_verified') is True,
        'team_identity': details.get('team_number') == '09' and details.get('partner_name') == 'Shibin Biji Thomas Lnu',
        'required_repository_name_and_privacy': False,
        'recordings_and_contribution_review': flags.get('contributions_and_recordings_verified') is True,
        'final_kaggle_selection_and_team': flags.get('kaggle_final_submission_selected') is True,
        'single_16gb_all_workloads': flags.get('single_16gb_execution_evidence_reviewed') is True,
        'report_reviewed_by_both_members': flags.get('combined_report_reviewed') is True,
        'three_final_tags': all(t in tags for t in ['p1-v1.0', 'p2-v1.0', 'p3-v1.0']),
        'live_demo_rehearsed': flags.get('demo_rehearsal_confirmed') is True}
    # Each row covers a rubric group, including limits that a file-existence check cannot settle.
    requirements = [
        ['1', 'Read-only three-PDF corpus, trust metadata, numbered image objects and vector index', 'Implemented in frozen v6; original execution missing'],
        ['1', 'Dense vs hybrid/BM25/CLIP/reranking, exact figures, revisions and trust precedence', 'Implemented in frozen v6; original execution missing'],
        ['1', 'Grounded JSON <=120 words, citations/statuses, session memory and no KB writes', 'Implemented; selected run validation evidence needs restoring'],
        ['1', '25 independent golden questions in required categories and team-reviewed references', 'Candidate set exists; genuine team review pending'],
        ['1', 'Correctness, Recall@5, citation accuracy, latency, groundedness and agreement', 'Original selected metric outputs and two real raters pending'],
        ['1', '40-row CSV, hidden QUESTIONS_PATH, <=90 minutes, final Kaggle selection', 'Pipeline implemented; original CSV/time record and final selection pending'],
        ['2', '500+ cleaned captioned images, seeded 90/10 split, 512-pixel training', 'Recorded: 1,221 Naruto images, 1,098 train / 123 test, seed 266'],
        ['2', 'Two comparable LoRA trainings, parameter/efficiency logs, adapters and model cards', 'Recorded: r4 and r8, 1,200 updates, RTX 4090; private Hub upload records'],
        ['2', '20 fixed prompts, two seeds, same settings, 120 images and eight comparison grids', 'Recorded in original executed notebook and result import'],
        ['2', 'KID/CLIP/LPIPS, selected model explanation and at least two failures', 'Recorded; noisy KID and safety-filtered output limits discussed'],
        ['2', '25 blinded pairs, both members plus outside rater, style/prompt win rates', 'Pending real ratings; blank forms and blinded images prepared'],
        ['2', '10-image memorization check, three closest pairs, safety and rights discussion', 'Measured neighbors/displayed pairs exist; rights/access confirmation pending'],
        ['3', 'Five agents, typed state, VQA/OCR, two generation modes, LoRA, critic and <=2 revisions', 'Partner executed implementation; Denisha baseline uses zero revisions'],
        ['3', '15 scenarios across five categories; 10 red-team cases; JSON request traces', 'Recorded T4 baseline: 25 states and 25 traces'],
        ['3', 'Without/with critique comparison, median latency and attack rate', 'Partner paired 4090 results available; T4 timing is a separate experiment'],
        ['3', 'Human task success, critic agreement, independent ratings and resolved disagreements', 'Pending real ratings; automatic 15/15 is not a human score'],
        ['3', 'Gradio demo, system card and HF input/output image dataset', 'Interface/card exist; artifact upload/access and demo confirmation pending'],
        ['Common', 'Team names, contributions, AI disclosure and knowledge-transfer recordings', 'Names and records added; actual recording links and member review pending'],
        ['Common', 'Private data266-lab2-team09 repo with grader access', 'Differs: user requested public Lab-2 without renaming'],
        ['Common', 'One nominal 16-GB GPU for every required workload', 'Part 3 baseline verified on T4; Part 2 training remains measured on 24-GB 4090'],
        ['Common', 'IEEE-format combined report, diagrams, results, analysis and safety per part', 'Local PDF prepared; missing ratings marked pending, team review required'],
        ['Common', 'p1-v1.0 / p2-v1.0 / p3-v1.0 tags, exact Canvas ZIP, links and final submission', 'Review ZIP prepared locally; final evidence, tags and external confirmations pending']]
    result = dict(ready=all(gates.values()), checks=gates, pending=[k for k, v in gates.items() if not v],
                  requirements=requirements, part1=p1, part2=p2, part3=p3, partner_part3=partner,
                  notebooks=notebooks, warning='No execution, human rating, recording, access or submission is invented.')
    if write:
        target = ROOT / 'submission/audit.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + '\n')
        lines = ['# Submission audit · Team 09', '', '**Not yet ready for final submission.**', '',
                 "Shibin's final executed notebook is present, but his individual final images and per-request traces were not included in the Git snapshot. Obtain his Part 3 output-folder ZIP; exporting existing files needs no new GPU run. Denisha's complete baseline artifacts are staged locally for Hub upload.", '',
                 '| Part | Requirement | Current evidence / remaining work |', '| :--- | :--- | :--- |']
        lines += ['| ' + ' | '.join(r) + ' |' for r in requirements]
        lines += ['', '## Final gates still open', ''] + ['* ' + k.replace('_', ' ') for k in result['pending']]
        (ROOT / 'SUBMISSION_STATUS.md').write_text('\n'.join(lines) + '\n')
    return result


if __name__ == '__main__':
    result = collect()
    print('Ready for submission:', result['ready'])
    print('Completed evidence gates:', sum(result['checks'].values()), '/', len(result['checks']))
    for item in result['pending']:
        print('PENDING:', item)
