"""Create blank real-rating forms and clearly synthetic format examples, never scores."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def write_if_absent(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(rows,indent=2)+'\n')


def prepare():
    golden=json.loads((ROOT/'part1_rag/eval/golden_candidate.json').read_text())
    forms=ROOT/'docs/rating_forms'
    for system in ['dense_baseline','hybrid_multimodal']:
        rows=[dict(id=r['id'],member_a=None,member_b=None,member_a_comment='',member_b_comment='') for r in golden]
        write_if_absent(forms/f'part1_{system}_blank.json',rows)
    for role in ['member_a','member_b','external_rater']:
        rows=[dict(rater_id='',role='external' if role=='external_rater' else 'team_member',pair_id=i,
                   domain_style=None,prompt_match=None,comment='') for i in range(25)]
        write_if_absent(forms/f'part2_{role}_blank.json',rows)
    scenarios=json.loads((ROOT/'part3_agents/eval/scenarios.json').read_text())
    rows=[dict(system=system,scenario_id=s['id'],rubric=s['rubric'],member_a=None,member_b=None,
               member_a_comment='',member_b_comment='',resolved_pass=None,resolution_note='')
          for system in ['baseline','critique'] for s in scenarios]
    write_if_absent(forms/'part3_blank.json',rows)
    examples=dict(sample_only=True,do_not_submit=True,
        explanation='These invented format examples are not observations or human-evaluation results. '
                    'They are intentionally outside the machine-read ratings schema.',
        part1_format=dict(id='EXAMPLE_ONLY',member_a=4,member_b=5,
            member_a_comment='Example only: a 4 would mean the answer is mostly supported but needs a small citation correction.'),
        part2_format=dict(sample_only=True,rater_id='EXAMPLE_ONLY',role='external',pair_id=0,
            domain_style='A',prompt_match='tie',comment='Example only: choose A for style only if it actually matches the domain better; use tie if both match the request equally.'),
        part3_format=dict(scenario_id='EXAMPLE_ONLY',system='critique',member_a=True,member_b=False,
            resolved_pass=False,resolution_note='Example only: after discussing the real rubric, the members could agree that a requested edit was not visible.'))
    write_if_absent(forms/'sample_formats_DO_NOT_SUBMIT.json',examples)
    print('Prepared blank forms:',forms)
    print('No observed score or human judgment has been generated. The synthetic example is not an input rating file.')
    return forms


def install(source,target):
    """Merge completed Part 2 ballots after people have supplied their own anonymous IDs."""
    rows=[]
    for path in source:
        value=json.loads(Path(path).read_text())
        if not isinstance(value,list) or any(r.get('sample_only') is True for r in value):
            raise ValueError('Use completed real ballot lists, not the sample-format object.')
        rows.extend(value)
    path=Path(target)
    if path.exists():
        raise FileExistsError('The target ratings file already exists. Review it manually; no ballots were overwritten.')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--merge-part2',nargs='+')
    parser.add_argument('--target')
    args=parser.parse_args()
    if args.merge_part2:
        if not args.target:
            parser.error('--target is required for a merge.')
        install(args.merge_part2,args.target)
    else:
        prepare()
