"""Separate model decisions from actual human rubric judgments and report missing data."""
import argparse
import json
from pathlib import Path
import statistics


def evaluate(folder):
    folder=Path(folder)
    scenarios=json.loads((Path(__file__).resolve().parents[1]/'eval/scenarios.json').read_text())
    systems={name:json.loads((folder/file).read_text()) for name,file in
             [('baseline','scenario_traces_no_critique.json'),('critique','scenario_traces.json')]}
    for rows in systems.values():
        if len(rows)!=15 or {r['scenario_id'] for r in rows}!={s['id'] for s in scenarios}:
            raise ValueError('Need exactly the 15 specified scenarios per system.')
    rating_path=folder/'human_ratings.json'
    ratings=json.loads(rating_path.read_text()) if rating_path.is_file() else []
    if len({(r['system'],r['scenario_id']) for r in ratings}) != len(ratings):
        raise ValueError('Duplicate scenario rating rows.')
    by_key={(r['system'],r['scenario_id']):r for r in ratings}
    template=[]
    for system in systems:
        for s in scenarios:
            template.append(dict(system=system,scenario_id=s['id'],rubric=s['rubric'],
                member_a=None,member_b=None,resolved_pass=None,resolution_note=''))
    if not (folder/'human_ratings_template.json').exists():
        (folder/'human_ratings_template.json').write_text(json.dumps(template,indent=2))
    resolved, agreements={},{'baseline':[],'critique':[]}
    for key,r in by_key.items():
        if key not in {(row['system'],row['scenario_id']) for row in template}:
            raise ValueError('A rating refers to an unknown system or scenario.')
        if type(r.get('member_a')) is not bool or type(r.get('member_b')) is not bool:
            continue
        a,b=r['member_a'],r['member_b']
        agreements[key[0]].append(a==b)
        if a==b:
            resolved[key]=a
        elif type(r.get('resolved_pass')) is bool and len(r.get('resolution_note','').strip())>=10:
            resolved[key]=r['resolved_pass']
    reds={system:json.loads((folder/name).read_text()) for system,name in
          [('baseline','red_team_results_no_critique.json'),('critique','red_team_results.json')]}
    full=all(json.loads((folder/file).read_text())['models_enabled'] for file in
             ['scenario_traces.metadata.json','scenario_traces_no_critique.metadata.json'])
    results=[]
    for system,rows in systems.items():
        red=reds[system]
        complete=all((system,r['scenario_id']) in resolved for r in rows)
        with_critic=[r for r in rows if any(t['agent']=='critique' for t in r['trace'])]
        critic_complete=all((system,r['scenario_id']) in resolved for r in with_critic)
        results.append(dict(system=system,task_success_rate=(sum(resolved[(system,r['scenario_id'])] for r in rows)/15) if complete and full else None,
            critic_agreement=(statistics.mean((r['decision']=='accept')==resolved[(system,r['scenario_id'])] for r in with_critic)) if with_critic and critic_complete and full else None,
            red_team_attack_success_rate=red.get('attack_success_rate') if full else None,
            human_agreement=statistics.mean(agreements[system]) if len(agreements[system])==15 else None,
            median_seconds_per_request=statistics.median(r['seconds'] for r in rows) if full else None,
            execution_errors=[dict(id=r['scenario_id'],error=r['error']) for r in rows if r['error']],
            automatic_accept_count=sum(r['decision']=='accept' for r in rows),
            note='Automatic accept count is not task success. Human measurements stay null until real independent judgments are supplied.'))
    result=dict(results=results,models_enabled=full,ready=full and len(resolved)==30 and all(red.get('image_based') is True
                and red.get('models_enabled') is True for red in reds.values()) and not any(r['execution_errors'] for r in results),
                pending=[] if len(resolved)==30 else ['Both members must judge baseline and final scenarios, then resolve disagreements.'])
    (folder/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',required=True)
    evaluate(parser.parse_args().folder)
