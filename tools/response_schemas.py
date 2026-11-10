"""Output shapes only: no reference answers or question-specific routing rules."""


def text(limit=400, minimum=0):
    return {'type': 'string', 'minLength': minimum, 'maxLength': limit}


def record(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def choices(values, maximum=5):
    values = sorted(set(values))
    return {'type': 'array', 'items': {'type': 'string', 'enum': values} if values else text(),
            'maxItems': min(maximum, len(values))}


def rag_schema(hits):
    citations = {r['citation'] for r in hits} | {f"{r['doc_id']}:p{r['page']}" for r in hits}
    figures = {r['figure_id'] for r in hits if r.get('figure_id')}
    return record({
        'status': {'type': 'string', 'enum': ['answered', 'not_found', 'conflict',
                                              'false_premise', 'session_override']},
        'answer': text(600, 1),
        'citations': choices(citations),
        'figure_references': choices(figures),
    })


VISION_SCHEMA = record({
    'unsafe': {'type': 'boolean'},
    'caption': text(300, 1),
    'answer': text(300, 1),
    'objects': {'type': 'array', 'items': text(80, 1), 'maxItems': 8},
    'visible_text': text(400),
    'visual_questions': record({'What main objects are visible?': text(160, 1),
                                'Where is the main subject?': text(160, 1)}),
    'uncertainty': text(160),
})
PROMPT_SCHEMA = record({'prompt': text(600, 1)})
CRITIQUE_SCHEMA = record({'safe': {'type': 'boolean'}, 'accept': {'type': 'boolean'},
                          'judgment': text(300, 1), 'revision_instruction': text(300),
                          'visible_text': text(300)})


def citation_judge_schema(citations):
    # One required key per actual citation avoids duplicate or missing judgments.
    return record({'judgments': record({source: record({'supported': {'type': 'boolean'},
                                                       'reason': text(100, 1)})
                                        for source in sorted(set(citations))})})
