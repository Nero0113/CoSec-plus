
ALL_VUL_TYPES = ['cwe-089', 'cwe-125', 'cwe-078', 'cwe-476', 'cwe-416', 'cwe-022', 'cwe-787', 'cwe-079', 'cwe-190']


VAL_SCENARIOS = {
    ('cwe-078', '2-py'),
    ('cwe-089', '2-py'),
    ('cwe-125', '2-c'),
    ('cwe-190', '2-c'),
    ('cwe-022', '2-py'),
    ('cwe-787', '2-c'),
}

TEST_SCENARIOS = [
    ['cwe-089', '0-py', '1-py'],
    ['cwe-125', '0-c', '1-c'],
    ['cwe-078', '0-py', '1-py'],
    ['cwe-476', '0-c', '2-c'],
    ['cwe-416', '0-c'],
    ['cwe-022', '0-py', '1-py'],
    ['cwe-787', '0-c', '1-c'],
    ['cwe-079', '0-py', '1-py'],
    ['cwe-190', '0-c', '1-c'],
    ['cwe-416', '1-c'],
]

NOTTRAINED_VUL_TYPES = ['cwe-119', 'cwe-502', 'cwe-732', 'cwe-798']
# NOTTRAINED_VUL_TYPES_NEW = ['cwe-020', 'cwe-094', 'cwe-117', 'cwe-209','cwe-215', 'cwe-312', 'cwe-643', 'cwe-777']
NOTTRAINED_VUL_TYPES_NEW = ['cwe-020', 'cwe-094', 'cwe-117', 'cwe-209','cwe-215', 'cwe-312', 'cwe-643', 'cwe-777', 'cwe-918']

DOP_VUL_TYPES = ['cwe-089']

DOP_SCENARIOS = [
    ('cwe-089', 'con'),
    ('cwe-089', 'm-1'),
    ('cwe-089', 'm-2'),
    ('cwe-089', 'm-3'),
    ('cwe-089', 'm-4'),
    ('cwe-089', 'd-1'),
    ('cwe-089', 'd-2'),
    ('cwe-089', 'd-3'),
    ('cwe-089', 'd-4'),
    ('cwe-089', 'd-5'),
    ('cwe-089', 'd-6'),
    ('cwe-089', 'd-7'),
    ('cwe-089', 'c-1'),
    ('cwe-089', 'c-2'),
    ('cwe-089', 'c-3'),
    ('cwe-089', 'c-4'),
    ('cwe-089', 'c-5'),
]