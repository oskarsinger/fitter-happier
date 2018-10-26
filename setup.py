from distutils.core import setup

packages = [
    '',
    'utils',
    'stepsize',
    'matrix',
    'qn',
    'distributed',
    'coordinate',
    'conjugate',
    'testers',
    'testers.aide',
    'testers.fsvrg'
]

setup(
    name='fitterhappier',
    version='0.01',
    packages=['fitterhappier.' + p for p in packages])
