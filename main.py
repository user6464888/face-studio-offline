"""Public release entry: camera-free demo by default, explicit local reconstruction."""
import argparse
import importlib.util
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'head_reconstruction'))


def check_demo():
    modules=['numpy','scipy','trimesh','open3d','networkx']
    missing=[name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing demo dependencies: '+', '.join(missing)+'. Run: python -m pip install -r requirements-demo.txt')
    from importlib.metadata import version
    print('Python:',sys.version.split()[0])
    for name in modules:print(name+':',version(name))


def main(argv=None):
    parser=argparse.ArgumentParser(description='Face Studio: 19 procedural hairstyles, offline after installation.')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--demo',action='store_true',help='Camera-free procedural mannequin demo (default)')
    mode.add_argument('--reconstruct',action='store_true',help='Use your own primary and top videos; optional dependencies required')
    parser.add_argument('--primary',type=Path)
    parser.add_argument('--top',type=Path)
    parser.add_argument('--force',action='store_true',help='Recompute reconstruction into a new run')
    parser.add_argument('--port',type=int,default=8770)
    parser.add_argument('--no-open',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--check',action='store_true',help='Check dependencies without generating assets or starting a server')
    args=parser.parse_args(argv)
    if not 1<=args.port<=65535:parser.error('--port must be between 1 and 65535')
    if not args.reconstruct and (args.primary or args.top or args.force):parser.error('Video inputs and --force require --reconstruct')
    if args.reconstruct and (not args.primary or not args.top):parser.error('--reconstruct requires --primary and --top')
    check_demo()
    if args.reconstruct:
        from reconstruction_check import check
        check(args.primary,args.top)
        if args.check:return
        from salon.pipeline import run
        run(args.primary,args.top,force=args.force,prebuild=False)
    else:
        if args.check:return
        from salon.demo import prepare
        prepare()
    if not args.prepare_only:
        from salon.server import serve
        serve(args.port,open_browser=not args.no_open)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    try:main()
    except KeyboardInterrupt:pass
    except Exception as exc:
        print('Unable to start: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
