"""Open already-prepared local state without resetting the demo."""
import argparse
from salon.geometry import ROOT
from salon.server import serve

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8770);parser.add_argument('--no-open',action='store_true')
    args=parser.parse_args()
    if not all((ROOT/name).is_file() for name in ['heads.json','catalog.json','workflow.json']):parser.error('No prepared state. Run main.py or salon.prepare first.')
    serve(args.port,open_browser=not args.no_open)
