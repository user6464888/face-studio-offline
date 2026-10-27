"""Loopback-only viewer with an explicit static-file allowlist."""
import json
import re
import threading
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit,parse_qs,unquote
from pathlib import Path
from salon.catalog import STYLES
from salon.geometry import ROOT,RECON,VERSION,style_assets

MODEL_LOCK=threading.Lock()
STATIC={'index.html','studio.js','studio.css','viewer.js','hybrid_material.js'}
VENDOR={'three.module.js','three.core.js','OrbitControls.js','PLYLoader.js'}


def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(RECON),**kwargs)

    def reply(self,data,status=200):
        body=json.dumps(data,ensure_ascii=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def allowed_static(self,path):
        path=unquote(path)
        if '\\' in path or '..' in Path(path).parts:return False
        if path.startswith('/salon/') and path.removeprefix('/salon/') in STATIC:return True
        if path.startswith('/viewer/vendor/') and path.removeprefix('/viewer/vendor/') in VENDOR:return True
        if re.fullmatch(r'/salon/cache/[a-z0-9_]+\.ply',path):return True
        if path.startswith('/salon/assets/'):
            return any(path=='/salon/'+(h.get(field) or '') for h in load('heads.json')['heads'] for field in ['bald_file','original_file','texture_file','texture_faces'])
        return False

    def do_HEAD(self):
        if not self.allowed_static(urlsplit(self.path).path):self.send_error(404);return
        super().do_HEAD()

    def do_GET(self):
        path=urlsplit(self.path).path
        try:
            if path=='/':self.send_response(302);self.send_header('Location','/salon/index.html');self.end_headers();return
            if path=='/api/salon/health':self.reply({'ready':True,'styles':len(STYLES),'geometry_version':VERSION});return
            if path=='/api/salon/state':
                from salon.face_advice import recommendations
                from salon.catalog import FACE_SHAPES
                self.reply({'workflow':load('workflow.json'),'catalog':load('catalog.json'),'heads':load('heads.json')['heads'],
                            'advice':{face:recommendations(face) for face in FACE_SHAPES}});return
            if path=='/api/salon/style':
                args=parse_qs(urlsplit(self.path).query)
                head=next((h for h in load('heads.json')['heads'] if h['id']==args.get('head',[''])[0]),None)
                style=next((s for s in STYLES if s['id']==args.get('style',[''])[0]),None)
                if head is None or style is None:self.reply({'error':'Unknown head or style'},400);return
                with MODEL_LOCK:result=style_assets(head,style)
                self.reply(result);return
            if not self.allowed_static(path):self.send_error(404);return
            super().do_GET()
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as exc:self.reply({'error':str(exc)},500)


def serve(port=8770,open_browser=False):
    import webbrowser
    try:server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    except OSError as exc:raise RuntimeError(f'Port {port} is unavailable. Try --port {port+1}.') from exc
    address=f'http://127.0.0.1:{port}/salon/index.html'
    print('Preview: '+address,flush=True)
    if open_browser:webbrowser.open(address)
    try:server.serve_forever()
    finally:server.server_close()
