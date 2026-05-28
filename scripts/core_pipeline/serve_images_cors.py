from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

import os

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    target_dir = "/data/brhanu/thesis_project/cvat_upload_temp/obj_train_data"
    os.chdir(target_dir)
    httpd = HTTPServer(('0.0.0.0', port), CORSRequestHandler)
    print(f"Serving {target_dir} at http://localhost:{port} with CORS enabled")
    httpd.serve_forever()
