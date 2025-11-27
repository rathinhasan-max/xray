import sys
import sys
import os

project_home = '/home/shopghor/exoticgames.store/xray_app'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.environ['FLASK_APP'] = 'app.py'

try:
    from app import app as application
except Exception as e:
    import traceback
    # If app fails to start, display the error in the browser
    def application(environ, start_response):
        status = '500 Internal Server Error'
        output = traceback.format_exc()
        response_headers = [('Content-type', 'text/plain'),
                            ('Content-Length', str(len(output)))]
        start_response(status, response_headers)
        return [output.encode('utf-8')]