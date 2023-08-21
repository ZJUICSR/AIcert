#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/03/22, ZJUICSR'
import argparse
import os
from flask import Flask
from jinja2.sandbox import SandboxedEnvironment
from werkzeug.routing import BaseConverter
import threading
from IOtool import IOtool
# import utils.functional as F
__version__ = "0.1.0"


class SandboxedBaseEnvironment(SandboxedEnvironment):
    """SandboxEnvironment that mimics the Flask BaseEnvironment"""
    def __init__(self, app, **options):
        if 'loader' not in options:
            options['loader'] = app.create_global_jinja_loader()
        SandboxedEnvironment.__init__(self, **options)
        self.app = app


class XAI_Flask(Flask):
    def __init__(self, *args, **kwargs):
        """Overriden Jinja constructor setting a custom jinja_environment"""
        self.jinja_environment = SandboxedBaseEnvironment
        Flask.__init__(self, *args, **kwargs)

    def create_jinja_environment(self):
        """Overridden jinja environment constructor"""
        return super(XAI_Flask, self).create_jinja_environment()


class RegexConverter(BaseConverter):
    def __init__(self, url_map, *items):
        super(RegexConverter, self).__init__(url_map)
        self.regex = items[0]


# def create_app(config="config.develop.Config"):
#     app = XAI_Flask(__name__, static_folder="web/static")
#     app.VERSION = __version__
#     print("-> Running at version={:s}".format(str(app.VERSION)))
#     with app.app_context():
#         app.config.from_object(config)

#         # cache & log & redis
#         F.init(app)

#         # init some utils
#         F.init_utils(app)

#         # register controller
#         from web.view.api import api
#         app.register_blueprint(api)
#     return app


# def main_api():
#     if os.environ.get("SecAladdin_STABLE"):
#         conf = "config.stable.Config"
#     else:
#         conf = "config.develop.Config"
#     app = create_app(config=conf)
#     app.run(debug=True, threaded=True, host="127.0.0.1", port=2000) #

def main_index():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str,default='0.0.0.0',help='ip')
    parser.add_argument('--port',type=int,default='14130',help='port')
    parser.add_argument('--debug',type=bool,default=True,help='debugifopen')
    args = parser.parse_args()
    from web.index import app_run
    
    app = app_run(args)
    # app.run(debug=True, threaded=True, host=args.host, port=args.port)

if __name__ == "__main__":
    t2 = threading.Thread(target=IOtool.check_sub_task_threading)
    t2.setDaemon(True)
    t2.start()
    main_index()
