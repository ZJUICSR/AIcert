#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/03/22'
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str,default='0.0.0.0',help='ip')
    parser.add_argument('--port',type=int,default='5500',help='port')
    parser.add_argument('--debug',type=bool,default=True,help='debugifopen')
    args = parser.parse_args()
    from web.index import app_run
    app = app_run(args)
    # app.run(debug=True, threaded=True, host=args.host, port=args.port)

if __name__ == "__main__":
    main()